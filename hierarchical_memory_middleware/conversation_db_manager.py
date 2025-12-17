"""Per-conversation DuckDB manager for concurrent conversation handling.

This module provides isolated DuckDB files per active conversation to avoid
lock contention when multiple conversations are happening simultaneously
(e.g., in Slack with multiple threads).

Lifecycle:
1. On startup: Scan for orphaned active conversation files → merge to main (healing)
2. On conversation start: Extract from main → create active file
3. During conversation: All ops go to isolated active file
4. On idle (5 min) / shutdown: Sync back to main → delete active file
"""

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

import duckdb

from .db_utils import _init_schema, get_db_connection
from .storage import DuckDBStorage

logger = logging.getLogger(__name__)


@dataclass
class ActiveConversationDB:
    """Tracks an active conversation's isolated database."""

    storage: DuckDBStorage
    temp_path: Path
    last_activity: datetime = field(default_factory=datetime.now)
    dirty: bool = False  # True if there are unsaved changes


class ConversationDBManager:
    """Manages per-conversation DuckDB files with sync-back to main DB.

    This manager provides:
    - Isolated DuckDB file per active conversation (zero lock contention)
    - Automatic extraction from main DB when conversation starts
    - Periodic sync-back to main DB after inactivity
    - Startup healing for orphaned active files from bad shutdowns
    """

    def __init__(
        self,
        main_db_path: str,
        active_dir: Optional[str] = None,
        idle_timeout: timedelta = timedelta(minutes=5),
        cleanup_interval: timedelta = timedelta(minutes=1),
    ):
        """Initialize the conversation DB manager.

        Args:
            main_db_path: Path to the main DuckDB file (cold storage)
            active_dir: Directory for active conversation files (default: alongside main DB)
            idle_timeout: Time after which idle conversations are synced back
            cleanup_interval: How often to check for idle conversations
        """
        self.main_db_path = Path(main_db_path)
        self.active_dir = Path(active_dir) if active_dir else self.main_db_path.parent / "active_conversations"
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval

        self.active_dbs: Dict[str, ActiveConversationDB] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Ensure directories exist
        self.active_dir.mkdir(parents=True, exist_ok=True)

        # Ensure main DB exists and has schema
        if not self.main_db_path.exists():
            with get_db_connection(str(self.main_db_path), init_schema=True):
                pass  # Just create the schema

    async def startup(self) -> int:
        """Run startup healing to merge any orphaned active files.

        Call this when the application starts to handle any active conversation
        files left over from a bad shutdown.

        Returns:
            Number of orphaned files that were merged
        """
        merged_count = 0

        # Find all .duckdb files in active directory
        orphaned_files = list(self.active_dir.glob("*.duckdb"))

        if orphaned_files:
            logger.info(f"Found {len(orphaned_files)} orphaned active conversation files, merging...")

        for orphan_path in orphaned_files:
            conversation_id = orphan_path.stem  # filename without .duckdb
            try:
                logger.info(f"Merging orphaned file for conversation: {conversation_id}")
                await self._merge_to_main(conversation_id, orphan_path)
                orphan_path.unlink()  # Delete the orphaned file
                merged_count += 1
                logger.info(f"Successfully merged and cleaned up: {conversation_id}")
            except Exception as e:
                logger.error(f"Failed to merge orphaned file {orphan_path}: {e}")
                # Keep the file for manual recovery
                backup_path = orphan_path.with_suffix(".duckdb.failed")
                shutil.move(str(orphan_path), str(backup_path))
                logger.warning(f"Moved failed file to: {backup_path}")

        # Start the cleanup task
        self._start_cleanup_task()

        return merged_count

    async def shutdown(self) -> None:
        """Gracefully shutdown, syncing all active conversations back to main.

        Call this before application exit to ensure all data is persisted.
        """
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Sync all active conversations
        async with self._lock:
            for conv_id in list(self.active_dbs.keys()):
                try:
                    await self._sync_and_close(conv_id)
                except Exception as e:
                    logger.error(f"Error syncing conversation {conv_id} during shutdown: {e}")

    async def get_storage(self, conversation_id: str) -> DuckDBStorage:
        """Get isolated storage for a conversation.

        If the conversation is not already active, extracts its data from main
        DB to a new isolated file.

        Args:
            conversation_id: The conversation ID

        Returns:
            DuckDBStorage instance for the isolated conversation DB
        """
        async with self._lock:
            if conversation_id not in self.active_dbs:
                await self._activate_conversation(conversation_id)

            active = self.active_dbs[conversation_id]
            active.last_activity = datetime.now()
            return active.storage

    def mark_dirty(self, conversation_id: str) -> None:
        """Mark a conversation as having unsaved changes.

        Call this after any write operation to ensure the conversation
        gets synced back to main.
        """
        if conversation_id in self.active_dbs:
            self.active_dbs[conversation_id].dirty = True

    async def force_sync(self, conversation_id: str) -> bool:
        """Force immediate sync of a conversation back to main.

        Args:
            conversation_id: The conversation to sync

        Returns:
            True if sync was successful
        """
        async with self._lock:
            if conversation_id in self.active_dbs:
                active = self.active_dbs[conversation_id]
                if active.dirty:
                    await self._merge_to_main(conversation_id, active.temp_path)
                    active.dirty = False
                    return True
        return False

    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background task to sync idle conversations back to main."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self._cleanup_idle_conversations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_idle_conversations(self) -> None:
        """Check for and sync idle conversations."""
        now = datetime.now()
        to_cleanup = []

        async with self._lock:
            for conv_id, active in self.active_dbs.items():
                if now - active.last_activity > self.idle_timeout:
                    to_cleanup.append(conv_id)

            for conv_id in to_cleanup:
                try:
                    logger.info(f"Syncing idle conversation: {conv_id}")
                    await self._sync_and_close(conv_id)
                except Exception as e:
                    logger.error(f"Error syncing idle conversation {conv_id}: {e}")

    async def _sync_and_close(self, conversation_id: str) -> None:
        """Sync a conversation back to main and close it.

        Must be called with self._lock held.
        """
        if conversation_id not in self.active_dbs:
            return

        active = self.active_dbs[conversation_id]

        # Only sync if dirty
        if active.dirty:
            await self._merge_to_main(conversation_id, active.temp_path)

        # Close the storage
        active.storage.close()

        # Delete the temp file
        try:
            active.temp_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temp file {active.temp_path}: {e}")

        # Remove from active dict
        del self.active_dbs[conversation_id]
        logger.debug(f"Closed active conversation: {conversation_id}")

    async def _activate_conversation(self, conversation_id: str) -> None:
        """Extract a conversation from main DB to active file.

        Must be called with self._lock held.
        """
        temp_path = self.active_dir / f"{conversation_id}.duckdb"

        # Create the isolated DB and initialize schema
        with get_db_connection(str(temp_path), init_schema=True) as active_conn:
            # Extract data from main DB
            with get_db_connection(str(self.main_db_path), init_schema=False) as main_conn:
                self._extract_conversation_data(main_conn, active_conn, conversation_id)

        # Create storage instance
        storage = DuckDBStorage(str(temp_path), enable_semantic_search=True)

        self.active_dbs[conversation_id] = ActiveConversationDB(
            storage=storage,
            temp_path=temp_path,
            last_activity=datetime.now(),
            dirty=False,
        )

        logger.debug(f"Activated conversation: {conversation_id}")

    def _extract_conversation_data(
        self,
        main_conn: duckdb.DuckDBPyConnection,
        active_conn: duckdb.DuckDBPyConnection,
        conversation_id: str,
    ) -> None:
        """Extract all data for a conversation from main to active DB."""
        # Check if conversation exists in main
        conv_exists = main_conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()

        if not conv_exists:
            logger.debug(f"Conversation {conversation_id} is new, no data to extract")
            return

        # Extract conversations table
        conv_data = main_conn.execute(
            """
            SELECT id, name, total_nodes, compression_stats, current_goal,
                   key_decisions, system_prompt, created_at, last_updated
            FROM conversations WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if conv_data:
            active_conn.execute(
                """
                INSERT INTO conversations (id, name, total_nodes, compression_stats,
                    current_goal, key_decisions, system_prompt, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                conv_data,
            )

        # Extract nodes table (all columns)
        nodes = main_conn.execute(
            """
            SELECT node_id, conversation_id, node_type, content, timestamp,
                   sequence_number, line_count, level, summary, summary_metadata,
                   parent_summary_node_id, tokens_used, expandable, ai_components,
                   topics, embedding, relates_to_node_id
            FROM nodes WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()

        for node in nodes:
            # Handle embedding separately due to array type
            embedding = node[15]
            if embedding is not None:
                embedding_dim = len(embedding)
                embedding_sql = f"[{', '.join(str(x) for x in embedding)}]::FLOAT[{embedding_dim}]"
            else:
                embedding_sql = "NULL"

            active_conn.execute(
                f"""
                INSERT INTO nodes (node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable, ai_components,
                    topics, embedding, relates_to_node_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {embedding_sql}, ?)
                """,
                (*node[:15], node[16]),  # All fields except embedding (index 15)
            )

        # Extract token_usage table
        token_usage = main_conn.execute(
            """
            SELECT id, conversation_id, timestamp, input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens, total_tokens,
                   cost_usd, duration_ms, model
            FROM token_usage WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()

        for usage in token_usage:
            active_conn.execute(
                """
                INSERT INTO token_usage (id, conversation_id, timestamp, input_tokens,
                    output_tokens, cache_read_tokens, cache_creation_tokens, total_tokens,
                    cost_usd, duration_ms, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                usage,
            )

        # Advance sequence to be past the max id we inserted
        if token_usage:
            max_id = max(u[0] for u in token_usage)
            # Advance sequence to be past max_id
            for _ in range(max_id):
                try:
                    active_conn.execute("SELECT nextval('token_usage_seq')")
                except Exception:
                    break  # Sequence might already be past this point

        logger.debug(
            f"Extracted {len(nodes)} nodes and {len(token_usage)} token records "
            f"for conversation {conversation_id}"
        )

    async def _merge_to_main(
        self,
        conversation_id: str,
        active_path: Path,
    ) -> None:
        """Merge changes from active DB back to main DB.

        This performs a smart merge:
        - nodes: Compare by (node_id, conversation_id), keep record with latest timestamp
        - conversations: Take record with latest last_updated
        - token_usage: Insert any records from active that don't exist in main
        """
        with get_db_connection(str(active_path), init_schema=False) as active_conn:
            with get_db_connection(str(self.main_db_path), init_schema=True) as main_conn:
                await self._merge_conversations_table(active_conn, main_conn, conversation_id)
                await self._merge_nodes_table(active_conn, main_conn, conversation_id)
                await self._merge_token_usage_table(active_conn, main_conn, conversation_id)

        logger.info(f"Merged conversation {conversation_id} back to main DB")

    def _table_exists(self, conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            (table_name,),
        ).fetchone()
        return result[0] > 0

    async def _merge_conversations_table(
        self,
        active_conn: duckdb.DuckDBPyConnection,
        main_conn: duckdb.DuckDBPyConnection,
        conversation_id: str,
    ) -> None:
        """Merge conversations table - take record with latest last_updated."""
        # Check if table exists in active DB (may be corrupted/incomplete)
        if not self._table_exists(active_conn, "conversations"):
            logger.warning(f"conversations table missing in active DB for {conversation_id}")
            return

        active_conv = active_conn.execute(
            """
            SELECT id, name, total_nodes, compression_stats, current_goal,
                   key_decisions, system_prompt, created_at, last_updated
            FROM conversations WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if not active_conv:
            return

        main_conv = main_conn.execute(
            "SELECT last_updated FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if not main_conv:
            # Insert new conversation
            main_conn.execute(
                """
                INSERT INTO conversations (id, name, total_nodes, compression_stats,
                    current_goal, key_decisions, system_prompt, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                active_conv,
            )
        else:
            # Update if active is newer (or always update since active had the live data)
            main_conn.execute(
                """
                UPDATE conversations SET
                    name = ?, total_nodes = ?, compression_stats = ?,
                    current_goal = ?, key_decisions = ?, system_prompt = ?,
                    last_updated = ?
                WHERE id = ?
                """,
                (
                    active_conv[1],  # name
                    active_conv[2],  # total_nodes
                    active_conv[3],  # compression_stats
                    active_conv[4],  # current_goal
                    active_conv[5],  # key_decisions
                    active_conv[6],  # system_prompt
                    active_conv[8],  # last_updated
                    conversation_id,
                ),
            )

    async def _merge_nodes_table(
        self,
        active_conn: duckdb.DuckDBPyConnection,
        main_conn: duckdb.DuckDBPyConnection,
        conversation_id: str,
    ) -> None:
        """Merge nodes table - compare by (node_id, conversation_id).

        For each node in active:
        - If not in main: insert
        - If in main: compare timestamps, keep the one with latest timestamp
        """
        # Check if table exists in active DB (may be corrupted/incomplete)
        if not self._table_exists(active_conn, "nodes"):
            logger.warning(f"nodes table missing in active DB for {conversation_id}")
            return

        # Get all nodes from active
        active_nodes = active_conn.execute(
            """
            SELECT node_id, conversation_id, node_type, content, timestamp,
                   sequence_number, line_count, level, summary, summary_metadata,
                   parent_summary_node_id, tokens_used, expandable, ai_components,
                   topics, embedding, relates_to_node_id
            FROM nodes WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()

        # Get all node_ids and timestamps from main for this conversation
        main_nodes = main_conn.execute(
            "SELECT node_id, timestamp FROM nodes WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        main_node_map = {row[0]: row[1] for row in main_nodes}

        inserted = 0
        updated = 0

        for node in active_nodes:
            node_id = node[0]
            active_timestamp = node[4]
            embedding = node[15]

            # Handle embedding
            if embedding is not None:
                embedding_dim = len(embedding)
                embedding_sql = f"[{', '.join(str(x) for x in embedding)}]::FLOAT[{embedding_dim}]"
            else:
                embedding_sql = "NULL"

            if node_id not in main_node_map:
                # Insert new node
                main_conn.execute(
                    f"""
                    INSERT INTO nodes (node_id, conversation_id, node_type, content, timestamp,
                        sequence_number, line_count, level, summary, summary_metadata,
                        parent_summary_node_id, tokens_used, expandable, ai_components,
                        topics, embedding, relates_to_node_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {embedding_sql}, ?)
                    """,
                    (*node[:15], node[16]),
                )
                inserted += 1
            else:
                # Compare timestamps - update if active is newer or same (active had live edits)
                main_timestamp = main_node_map[node_id]
                if active_timestamp >= main_timestamp:
                    main_conn.execute(
                        f"""
                        UPDATE nodes SET
                            node_type = ?, content = ?, timestamp = ?, sequence_number = ?,
                            line_count = ?, level = ?, summary = ?, summary_metadata = ?,
                            parent_summary_node_id = ?, tokens_used = ?, expandable = ?,
                            ai_components = ?, topics = ?, embedding = {embedding_sql},
                            relates_to_node_id = ?
                        WHERE node_id = ? AND conversation_id = ?
                        """,
                        (
                            node[2],   # node_type
                            node[3],   # content
                            node[4],   # timestamp
                            node[5],   # sequence_number
                            node[6],   # line_count
                            node[7],   # level
                            node[8],   # summary
                            node[9],   # summary_metadata
                            node[10],  # parent_summary_node_id
                            node[11],  # tokens_used
                            node[12],  # expandable
                            node[13],  # ai_components
                            node[14],  # topics
                            node[16],  # relates_to_node_id
                            node_id,
                            conversation_id,
                        ),
                    )
                    updated += 1

        # Also check for nodes that exist in main but not in active (shouldn't happen normally)
        active_node_ids = {node[0] for node in active_nodes}
        for main_node_id in main_node_map:
            if main_node_id not in active_node_ids:
                logger.warning(
                    f"Node {main_node_id} exists in main but not in active for "
                    f"conversation {conversation_id} - keeping main version"
                )

        logger.debug(
            f"Nodes merge for {conversation_id}: {inserted} inserted, {updated} updated"
        )

    async def _merge_token_usage_table(
        self,
        active_conn: duckdb.DuckDBPyConnection,
        main_conn: duckdb.DuckDBPyConnection,
        conversation_id: str,
    ) -> None:
        """Merge token_usage table - insert records that don't exist in main.

        We compare by (conversation_id, timestamp, input_tokens, output_tokens)
        since the id might differ between databases.
        """
        # Check if table exists in active DB (may be corrupted/incomplete)
        if not self._table_exists(active_conn, "token_usage"):
            logger.warning(f"token_usage table missing in active DB for {conversation_id}")
            return

        # Get all token usage from active
        active_usage = active_conn.execute(
            """
            SELECT id, conversation_id, timestamp, input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens, total_tokens,
                   cost_usd, duration_ms, model
            FROM token_usage WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()

        # Get existing records from main (use timestamp + tokens as key)
        main_usage = main_conn.execute(
            """
            SELECT timestamp, input_tokens, output_tokens
            FROM token_usage WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()
        main_keys = {(row[0], row[1], row[2]) for row in main_usage}

        inserted = 0
        for usage in active_usage:
            key = (usage[2], usage[3], usage[4])  # timestamp, input_tokens, output_tokens
            if key not in main_keys:
                # Insert without specifying id (let it auto-generate)
                main_conn.execute(
                    """
                    INSERT INTO token_usage (conversation_id, timestamp, input_tokens,
                        output_tokens, cache_read_tokens, cache_creation_tokens,
                        total_tokens, cost_usd, duration_ms, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    usage[1:],  # Skip id
                )
                inserted += 1

        logger.debug(f"Token usage merge for {conversation_id}: {inserted} inserted")

    def get_active_conversations(self) -> Set[str]:
        """Get the set of currently active conversation IDs."""
        return set(self.active_dbs.keys())

    def is_active(self, conversation_id: str) -> bool:
        """Check if a conversation is currently active."""
        return conversation_id in self.active_dbs
