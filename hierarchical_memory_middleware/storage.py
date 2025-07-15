"""DuckDB storage layer for hierarchical memory system."""

import json
import re
from typing import List, Optional, Dict, Any, Tuple
import duckdb
from contextlib import contextmanager
from .db_utils import get_db_connection
from .models import (
    ConversationNode,
    ConversationState,
    CompressionLevel,
    NodeType,
    SearchResult,
)
from .db_utils import _init_schema


class DuckDBStorage:
    """DuckDB storage implementation for conversation nodes."""

    def __init__(self, db_path: str):
        """Initialize storage with database path."""
        self.db_path = db_path
        self._is_memory_db = db_path == ":memory:"
        self._persistent_conn = None

        if self._is_memory_db:
            self._persistent_conn = duckdb.connect(db_path)
            _init_schema(self._persistent_conn)
        else:
            with get_db_connection(self.db_path, init_schema=True) as conn:
                pass  # Schema initialization happens in get_db_connection

    @contextmanager
    def _get_connection(self):
        """Get appropriate database connection with proper cleanup."""
        if self._is_memory_db:
            # For memory databases, yield the persistent connection
            yield self._persistent_conn
        else:
            # For file databases, use the existing context manager
            with get_db_connection(self.db_path, init_schema=False) as conn:
                yield conn

    def close(self):
        """Close persistent connection if exists."""
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None

    async def get_conversation_nodes(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        level: Optional[CompressionLevel] = None,
    ) -> List[ConversationNode]:
        """Get conversation nodes, optionally filtered by compression level."""
        with self._get_connection() as conn:
            query = """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE conversation_id = ?
            """
            params = [conversation_id]

            if level is not None:
                query += " AND level = ?"
                params.append(level.value)

            query += " ORDER BY sequence_number"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            result = conn.execute(query, params)

            return self._rows_to_nodes(result)

    async def get_node(
        self, node_id: int, conversation_id: str
    ) -> Optional[ConversationNode]:
        """Get a specific node by composite primary key."""
        with self._get_connection() as conn:
            result = conn.execute(
                """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE node_id = ? AND conversation_id = ?
            """,
                (node_id, conversation_id),
            )

            return self._row_to_single_node(result)

    async def save_conversation_node(
        self,
        conversation_id: str,
        node_type: NodeType,
        content: str,
        tokens_used: Optional[int] = None,
        ai_components: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        relates_to_node_id: Optional[int] = None,
    ) -> ConversationNode:
        """Save a conversation node and return the created node."""
        import json
        from datetime import datetime

        # Ensure conversation exists
        await self._ensure_conversation_exists(conversation_id)

        with self._get_connection() as conn:
            # Get the next sequence number for this conversation
            seq_result = conn.execute(
                "SELECT COALESCE(MAX(sequence_number), -1) + 1 FROM nodes WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            sequence_number = seq_result[0] if seq_result else 0

            # Get the next node_id for this conversation (starting from 1)
            node_id_result = conn.execute(
                "SELECT COALESCE(MAX(node_id), 0) + 1 FROM nodes WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            node_id = node_id_result[0] if node_id_result else 1

            # Calculate line count
            line_count = len(content.split("\n"))

            # Insert the new node
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id, conversation_id, node_type, content, timestamp, sequence_number,
                    line_count, level, tokens_used, ai_components, topics, relates_to_node_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    conversation_id,
                    node_type.value,
                    content,
                    datetime.now(),
                    sequence_number,
                    line_count,
                    CompressionLevel.FULL.value,
                    tokens_used,
                    json.dumps(ai_components) if ai_components else None,
                    json.dumps(topics) if topics else None,
                    relates_to_node_id,
                ),
            )

            # Update conversation stats
            conn.execute(
                """
                UPDATE conversations
                SET total_nodes = total_nodes + 1, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (conversation_id,),
            )

            # Fetch and return the created node
            node_result = conn.execute(
                """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE node_id = ? AND conversation_id = ?
                """,
                (node_id, conversation_id),
            )

            return self._row_to_single_node(node_result)

    async def _ensure_conversation_exists(self, conversation_id: str) -> None:
        """Ensure a conversation exists in the conversations table."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO conversations (id, total_nodes, compression_stats)
                VALUES (?, 0, '{}')
                """,
                (conversation_id,),
            )

    async def compress_node(
        self,
        node_id: int,
        conversation_id: str,
        compression_level: CompressionLevel,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Compress a node to a summary."""
        with self._get_connection() as conn:
            result = conn.execute(
                """
                UPDATE nodes
                SET level = ?, summary = ?, summary_metadata = ?
                WHERE node_id = ? AND conversation_id = ?
                RETURNING node_id
            """,
                (
                    compression_level.value,
                    summary,
                    json.dumps(metadata) if metadata else None,
                    node_id,
                    conversation_id,
                ),
            ).fetchone()

            return result is not None

    async def get_recent_nodes(
        self, conversation_id: str, limit: int = 10
    ) -> List[ConversationNode]:
        """Get the most recent nodes at FULL compression level."""
        return await self.get_conversation_nodes(
            conversation_id=conversation_id, limit=limit, level=CompressionLevel.FULL
        )

    async def get_recent_compressed_nodes(
        self, conversation_id: str, limit: int = 5
    ) -> List[ConversationNode]:
        """Get the most recent nodes at SUMMARY compression level."""
        with self._get_connection() as conn:
            query = """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE conversation_id = ? AND level = ?
                ORDER BY sequence_number DESC
                LIMIT ?
            """
            params = [conversation_id, CompressionLevel.SUMMARY.value, limit]

            result = conn.execute(query, params)

            # Get nodes using arrow and convert
            nodes = self._rows_to_nodes(result)

            # Reverse to get chronological order (oldest first)
            nodes.reverse()
            return nodes

    async def search_nodes(
        self, conversation_id: str, query: str, limit: int = 10, regex: bool = False
    ) -> List[SearchResult]:
        """Text search across nodes with support for exact matches or regex patterns."""
        with self._get_connection() as conn:
            if regex:
                # For regex search, get all nodes and filter in Python
                result = conn.execute(
                    """
                    SELECT
                        node_id, conversation_id, node_type, content, timestamp,
                        sequence_number, line_count, level, summary, summary_metadata,
                        parent_summary_node_id, tokens_used, expandable,
                        ai_components, topics, embedding, relates_to_node_id
                    FROM nodes
                    WHERE conversation_id = ?
                    ORDER BY sequence_number DESC
                    """,
                    (conversation_id,),
                )
            else:
                # For exact search, use SQL ILIKE for efficiency
                result = conn.execute(
                    """
                    SELECT
                        node_id, conversation_id, node_type, content, timestamp,
                        sequence_number, line_count, level, summary, summary_metadata,
                        parent_summary_node_id, tokens_used, expandable,
                        ai_components, topics, embedding, relates_to_node_id
                    FROM nodes
                    WHERE conversation_id = ?
                        AND (content ILIKE ? OR summary ILIKE ?)
                    ORDER BY sequence_number DESC
                    LIMIT ?
                    """,
                    (conversation_id, f"%{query}%", f"%{query}%", limit),
                )

            nodes = self._rows_to_nodes(result)
            search_results = []

            # Compile regex pattern if using regex mode
            compiled_pattern = None
            if regex:
                try:
                    compiled_pattern = re.compile(query, re.IGNORECASE)
                except re.error as e:
                    # Invalid regex pattern, return empty results
                    return []

            matches_found = 0
            for node in nodes:
                if regex and compiled_pattern:
                    # Use regex matching
                    content_matches = bool(compiled_pattern.search(node.content))
                    summary_matches = node.summary and bool(
                        compiled_pattern.search(node.summary)
                    )
                else:
                    # Use exact substring matching
                    content_matches = query.lower() in node.content.lower()
                    summary_matches = (
                        node.summary and query.lower() in node.summary.lower()
                    )

                # Skip nodes that don't match
                if not content_matches and not summary_matches:
                    continue

                relevance_score = (
                    0.8 if content_matches else 0.4 if summary_matches else 0.2
                )
                match_type = (
                    "content"
                    if content_matches
                    else "summary"
                    if summary_matches
                    else "none"
                )

                search_results.append(
                    SearchResult(
                        node=node,
                        relevance_score=relevance_score,
                        match_type=match_type,
                        matched_text=query,
                    )
                )

                matches_found += 1
                if matches_found >= limit:
                    break

            return search_results

    async def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            return result is not None

    async def get_conversation_stats(
        self, conversation_id: str
    ) -> Optional[ConversationState]:
        """Get conversation statistics and state."""
        with self._get_connection() as conn:
            conv_result = conn.execute(
                """
                SELECT total_nodes, compression_stats, current_goal, key_decisions, last_updated
                FROM conversations
                WHERE id = ?
            """,
                (conversation_id,),
            ).fetchone()

            if not conv_result:
                return None

            level_counts = {}
            for level in CompressionLevel:
                count_result = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE conversation_id = ? AND level = ?",
                    (conversation_id, level.value),
                ).fetchone()
                level_counts[level] = count_result[0] if count_result else 0

            return ConversationState(
                conversation_id=conversation_id,
                total_nodes=conv_result[0],
                compression_stats=level_counts,
                current_goal=conv_result[2],
                key_decisions=json.loads(conv_result[3]) if conv_result[3] else [],
                last_updated=conv_result[4],
            )

    def _rows_to_nodes(self, result) -> List[ConversationNode]:
        """Convert DuckDB result to ConversationNode objects using PyArrow."""
        arrow_table = result.arrow()
        nodes = [
            ConversationNode.model_validate(row) for row in arrow_table.to_pylist()
        ]
        return [self._enhance_node_summary(node) for node in nodes]

    def _row_to_single_node(self, result) -> Optional[ConversationNode]:
        """Convert a single DuckDB result row to ConversationNode using PyArrow."""
        arrow_table = result.arrow()
        rows = arrow_table.to_pylist()
        if not rows:
            return None
        node = ConversationNode.model_validate(rows[0])
        return self._enhance_node_summary(node)

    def _enhance_node_summary(self, node: ConversationNode) -> ConversationNode:
        """Enhance summary/archive nodes by appending line count information."""
        # Only enhance nodes that have been compressed (not FULL level) and have a summary
        if node.level != CompressionLevel.FULL and node.summary:
            # Create a copy of the node with enhanced summary
            enhanced_summary = (
                f"ID {node.node_id}: {node.summary} ({node.line_count} lines)"
            )
            # Create a new node with the enhanced summary
            node_dict = node.model_dump()
            node_dict["summary"] = enhanced_summary
            return ConversationNode.model_validate(node_dict)
        return node

    async def get_nodes_in_range(
        self, 
        conversation_id: str, 
        start_node_id: int, 
        end_node_id: int
    ) -> List[ConversationNode]:
        """Get nodes within a specific ID range for a conversation."""
        with self._get_connection() as conn:
            result = conn.execute(
                """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE conversation_id = ? AND node_id >= ? AND node_id <= ?
                ORDER BY sequence_number
                """,
                (conversation_id, start_node_id, end_node_id),
            )

            return self._rows_to_nodes(result)

    async def create_meta_group_node(
        self,
        conversation_id: str,
        meta_group,  # MetaGroup object
        grouped_node_ids: List[int]
    ) -> ConversationNode:
        """Create a META-level node that represents a group of SUMMARY nodes."""
        from datetime import datetime
        import json

        # Create content for the META node
        meta_content = f"META GROUP: {meta_group.summary}"
        
        # Create comprehensive summary with group information
        meta_summary = (
            f"Meta group of nodes {meta_group.start_node_id}-{meta_group.end_node_id}: "
            f"{meta_group.summary} "
            f"(groups {meta_group.node_count} nodes, {meta_group.total_lines} total lines)"
        )

        # Create metadata about the grouped nodes
        meta_metadata = {
            "meta_group_info": {
                "start_node_id": meta_group.start_node_id,
                "end_node_id": meta_group.end_node_id,
                "start_sequence": meta_group.start_sequence,
                "end_sequence": meta_group.end_sequence,
                "node_count": meta_group.node_count,
                "total_lines": meta_group.total_lines,
                "main_topics": meta_group.main_topics,
                "timestamp_range": [
                    meta_group.timestamp_range[0].isoformat(),
                    meta_group.timestamp_range[1].isoformat()
                ]
            },
            "grouped_node_ids": grouped_node_ids,
            "compression_level": "META",
            "is_group_node": True
        }

        # Ensure conversation exists
        await self._ensure_conversation_exists(conversation_id)

        with self._get_connection() as conn:
            # Get the next sequence number for this conversation
            seq_result = conn.execute(
                "SELECT COALESCE(MAX(sequence_number), -1) + 1 FROM nodes WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            sequence_number = seq_result[0] if seq_result else 0

            # Get the next node_id for this conversation
            node_id_result = conn.execute(
                "SELECT COALESCE(MAX(node_id), 0) + 1 FROM nodes WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            node_id = node_id_result[0] if node_id_result else 1

            # Calculate line count for the META content
            line_count = len(meta_content.split("\n"))

            # Insert the META group node
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id, conversation_id, node_type, content, timestamp, sequence_number,
                    line_count, level, summary, summary_metadata, topics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    conversation_id,
                    NodeType.AI.value,  # META groups are typically AI-generated summaries
                    meta_content,
                    datetime.now(),
                    sequence_number,
                    line_count,
                    CompressionLevel.META.value,
                    meta_summary,
                    json.dumps(meta_metadata),
                    json.dumps(meta_group.main_topics) if meta_group.main_topics else None,
                ),
            )

            # Mark the original nodes as part of this META group
            for grouped_node_id in grouped_node_ids:
                conn.execute(
                    """
                    UPDATE nodes
                    SET parent_summary_node_id = ?, level = ?
                    WHERE node_id = ? AND conversation_id = ?
                    """,
                    (node_id, CompressionLevel.META.value, grouped_node_id, conversation_id),
                )

            # Update conversation stats
            conn.execute(
                """
                UPDATE conversations
                SET total_nodes = total_nodes + 1, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (conversation_id,),
            )

            # Fetch and return the created META node
            node_result = conn.execute(
                """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE node_id = ? AND conversation_id = ?
                """,
                (node_id, conversation_id),
            )

            return self._row_to_single_node(node_result)
