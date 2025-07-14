"""DuckDB storage layer for hierarchical memory system."""

import json
from typing import List, Optional, Dict, Any, Tuple
import duckdb
from contextlib import contextmanager
from .db_utils import get_db_connection
from .models import (
    ConversationNode,
    ConversationState,
    ConversationTurn,
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

    async def save_conversation_turn(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        tokens_used: Optional[int] = None,
        ai_components: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Save a complete conversation turn (user message + AI response)."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM nodes WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()
            next_sequence = result[0] if result else 1

            user_node_result = conn.execute("""
                INSERT INTO nodes (
                    conversation_id, node_type, content, sequence_number, line_count
                ) VALUES (?, ?, ?, ?, ?)
                RETURNING id, timestamp
            """, (
                conversation_id,
                NodeType.USER.value,
                user_message,
                next_sequence,
                len(user_message.splitlines())
            )).fetchone()

            user_node_id, timestamp = user_node_result

            ai_node_result = conn.execute("""
                INSERT INTO nodes (
                    conversation_id, node_type, content, sequence_number, line_count,
                    tokens_used, ai_components
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """, (
                conversation_id,
                NodeType.AI.value,
                ai_response,
                next_sequence + 1,
                len(ai_response.splitlines()),
                tokens_used,
                json.dumps(ai_components) if ai_components else None
            )).fetchone()

            ai_node_id = ai_node_result[0]

            conn.execute("""
                INSERT INTO conversations (id, total_nodes)
                VALUES (?, 2)
                ON CONFLICT (id) DO UPDATE SET
                    total_nodes = total_nodes + 2,
                    last_updated = NOW()
            """, (conversation_id,))

            return ConversationTurn(
                turn_id=user_node_id,
                conversation_id=conversation_id,
                user_message=user_message,
                ai_response=ai_response,
                timestamp=timestamp,
                tokens_used=tokens_used,
                user_node_id=user_node_id,
                ai_node_id=ai_node_id,
            )

    async def get_conversation_nodes(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        level: Optional[CompressionLevel] = None
    ) -> List[ConversationNode]:
        """Get conversation nodes, optionally filtered by compression level."""
        with self._get_connection() as conn:
            query = """
                SELECT
                    id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_id, tokens_used, expandable,
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

            results = conn.execute(query, params).fetchall()

            return [self._row_to_node(row) for row in results]

    async def get_node(self, node_id: int) -> Optional[ConversationNode]:
        """Get a specific node by ID."""
        with self._get_connection() as conn:
            result = conn.execute("""
                SELECT
                    id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE id = ?
            """, (node_id,)).fetchone()

            return self._row_to_node(result) if result else None

    async def compress_node(
        self,
        node_id: int,
        compression_level: CompressionLevel,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Compress a node to a summary."""
        with self._get_connection() as conn:
            result = conn.execute("""
                UPDATE nodes
                SET level = ?, summary = ?, summary_metadata = ?
                WHERE id = ?
                RETURNING id
            """, (
                compression_level.value,
                summary,
                json.dumps(metadata) if metadata else None,
                node_id
            )).fetchone()

            return result is not None

    async def get_recent_nodes(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[ConversationNode]:
        """Get the most recent nodes at FULL compression level."""
        return await self.get_conversation_nodes(
            conversation_id=conversation_id,
            limit=limit,
            level=CompressionLevel.FULL
        )

    async def search_nodes(
        self,
        conversation_id: str,
        query: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """Basic text search across nodes (Phase 1 implementation)."""
        with self._get_connection() as conn:

            results = conn.execute("""
                SELECT
                    id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE conversation_id = ?
                    AND (content ILIKE ? OR summary ILIKE ?)
                ORDER BY sequence_number DESC
                LIMIT ?
            """, (conversation_id, f"%{query}%", f"%{query}%", limit)).fetchall()

            search_results = []
            for row in results:
                node = self._row_to_node(row)

                content_matches = query.lower() in node.content.lower()
                summary_matches = node.summary and query.lower() in node.summary.lower()
                
                relevance_score = 0.8 if content_matches else 0.4 if summary_matches else 0.2
                match_type = "content" if content_matches else "summary" if summary_matches else "none"
                
                search_results.append(SearchResult(
                    node=node,
                    relevance_score=relevance_score,
                    match_type=match_type,
                    matched_text=query
                ))

            return search_results

    async def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()
            return result is not None

    async def get_conversation_stats(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation statistics and state."""
        with self._get_connection() as conn:

            conv_result = conn.execute("""
                SELECT total_nodes, compression_stats, current_goal, key_decisions, last_updated
                FROM conversations
                WHERE id = ?
            """, (conversation_id,)).fetchone()

            if not conv_result:
                return None


            level_counts = {}
            for level in CompressionLevel:
                count_result = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE conversation_id = ? AND level = ?",
                    (conversation_id, level.value)
                ).fetchone()
                level_counts[level] = count_result[0] if count_result else 0

            return ConversationState(
                conversation_id=conversation_id,
                total_nodes=conv_result[0],
                compression_stats=level_counts,
                current_goal=conv_result[2],
                key_decisions=json.loads(conv_result[3]) if conv_result[3] else [],
                last_updated=conv_result[4]
            )

    def _row_to_node(self, row: tuple) -> ConversationNode:
        """Convert database row to ConversationNode."""
        return ConversationNode(
            id=row[0],
            conversation_id=row[1],
            node_type=NodeType(row[2]),
            content=row[3],
            timestamp=row[4],
            sequence_number=row[5],
            line_count=row[6],
            level=CompressionLevel(row[7]),
            summary=row[8],
            summary_metadata=json.loads(row[9]) if row[9] else None,
            parent_summary_id=row[10],
            tokens_used=row[11],
            expandable=row[12],
            ai_components=json.loads(row[13]) if row[13] else None,
            topics=json.loads(row[14]) if row[14] else [],
            embedding=row[15],
            relates_to_node_id=row[16]
        )
