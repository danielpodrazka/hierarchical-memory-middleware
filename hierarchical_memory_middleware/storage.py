"""DuckDB storage layer for hierarchical memory system."""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple, Literal
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

logger = logging.getLogger(__name__)

# Type alias for search modes
SearchMode = Literal["keyword", "semantic", "hybrid"]

# Module-level flag to avoid repeated warnings about missing embeddings
_embeddings_warning_shown = False


class DuckDBStorage:
    """DuckDB storage implementation for conversation nodes."""

    def __init__(self, db_path: str, enable_semantic_search: bool = True):
        """Initialize storage with database path.

        Args:
            db_path: Path to the DuckDB database file, or ":memory:" for in-memory
            enable_semantic_search: Whether to enable semantic search (requires embeddings deps)
        """
        self.db_path = db_path
        self._is_memory_db = db_path == ":memory:"
        self._persistent_conn = None
        self._vss_loaded = False
        self._embedder = None
        self._enable_semantic_search = enable_semantic_search

        if self._is_memory_db:
            self._persistent_conn = duckdb.connect(db_path)
            _init_schema(self._persistent_conn)
            if enable_semantic_search:
                self._init_vss(self._persistent_conn)
        else:
            with get_db_connection(self.db_path, init_schema=True) as conn:
                if enable_semantic_search:
                    self._init_vss(conn)

    def _ensure_schema_exists(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Check if schema exists and initialize if missing.

        This is a safety check for edge cases where the DB file exists
        but schema wasn't properly initialized (e.g., crash during init,
        corrupted file, etc.).
        """
        try:
            # Quick check - just see if token_usage table exists
            conn.execute("SELECT 1 FROM token_usage LIMIT 0")
        except duckdb.CatalogException:
            # Table doesn't exist - reinitialize schema
            logger.warning(f"Schema missing in {self.db_path}, reinitializing...")
            _init_schema(conn)

    @contextmanager
    def _get_connection(self):
        """Get appropriate database connection with proper cleanup."""
        if self._is_memory_db:
            # For memory databases, yield the persistent connection
            yield self._persistent_conn
        else:
            # For file databases, use the existing context manager
            with get_db_connection(self.db_path, init_schema=False) as conn:
                # Safety check for schema existence
                self._ensure_schema_exists(conn)
                yield conn

    def close(self):
        """Close persistent connection if exists."""
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None

    def _init_vss(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Initialize VSS extension for vector similarity search.

        This loads the DuckDB VSS extension which enables HNSW indexes
        for efficient approximate nearest neighbor search.
        """
        if self._vss_loaded:
            return

        try:
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
            self._vss_loaded = True
            logger.debug("VSS extension loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load VSS extension: {e}. Semantic search will use brute-force.")
            self._vss_loaded = False

    def _get_embedder(self):
        """Get or create the embedder instance (lazy initialization)."""
        global _embeddings_warning_shown
        if self._embedder is None:
            try:
                from .embeddings import get_embedder, is_embeddings_available

                if not is_embeddings_available():
                    if not _embeddings_warning_shown:
                        logger.warning(
                            "Embeddings dependencies not available. "
                            "Install with: pip install 'hierarchical-memory-middleware[embeddings]'"
                        )
                        _embeddings_warning_shown = True
                    return None

                self._embedder = get_embedder()
                logger.debug(f"Embedder initialized: {self._embedder.model.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedder: {e}")
                return None

        return self._embedder

    def _ensure_vss_loaded(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Ensure VSS is loaded for the given connection."""
        if not self._vss_loaded:
            self._init_vss(conn)
        return self._vss_loaded

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text string.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding, or None if unavailable
        """
        embedder = self._get_embedder()
        if embedder is None:
            return None

        try:
            # Truncate very long texts to avoid model limits
            # Most embedding models handle ~512 tokens well
            max_chars = 8000  # ~2000 tokens for typical text
            if len(text) > max_chars:
                text = text[:max_chars]

            return embedder.embed(text)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (or None for failed ones)
        """
        embedder = self._get_embedder()
        if embedder is None:
            return [None] * len(texts)

        try:
            # Truncate long texts
            max_chars = 8000
            truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]
            return embedder.embed_batch(truncated)
        except Exception as e:
            logger.warning(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)

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
        generate_embedding: bool = False,
    ) -> ConversationNode:
        """Save a conversation node and return the created node.

        Args:
            conversation_id: ID of the conversation
            node_type: Type of node (USER or AI)
            content: The message content
            tokens_used: Optional token count
            ai_components: Optional AI response components
            topics: Optional list of topics
            relates_to_node_id: Optional reference to related node
            generate_embedding: If True, generate embedding for semantic search (slower)

        Returns:
            The created ConversationNode
        """
        import json
        from datetime import datetime

        # Ensure conversation exists
        await self._ensure_conversation_exists(conversation_id)

        # Optionally generate embedding
        embedding = None
        embedding_sql = "NULL"
        if generate_embedding and self._enable_semantic_search:
            embedding = await self.generate_embedding(content)
            if embedding:
                embedding_dim = len(embedding)
                embedding_sql = f"[{', '.join(str(x) for x in embedding)}]::FLOAT[{embedding_dim}]"

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

            # Insert the new node (with embedding if generated)
            conn.execute(
                f"""
                INSERT INTO nodes (
                    node_id, conversation_id, node_type, content, timestamp, sequence_number,
                    line_count, level, tokens_used, ai_components, topics, relates_to_node_id, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {embedding_sql})
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
        topics: Optional[List[str]] = None,
    ) -> bool:
        """Compress a node to a summary."""
        with self._get_connection() as conn:
            result = conn.execute(
                """
                UPDATE nodes
                SET level = ?, summary = ?, summary_metadata = ?, topics = ?
                WHERE node_id = ? AND conversation_id = ?
                RETURNING node_id
            """,
                (
                    compression_level.value,
                    summary,
                    json.dumps(metadata) if metadata else None,
                    json.dumps(topics) if topics else None,
                    node_id,
                    conversation_id,
                ),
            ).fetchone()

            return result is not None

    async def apply_compression_result(
        self,
        conversation_id: str,
        compression_result,  # CompressionResult object
    ) -> bool:
        """Apply a compression result to update a node with TF-IDF topics."""
        return await self.compress_node(
            node_id=compression_result.original_node_id,
            conversation_id=conversation_id,
            compression_level=CompressionLevel.SUMMARY,
            summary=compression_result.compressed_content,
            metadata=compression_result.metadata,
            topics=compression_result.topics_extracted,
        )

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

    async def get_recent_hierarchical_nodes(
        self, conversation_id: str, limit: int = 10
    ) -> List[ConversationNode]:
        """Get the most recent compressed nodes from all hierarchy levels (SUMMARY, META, ARCHIVE)."""
        with self._get_connection() as conn:
            query = """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE conversation_id = ? AND level IN (?, ?, ?)
                ORDER BY sequence_number DESC
                LIMIT ?
            """
            params = [
                conversation_id,
                CompressionLevel.SUMMARY.value,
                CompressionLevel.META.value,
                CompressionLevel.ARCHIVE.value,
                limit,
            ]

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

    async def search_nodes_semantic(
        self,
        conversation_id: str,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.3,
    ) -> List[SearchResult]:
        """Semantic search across nodes using vector similarity.

        Args:
            conversation_id: The conversation to search within
            query: Natural language query
            limit: Maximum number of results
            similarity_threshold: Minimum cosine similarity (0-1) for results

        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Generate embedding for the query
        query_embedding = await self.generate_embedding(query)
        if query_embedding is None:
            logger.warning("Could not generate query embedding, falling back to keyword search")
            return await self.search_nodes(conversation_id, query, limit)

        with self._get_connection() as conn:
            # Ensure VSS is loaded
            self._ensure_vss_loaded(conn)

            # Get embedding dimension
            embedding_dim = len(query_embedding)

            # Convert query embedding to DuckDB array format with fixed size
            query_array = f"[{', '.join(str(x) for x in query_embedding)}]::FLOAT[{embedding_dim}]"

            # First, get all nodes with embeddings and compute similarity in Python
            # (DuckDB's array_cosine_similarity requires both arrays to have same fixed size)
            result = conn.execute(
                """
                SELECT
                    node_id, conversation_id, node_type, content, timestamp,
                    sequence_number, line_count, level, summary, summary_metadata,
                    parent_summary_node_id, tokens_used, expandable,
                    ai_components, topics, embedding, relates_to_node_id
                FROM nodes
                WHERE conversation_id = ?
                    AND embedding IS NOT NULL
                """,
                (conversation_id,),
            )

            nodes = self._rows_to_nodes(result)

            # Calculate cosine similarity for each node
            import math

            def cosine_similarity(a: List[float], b: List[float]) -> float:
                """Compute cosine similarity between two vectors."""
                if len(a) != len(b):
                    return 0.0
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot / (norm_a * norm_b)

            # Score all nodes
            scored_nodes = []
            for node in nodes:
                if node.embedding and len(node.embedding) == embedding_dim:
                    similarity = cosine_similarity(query_embedding, node.embedding)
                    if similarity >= similarity_threshold:
                        scored_nodes.append((node, similarity))

            # Sort by similarity descending
            scored_nodes.sort(key=lambda x: x[1], reverse=True)

            # Build search results
            search_results = []
            for node, similarity in scored_nodes[:limit]:
                search_results.append(
                    SearchResult(
                        node=node,
                        relevance_score=float(similarity),
                        match_type="semantic",
                        matched_text=query,
                    )
                )

            return search_results

    async def search_nodes_hybrid(
        self,
        conversation_id: str,
        query: str,
        limit: int = 10,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[SearchResult]:
        """Hybrid search combining keyword and semantic matching.

        Uses reciprocal rank fusion to combine results from both methods.

        Args:
            conversation_id: The conversation to search within
            query: Natural language query
            limit: Maximum number of results
            keyword_weight: Weight for keyword search results (0-1)
            semantic_weight: Weight for semantic search results (0-1)

        Returns:
            List of SearchResult objects sorted by combined relevance
        """
        # Run both searches in parallel conceptually (but sequentially for simplicity)
        keyword_results = await self.search_nodes(conversation_id, query, limit * 2)
        semantic_results = await self.search_nodes_semantic(
            conversation_id, query, limit * 2
        )

        # Build score maps using reciprocal rank fusion
        # RRF score = sum(1 / (k + rank)) where k is typically 60
        k = 60
        node_scores: Dict[int, Dict[str, Any]] = {}

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            node_id = result.node.node_id
            rrf_score = keyword_weight * (1 / (k + rank + 1))
            if node_id not in node_scores:
                node_scores[node_id] = {
                    "node": result.node,
                    "score": 0.0,
                    "match_types": [],
                    "keyword_score": result.relevance_score,
                    "semantic_score": 0.0,
                }
            node_scores[node_id]["score"] += rrf_score
            node_scores[node_id]["match_types"].append("keyword")

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            node_id = result.node.node_id
            rrf_score = semantic_weight * (1 / (k + rank + 1))
            if node_id not in node_scores:
                node_scores[node_id] = {
                    "node": result.node,
                    "score": 0.0,
                    "match_types": [],
                    "keyword_score": 0.0,
                    "semantic_score": result.relevance_score,
                }
            node_scores[node_id]["score"] += rrf_score
            node_scores[node_id]["semantic_score"] = result.relevance_score
            if "semantic" not in node_scores[node_id]["match_types"]:
                node_scores[node_id]["match_types"].append("semantic")

        # Sort by combined score and take top results
        sorted_results = sorted(
            node_scores.values(), key=lambda x: x["score"], reverse=True
        )[:limit]

        # Convert to SearchResult objects
        search_results = []
        for item in sorted_results:
            match_type = "+".join(item["match_types"])
            search_results.append(
                SearchResult(
                    node=item["node"],
                    relevance_score=item["score"],
                    match_type=match_type,
                    matched_text=query,
                )
            )

        return search_results

    async def update_node_embedding(
        self,
        node_id: int,
        conversation_id: str,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update the embedding for a specific node.

        If embedding is None, generates a new embedding from the node's content.

        Args:
            node_id: The node to update
            conversation_id: The conversation the node belongs to
            embedding: Optional pre-computed embedding

        Returns:
            True if successful
        """
        if embedding is None:
            # Fetch node content and generate embedding
            node = await self.get_node(node_id, conversation_id)
            if node is None:
                return False

            # Use summary if available, otherwise content
            text = node.summary if node.summary else node.content
            embedding = await self.generate_embedding(text)

            if embedding is None:
                return False

        with self._get_connection() as conn:
            embedding_dim = len(embedding)
            embedding_str = f"[{', '.join(str(x) for x in embedding)}]::FLOAT[{embedding_dim}]"

            conn.execute(
                f"""
                UPDATE nodes
                SET embedding = {embedding_str}
                WHERE node_id = ? AND conversation_id = ?
                """,
                (node_id, conversation_id),
            )
            return True

    async def backfill_embeddings(
        self,
        conversation_id: str,
        batch_size: int = 50,
    ) -> int:
        """Generate embeddings for all nodes that don't have them.

        Args:
            conversation_id: The conversation to backfill
            batch_size: Number of nodes to process at once

        Returns:
            Number of nodes updated
        """
        updated_count = 0

        with self._get_connection() as conn:
            # Get nodes without embeddings
            result = conn.execute(
                """
                SELECT node_id, content, summary
                FROM nodes
                WHERE conversation_id = ?
                    AND embedding IS NULL
                ORDER BY sequence_number
                """,
                (conversation_id,),
            )
            rows = result.fetchall()

            if not rows:
                return 0

            # Process in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                node_ids = [row[0] for row in batch]
                texts = [row[2] if row[2] else row[1] for row in batch]  # summary or content

                embeddings = await self.generate_embeddings_batch(texts)

                # Update each node
                for node_id, embedding in zip(node_ids, embeddings):
                    if embedding is not None:
                        await self.update_node_embedding(
                            node_id, conversation_id, embedding
                        )
                        updated_count += 1

        logger.info(f"Backfilled {updated_count} embeddings for conversation {conversation_id}")
        return updated_count

    async def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            return result is not None

    async def get_conversation_list(self) -> List[Dict[str, Any]]:
        """Get a list of all conversations with basic information."""
        with self._get_connection() as conn:
            result = conn.execute(
                """
                SELECT
                    id,
                    name,
                    total_nodes,
                    created_at,
                    last_updated,
                    CASE WHEN total_nodes > 0 THEN 1 ELSE 0 END as is_active
                FROM conversations
                ORDER BY last_updated DESC
                """
            ).fetchall()

            conversations = []
            for row in result:
                conversations.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "node_count": row[2],
                        "created": row[3].isoformat() if row[3] else None,
                        "last_updated": row[4].isoformat() if row[4] else None,
                        "is_active": bool(row[5]),
                    }
                )

            return conversations

    async def set_conversation_name(self, conversation_id: str, name: str) -> bool:
        """Set the name of a conversation."""
        with self._get_connection() as conn:
            try:
                result = conn.execute(
                    """
                    UPDATE conversations
                    SET name = ?
                    WHERE id = ?
                    RETURNING id
                    """,
                    (name, conversation_id),
                ).fetchone()
                return result is not None
            except Exception:
                # Handle unique constraint violation
                return False

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

    def _arrow_to_pylist(self, arrow_obj) -> list:
        """Convert a DuckDB arrow result to a list of dicts.

        Handles both pyarrow.Table (older DuckDB) and pyarrow.RecordBatchReader (newer DuckDB).
        """
        # Newer DuckDB returns RecordBatchReader, need to call read_all()
        if hasattr(arrow_obj, 'read_all'):
            arrow_obj = arrow_obj.read_all()
        return arrow_obj.to_pylist()

    def _rows_to_nodes(self, result) -> List[ConversationNode]:
        """Convert DuckDB result to ConversationNode objects using PyArrow."""
        arrow_obj = result.arrow()
        rows = self._arrow_to_pylist(arrow_obj)
        nodes = [
            ConversationNode.model_validate(row) for row in rows
        ]
        return [self._enhance_node_summary(node) for node in nodes]

    def _row_to_single_node(self, result) -> Optional[ConversationNode]:
        """Convert a single DuckDB result row to ConversationNode using PyArrow."""
        arrow_obj = result.arrow()
        rows = self._arrow_to_pylist(arrow_obj)
        if not rows:
            return None
        node = ConversationNode.model_validate(rows[0])
        return self._enhance_node_summary(node)

    def _enhance_node_summary(self, node: ConversationNode) -> ConversationNode:
        """Enhance summary/archive nodes by appending line count and TF-IDF topics information."""
        # Only enhance nodes that have been compressed (not FULL level) and have a summary
        if node.level == CompressionLevel.SUMMARY and node.summary:
            # Build enhanced summary with line count
            enhanced_summary = (
                f"ID {node.node_id}: {node.summary} ({node.line_count} lines)"
            )

            # Add TF-IDF topics if available
            # Ensure topics is a list and not None
            if node.topics and isinstance(node.topics, list) and len(node.topics) > 0:
                try:
                    # Filter out empty strings and None values
                    valid_topics = [
                        topic
                        for topic in node.topics
                        if topic and isinstance(topic, str)
                    ]
                    if valid_topics:
                        topics_str = ", ".join(valid_topics[:3])  # Show top 3 topics
                        enhanced_summary += f" [Topics: {topics_str}]"
                except Exception as e:
                    # Log the error but don't break the summary enhancement
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug(
                        f"Error processing topics for node {node.node_id}: {e}"
                    )

            # Create a new node with the enhanced summary
            node_dict = node.model_dump()
            node_dict["summary"] = enhanced_summary
            return ConversationNode.model_validate(node_dict)
        return node

    async def get_nodes_in_range(
        self, conversation_id: str, start_node_id: int, end_node_id: int
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
        grouped_node_ids: List[int],
    ) -> ConversationNode:
        """Create a META-level node that represents a group of SUMMARY nodes."""
        from datetime import datetime
        import json

        all_topics_str = (
            ", ".join(meta_group.main_topics) if meta_group.main_topics else ""
        )
        topics_section = f" [Topics: {all_topics_str}]" if all_topics_str else ""

        meta_content = (
            f"Nodes {meta_group.start_node_id}-{meta_group.end_node_id}:"
            f"({meta_group.node_count} nodes, {meta_group.total_lines} lines){topics_section}\n"
        )

        # Create summary for storage (not displayed to AI)
        meta_summary = (
            f"META: Nodes {meta_group.start_node_id}-{meta_group.end_node_id} "
            f"({meta_group.node_count} nodes, {meta_group.total_lines} lines)"
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
                    meta_group.timestamp_range[1].isoformat(),
                ],
            },
            "grouped_node_ids": grouped_node_ids,
            "compression_level": "META",
            "is_group_node": True,
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
                    json.dumps(meta_group.main_topics)
                    if meta_group.main_topics
                    else None,
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
                    (
                        node_id,
                        CompressionLevel.META.value,
                        grouped_node_id,
                        conversation_id,
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

    async def remove_node(self, node_id: int, conversation_id: str) -> bool:
        """Remove a specific node by composite primary key.

        Args:
            node_id: The node ID to remove
            conversation_id: The conversation ID the node belongs to

        Returns:
            True if the node was successfully removed, False if the node was not found
        """
        with self._get_connection() as conn:
            # First check if the node exists
            check_result = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_id = ? AND conversation_id = ?",
                (node_id, conversation_id),
            ).fetchone()

            if not check_result or check_result[0] == 0:
                return False  # Node not found

            # Remove the node
            conn.execute(
                "DELETE FROM nodes WHERE node_id = ? AND conversation_id = ?",
                (node_id, conversation_id),
            )

            # Update conversation stats (decrement total_nodes)
            conn.execute(
                """
                UPDATE conversations
                SET total_nodes = GREATEST(total_nodes - 1, 0), last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (conversation_id,),
            )

            return True

    async def get_system_prompt(self, conversation_id: str) -> Optional[str]:
        """Get the system prompt for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The system prompt text, or None if not set
        """
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT system_prompt FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if result and result[0]:
                return result[0]
            return None

    async def set_system_prompt(self, conversation_id: str, content: str) -> bool:
        """Set or replace the system prompt for a conversation.

        Args:
            conversation_id: The conversation ID
            content: The new system prompt content

        Returns:
            True if successful
        """
        # Ensure conversation exists
        await self._ensure_conversation_exists(conversation_id)

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET system_prompt = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (content, conversation_id),
            )
            return True

    async def append_system_prompt(self, conversation_id: str, content: str) -> str:
        """Append content to the system prompt for a conversation.

        Args:
            conversation_id: The conversation ID
            content: The content to append

        Returns:
            The updated system prompt
        """
        # Ensure conversation exists
        await self._ensure_conversation_exists(conversation_id)

        with self._get_connection() as conn:
            # Get current system prompt
            result = conn.execute(
                "SELECT system_prompt FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            current_prompt = result[0] if result and result[0] else ""

            # Append new content
            if current_prompt:
                new_prompt = f"{current_prompt}\n{content}"
            else:
                new_prompt = content

            # Update
            conn.execute(
                """
                UPDATE conversations
                SET system_prompt = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_prompt, conversation_id),
            )

            return new_prompt

    async def save_token_usage(
        self,
        conversation_id: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        duration_ms: Optional[int] = None,
        model: Optional[str] = None,
    ) -> int:
        """Save token usage for a conversation turn.

        Args:
            conversation_id: The conversation ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_read_tokens: Tokens read from cache
            cache_creation_tokens: Tokens used for cache creation
            cost_usd: Cost in USD
            duration_ms: Duration in milliseconds
            model: Model used

        Returns:
            The ID of the created usage record
        """
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

        with self._get_connection() as conn:
            result = conn.execute(
                """
                INSERT INTO token_usage (
                    conversation_id, input_tokens, output_tokens,
                    cache_read_tokens, cache_creation_tokens, total_tokens,
                    cost_usd, duration_ms, model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    conversation_id,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                    total_tokens,
                    cost_usd,
                    duration_ms,
                    model,
                ),
            ).fetchone()
            return result[0] if result else 0

    async def get_conversation_token_usage(
        self, conversation_id: str
    ) -> Dict[str, Any]:
        """Get aggregated token usage for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            Dictionary with total and per-turn token usage statistics
        """
        with self._get_connection() as conn:
            # Get aggregated totals
            totals = conn.execute(
                """
                SELECT
                    COUNT(*) as turn_count,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COALESCE(SUM(cache_read_tokens), 0) as total_cache_read_tokens,
                    COALESCE(SUM(cache_creation_tokens), 0) as total_cache_creation_tokens,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                    COALESCE(SUM(duration_ms), 0) as total_duration_ms
                FROM token_usage
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

            if not totals or totals[0] == 0:
                return {
                    "turn_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cache_read_tokens": 0,
                    "total_cache_creation_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "total_duration_ms": 0,
                    "recent_turns": [],
                }

            # Get recent turns (last 5)
            recent = conn.execute(
                """
                SELECT
                    timestamp, input_tokens, output_tokens,
                    cache_read_tokens, cache_creation_tokens,
                    total_tokens, cost_usd, duration_ms, model
                FROM token_usage
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT 5
                """,
                (conversation_id,),
            ).fetchall()

            recent_turns = [
                {
                    "timestamp": row[0].isoformat() if row[0] else None,
                    "input_tokens": row[1],
                    "output_tokens": row[2],
                    "cache_read_tokens": row[3],
                    "cache_creation_tokens": row[4],
                    "total_tokens": row[5],
                    "cost_usd": row[6],
                    "duration_ms": row[7],
                    "model": row[8],
                }
                for row in recent
            ]

            return {
                "turn_count": totals[0],
                "total_input_tokens": totals[1],
                "total_output_tokens": totals[2],
                "total_cache_read_tokens": totals[3],
                "total_cache_creation_tokens": totals[4],
                "total_tokens": totals[5],
                "total_cost_usd": float(totals[6]) if totals[6] else 0.0,
                "total_duration_ms": totals[7],
                "recent_turns": recent_turns,
            }
