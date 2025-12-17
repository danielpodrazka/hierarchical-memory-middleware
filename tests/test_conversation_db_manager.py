"""Tests for ConversationDBManager - per-conversation DuckDB isolation."""

import asyncio
import tempfile
import os
from pathlib import Path
import pytest

from hierarchical_memory_middleware.conversation_db_manager import ConversationDBManager
from hierarchical_memory_middleware.models import NodeType
from hierarchical_memory_middleware.db_utils import get_db_connection, _init_schema


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for database files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestConversationDBManager:
    """Tests for ConversationDBManager functionality."""

    @pytest.mark.asyncio
    async def test_startup_no_orphans(self, temp_db_dir):
        """Test startup when there are no orphaned files."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")

        manager = ConversationDBManager(main_db, active_dir)
        orphan_count = await manager.startup()

        assert orphan_count == 0
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_storage_creates_active_file(self, temp_db_dir):
        """Test that get_storage creates an isolated active file."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")

        manager = ConversationDBManager(main_db, active_dir)
        await manager.startup()

        storage = await manager.get_storage("test-conv")
        active_file = Path(active_dir) / "test-conv.duckdb"

        assert active_file.exists()
        assert manager.is_active("test-conv")

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_save_and_merge(self, temp_db_dir):
        """Test that nodes saved to active DB are merged back to main."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")

        manager = ConversationDBManager(main_db, active_dir)
        await manager.startup()

        # Save nodes to active storage
        storage = await manager.get_storage("test-conv")
        await storage.save_conversation_node(
            conversation_id="test-conv",
            node_type=NodeType.USER,
            content="Hello, world!",
            tokens_used=10,
        )
        await storage.save_conversation_node(
            conversation_id="test-conv",
            node_type=NodeType.AI,
            content="Hi there!",
            tokens_used=15,
        )
        manager.mark_dirty("test-conv")

        # Shutdown merges back to main
        await manager.shutdown()

        # Verify data in main DB
        with get_db_connection(main_db) as conn:
            nodes = conn.execute(
                "SELECT node_id, content FROM nodes WHERE conversation_id = ? ORDER BY node_id",
                ("test-conv",),
            ).fetchall()

        assert len(nodes) == 2
        assert nodes[0][1] == "Hello, world!"
        assert nodes[1][1] == "Hi there!"

    @pytest.mark.asyncio
    async def test_orphan_healing(self, temp_db_dir):
        """Test that orphaned active files are healed on startup."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")
        os.makedirs(active_dir)

        # Create an "orphan" file (simulating bad shutdown)
        orphan_path = Path(active_dir) / "orphan-conv.duckdb"
        with get_db_connection(str(orphan_path), init_schema=True) as conn:
            conn.execute("""
                INSERT INTO conversations (id, total_nodes, compression_stats)
                VALUES ('orphan-conv', 1, '{}')
            """)
            conn.execute("""
                INSERT INTO nodes (node_id, conversation_id, node_type, content,
                                   timestamp, sequence_number, line_count, level)
                VALUES (1, 'orphan-conv', 'user', 'orphan message',
                        CURRENT_TIMESTAMP, 0, 1, 0)
            """)

        # Start manager - should heal orphan
        manager = ConversationDBManager(main_db, active_dir)
        orphan_count = await manager.startup()

        assert orphan_count == 1
        assert not orphan_path.exists()  # Orphan file should be deleted

        # Verify orphan data was merged to main
        with get_db_connection(main_db) as conn:
            result = conn.execute(
                "SELECT content FROM nodes WHERE conversation_id = 'orphan-conv'"
            ).fetchone()

        assert result is not None
        assert result[0] == "orphan message"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_parallel_conversations(self, temp_db_dir):
        """Test multiple parallel conversations don't have lock contention."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")

        manager = ConversationDBManager(main_db, active_dir)
        await manager.startup()

        async def simulate_conversation(conv_id, num_messages):
            storage = await manager.get_storage(conv_id)
            for i in range(num_messages):
                await storage.save_conversation_node(
                    conversation_id=conv_id,
                    node_type=NodeType.USER,
                    content=f"Message {i}",
                    tokens_used=10,
                )
                await asyncio.sleep(0.01)  # Small delay
            manager.mark_dirty(conv_id)

        # Run 5 conversations in parallel
        tasks = [
            simulate_conversation(f"conv-{i}", 10)
            for i in range(5)
        ]
        await asyncio.gather(*tasks)

        assert len(manager.get_active_conversations()) == 5

        await manager.shutdown()

        # Verify all data was merged
        with get_db_connection(main_db) as conn:
            for i in range(5):
                count = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE conversation_id = ?",
                    (f"conv-{i}",),
                ).fetchone()[0]
                assert count == 10

    @pytest.mark.asyncio
    async def test_force_sync(self, temp_db_dir):
        """Test force sync without closing conversation."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")

        manager = ConversationDBManager(main_db, active_dir)
        await manager.startup()

        storage = await manager.get_storage("test-conv")
        await storage.save_conversation_node(
            conversation_id="test-conv",
            node_type=NodeType.USER,
            content="Test message",
            tokens_used=10,
        )
        manager.mark_dirty("test-conv")

        # Force sync while still active
        result = await manager.force_sync("test-conv")
        assert result is True

        # Conversation should still be active
        assert manager.is_active("test-conv")

        # Data should be in main DB
        with get_db_connection(main_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE conversation_id = 'test-conv'"
            ).fetchone()[0]
        assert count == 1

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_extract_existing_conversation(self, temp_db_dir):
        """Test extracting an existing conversation from main DB."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")

        # First, create some data in main DB
        with get_db_connection(main_db, init_schema=True) as conn:
            conn.execute("""
                INSERT INTO conversations (id, total_nodes, compression_stats)
                VALUES ('existing-conv', 2, '{}')
            """)
            conn.execute("""
                INSERT INTO nodes (node_id, conversation_id, node_type, content,
                                   timestamp, sequence_number, line_count, level)
                VALUES
                    (1, 'existing-conv', 'user', 'Previous message 1',
                     CURRENT_TIMESTAMP, 0, 1, 0),
                    (2, 'existing-conv', 'ai', 'Previous response 1',
                     CURRENT_TIMESTAMP, 1, 1, 0)
            """)

        manager = ConversationDBManager(main_db, active_dir)
        await manager.startup()

        # Get storage - should extract existing data
        storage = await manager.get_storage("existing-conv")

        # Verify existing data is accessible
        nodes = await storage.get_conversation_nodes("existing-conv")
        assert len(nodes) == 2

        # Add new message
        await storage.save_conversation_node(
            conversation_id="existing-conv",
            node_type=NodeType.USER,
            content="New message",
            tokens_used=10,
        )
        manager.mark_dirty("existing-conv")

        await manager.shutdown()

        # Verify all data (old + new) in main DB
        with get_db_connection(main_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE conversation_id = 'existing-conv'"
            ).fetchone()[0]
        assert count == 3

    @pytest.mark.asyncio
    async def test_not_dirty_not_synced(self, temp_db_dir):
        """Test that non-dirty conversations are not synced."""
        main_db = os.path.join(temp_db_dir, "main.duckdb")
        active_dir = os.path.join(temp_db_dir, "active")

        manager = ConversationDBManager(main_db, active_dir)
        await manager.startup()

        # Get storage but don't mark dirty
        storage = await manager.get_storage("test-conv")
        await storage.save_conversation_node(
            conversation_id="test-conv",
            node_type=NodeType.USER,
            content="Test message",
            tokens_used=10,
        )
        # Note: NOT calling manager.mark_dirty()

        # Force sync should return False
        result = await manager.force_sync("test-conv")
        assert result is False

        await manager.shutdown()
