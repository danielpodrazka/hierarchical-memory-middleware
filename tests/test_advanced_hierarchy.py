"""Tests for the advanced hierarchical compression system."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from hierarchical_memory_middleware.advanced_hierarchy import (
    AdvancedHierarchyManager,
    AdvancedCompressionManager,
)
from hierarchical_memory_middleware.compression import SimpleCompressor
from hierarchical_memory_middleware.models import (
    ConversationNode,
    CompressionLevel,
    NodeType,
    HierarchyThresholds,
)
from hierarchical_memory_middleware.storage import DuckDBStorage


def create_test_node(
    node_id: int,
    sequence_number: int,
    content: str,
    node_type: NodeType = NodeType.USER,
    level: CompressionLevel = CompressionLevel.FULL,
    conversation_id: str = "test-conv",
) -> ConversationNode:
    """Create a test conversation node."""
    return ConversationNode(
        node_id=node_id,
        conversation_id=conversation_id,
        node_type=node_type,
        content=content,
        timestamp=datetime.now(),
        sequence_number=sequence_number,
        line_count=len(content.split("\n")),
        level=level,
    )


@pytest.mark.asyncio
async def test_advanced_hierarchy_manager_initialization():
    """Test that the advanced hierarchy manager initializes correctly."""
    compressor = SimpleCompressor()
    thresholds = HierarchyThresholds(
        summary_threshold=5,
        meta_threshold=15,
        archive_threshold=50,
    )

    manager = AdvancedHierarchyManager(compressor, thresholds)

    assert manager.base_compressor == compressor
    assert manager.thresholds == thresholds


@pytest.mark.asyncio
async def test_analyze_compression_needs_no_compression():
    """Test compression analysis when no compression is needed."""
    compressor = SimpleCompressor()
    thresholds = HierarchyThresholds(summary_threshold=10)
    manager = AdvancedHierarchyManager(compressor, thresholds)

    # Create 5 nodes (below threshold)
    nodes = [create_test_node(i, i, f"Test content {i}") for i in range(1, 6)]

    compression_needs = manager.analyze_compression_needs(nodes)

    # No compression should be needed
    assert len(compression_needs) == 0


@pytest.mark.asyncio
async def test_analyze_compression_needs_summary_compression():
    """Test compression analysis when SUMMARY compression is needed."""
    compressor = SimpleCompressor()
    thresholds = HierarchyThresholds(summary_threshold=5)
    manager = AdvancedHierarchyManager(compressor, thresholds)

    # Create 10 nodes (above threshold)
    nodes = [create_test_node(i, i, f"Test content {i}") for i in range(1, 11)]

    compression_needs = manager.analyze_compression_needs(nodes)

    # Should need SUMMARY compression for older nodes
    assert CompressionLevel.SUMMARY in compression_needs
    assert len(compression_needs[CompressionLevel.SUMMARY]) == 5  # 10 - 5 threshold


@pytest.mark.asyncio
async def test_compress_to_summary():
    """Test FULL to SUMMARY compression."""
    compressor = SimpleCompressor()
    manager = AdvancedHierarchyManager(compressor)

    # Create test nodes
    nodes = [
        create_test_node(1, 1, "This is a test conversation with some content"),
        create_test_node(2, 2, "Another test message with different content"),
    ]

    results = manager.compress_to_summary(nodes)

    assert len(results) == 2
    for result in results:
        assert result.original_node_id in [1, 2]
        assert len(result.compressed_content) > 0
        assert result.compression_ratio > 0
        assert "compression_level" in result.metadata
        assert result.metadata["compression_level"] == "SUMMARY"


@pytest.mark.asyncio
async def test_compress_to_meta():
    """Test SUMMARY to META compression."""
    compressor = SimpleCompressor()
    thresholds = HierarchyThresholds(meta_group_size=3, meta_group_max=5)
    manager = AdvancedHierarchyManager(compressor, thresholds)

    # Create test SUMMARY nodes
    nodes = [
        create_test_node(i, i, f"Summary content {i}", level=CompressionLevel.SUMMARY)
        for i in range(1, 7)  # 6 nodes
    ]

    # Add topics to some nodes
    nodes[0].topics = ["topic1", "topic2"]
    nodes[1].topics = ["topic1", "topic3"]
    nodes[2].topics = ["topic2", "topic4"]

    groups = manager.compress_to_meta(nodes)

    assert len(groups) >= 1
    for meta_group, node_ids in groups:
        assert len(node_ids) >= 3  # At least meta_group_size
        assert len(node_ids) <= 5  # At most meta_group_max
        assert meta_group.node_count == len(node_ids)
        assert meta_group.start_node_id <= meta_group.end_node_id
        assert meta_group.start_sequence <= meta_group.end_sequence


@pytest.mark.asyncio
async def test_advanced_compression_manager_backward_compatibility():
    """Test that backward compatibility methods work."""
    compressor = SimpleCompressor()
    thresholds = HierarchyThresholds(summary_threshold=5)
    manager = AdvancedCompressionManager(compressor, thresholds)

    # Test backward compatibility properties
    assert manager.recent_node_limit == 5

    # Create test nodes
    nodes = [
        create_test_node(i, i, f"Test content {i}")
        for i in range(1, 11)  # 10 nodes
    ]

    # Test identify_nodes_to_compress
    nodes_to_compress = manager.identify_nodes_to_compress(nodes)
    assert len(nodes_to_compress) == 5  # 10 - 5 threshold

    # Test compress_nodes
    compression_results = manager.compress_nodes(nodes_to_compress)
    assert len(compression_results) == 5
    for result in compression_results:
        assert result.compressed_content
        assert result.compression_ratio > 0


@pytest.mark.asyncio
async def test_process_hierarchy_compression_integration():
    """Test the complete hierarchy compression process."""
    # Use in-memory database for testing
    storage = DuckDBStorage(":memory:")
    await storage._ensure_conversation_exists("test-conv")

    compressor = SimpleCompressor()
    thresholds = HierarchyThresholds(summary_threshold=3)
    manager = AdvancedCompressionManager(compressor, thresholds)

    # Create and save test nodes
    nodes = []
    for i in range(1, 8):  # 7 nodes
        node = await storage.save_conversation_node(
            conversation_id="test-conv",
            node_type=NodeType.USER if i % 2 == 1 else NodeType.AI,
            content=f"Test content for node {i} with enough text to make it interesting",
        )
        nodes.append(node)

    # Process hierarchy compression
    results = await manager.process_hierarchy_compression(nodes, storage)

    # Check results
    assert "error" not in results
    assert "total_processed" in results

    # Should have compressed some nodes to SUMMARY level
    assert results.get("summary_compressed", 0) > 0

    # Verify nodes were actually compressed in storage
    all_nodes = await storage.get_conversation_nodes("test-conv")
    summary_nodes = [n for n in all_nodes if n.level == CompressionLevel.SUMMARY]
    assert len(summary_nodes) > 0

    # Verify summary content has ID prefix
    for node in summary_nodes:
        assert node.summary
        assert f"ID {node.node_id}:" in node.summary


@pytest.mark.asyncio
async def test_meta_group_formatting():
    """Test that META group nodes are formatted correctly without ID prefixes."""
    # Use in-memory database for testing
    storage = DuckDBStorage(":memory:")
    await storage._ensure_conversation_exists("test-conv")

    compressor = SimpleCompressor()
    # Set low thresholds to trigger META group creation
    thresholds = HierarchyThresholds(
        summary_threshold=3,  # Keep 3 recent nodes as FULL
        meta_threshold=5,  # Keep 5 recent nodes as SUMMARY
        meta_group_size=3,  # Minimum 3 nodes per META group (instead of default 20)
        meta_group_max=6,  # Maximum 6 nodes per META group
    )
    manager = AdvancedCompressionManager(compressor, thresholds)

    # Create enough nodes to trigger META group creation
    nodes = []
    for i in range(1, 15):  # 14 nodes - enough to trigger SUMMARY then META compression
        node = await storage.save_conversation_node(
            conversation_id="test-conv",
            node_type=NodeType.USER if i % 2 == 1 else NodeType.AI,
            content=f"Test content for node {i} with enough text to make it interesting and trigger compression",
            topics=["test_topic", "compression"] if i % 3 == 0 else None,
        )
        nodes.append(node)

    # Process hierarchy compression multiple times to create META groups
    # First pass will compress FULL -> SUMMARY
    results1 = await manager.process_hierarchy_compression(nodes, storage)
    print(f"First compression pass: {results1}")

    # Get updated nodes and run compression again to create META groups
    all_nodes_after_first = await storage.get_conversation_nodes("test-conv")
    results2 = await manager.process_hierarchy_compression(
        all_nodes_after_first, storage
    )
    print(f"Second compression pass: {results2}")

    # Combine results
    results = {
        "summary_compressed": results1.get("summary_compressed", 0)
        + results2.get("summary_compressed", 0),
        "meta_groups_created": results1.get("meta_groups_created", 0)
        + results2.get("meta_groups_created", 0),
        "archive_compressed": results1.get("archive_compressed", 0)
        + results2.get("archive_compressed", 0),
        "total_processed": results1.get("total_processed", 0)
        + results2.get("total_processed", 0),
    }
    print(f"Combined compression results: {results}")

    # Get all nodes to find META group nodes
    all_nodes = await storage.get_conversation_nodes("test-conv")
    print(f"Total nodes after compression: {len(all_nodes)}")
    print(f"Node levels: {[n.level for n in all_nodes]}")

    # Check what was compressed to SUMMARY first
    summary_nodes = [n for n in all_nodes if n.level == CompressionLevel.SUMMARY]
    print(f"SUMMARY nodes: {len(summary_nodes)}")

    meta_group_nodes = [n for n in all_nodes if n.level == CompressionLevel.META]
    print(f"META group nodes: {len(meta_group_nodes)}")

    # Should have created some META group nodes
    assert len(meta_group_nodes) > 0, "No META group nodes were created"

    for meta_node in meta_group_nodes:
        # META group nodes should not have ID prefixes in their summary
        if meta_node.summary:
            assert not meta_node.summary.startswith(
                f"ID {meta_node.node_id}:"
            ), f"META group node {meta_node.node_id} has unexpected ID prefix: {meta_node.summary}"

        # Should not end with line count suffix like "(1 lines)"
        assert not meta_node.content.endswith(
            " lines)"
        ), f"META group node {meta_node.node_id} has unexpected line count suffix: {meta_node.content}"

        assert meta_node.summary_metadata

        print(f"✓ META group node {meta_node.node_id=} formatted correctly:")
        print(f"  Content: {meta_node.content[:100]}...")
        print(f"  Summary: {meta_node.summary=}")
        print(f"  Summary: {meta_node.summary_metadata=}")


if __name__ == "__main__":
    pytest.main([__file__])
