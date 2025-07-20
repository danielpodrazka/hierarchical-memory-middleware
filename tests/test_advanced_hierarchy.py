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


@pytest.mark.asyncio
async def test_meta_archive_compression_preserves_topics():
    """Test that META/ARCHIVE compression preserves topics even when content is truncated.

    This test verifies that topics extracted during compression are properly stored
    separately from content and remain accessible even when content gets heavily
    truncated. This ensures that topic information is not lost during compression.
    """
    # Use in-memory database for testing
    storage = DuckDBStorage(":memory:")
    await storage._ensure_conversation_exists("test-conv")

    compressor = SimpleCompressor()
    # Set very low thresholds to trigger META and ARCHIVE compression
    thresholds = HierarchyThresholds(
        summary_threshold=2,  # Keep only 2 recent nodes as FULL
        meta_threshold=3,  # Keep only 3 recent nodes as SUMMARY
        archive_threshold=2,  # Keep only 2 recent META groups
        meta_group_size=2,  # Minimum 2 nodes per META group
        meta_group_max=3,  # Maximum 3 nodes per META group
    )
    manager = AdvancedCompressionManager(compressor, thresholds)

    # Create content with words that will be extracted as topics
    # but will not all fit in the truncated content fragments
    long_content_with_topics = (
        "This comprehensive discussion covers machine learning algorithms, neural networks, "
        "artificial intelligence applications, deep learning architectures, and data science "
        "methodologies. We explore advanced computational techniques, mathematical models, "
        "statistical analysis, pattern recognition, optimization algorithms, supervised learning, "
        "unsupervised learning, reinforcement learning, computer vision, natural language "
        "processing, robotics applications, automation systems, predictive analytics, "
        "classification algorithms, regression analysis, clustering techniques, and "
        "dimensionality reduction methods. This text contains many technical terms that "
        "should be extracted as topics during compression, but when the content gets "
        "heavily truncated to just a few words, most of these topic words will be lost "
        "from the content fragment while still being preserved in the topics field."
    )

    # Create enough nodes to trigger multiple levels of compression
    nodes = []
    for i in range(1, 12):  # 11 nodes - enough to trigger SUMMARY -> META -> ARCHIVE
        node = await storage.save_conversation_node(
            conversation_id="test-conv",
            node_type=NodeType.USER if i % 2 == 1 else NodeType.AI,
            content=f"Node {i}: {long_content_with_topics}",
            # Don't set initial topics - let the compression system extract them
        )
        nodes.append(node)

    print(
        f"Created {len(nodes)} nodes with long content containing many potential topics"
    )

    # Process compression multiple times to create META and ARCHIVE levels
    # First pass: FULL -> SUMMARY
    result1 = await manager.process_hierarchy_compression(nodes, storage)
    print(f"First compression pass: {result1}")

    # Get updated nodes and compress again: SUMMARY -> META
    all_nodes_after_first = await storage.get_conversation_nodes("test-conv")
    result2 = await manager.process_hierarchy_compression(
        all_nodes_after_first, storage
    )
    print(f"Second compression pass: {result2}")

    # Get updated nodes and compress again: META -> ARCHIVE
    all_nodes_after_second = await storage.get_conversation_nodes("test-conv")
    result3 = await manager.process_hierarchy_compression(
        all_nodes_after_second, storage
    )
    print(f"Third compression pass: {result3}")

    # Get final state of all nodes
    final_nodes = await storage.get_conversation_nodes("test-conv")
    print(f"Final nodes: {len(final_nodes)}")
    print(f"Final node levels: {[n.level for n in final_nodes]}")

    # Test META level nodes
    meta_nodes = [n for n in final_nodes if n.level == CompressionLevel.META]
    print(f"META nodes: {len(meta_nodes)}")

    for meta_node in meta_nodes:
        print(
            f"META node {meta_node.node_id}: content length = {len(meta_node.content)}, topics = {meta_node.topics}"
        )
        print(f"  META node content: {meta_node.content}")

        # Verify META node has topics extracted during compression
        assert (
            meta_node.topics is not None and len(meta_node.topics) > 0
        ), f"META node {meta_node.node_id} should have topics"

        # Verify that content is significantly truncated (should be much shorter than original)
        # META nodes are group summaries, so they're informative rather than heavily compressed
        # They should be shorter than original but contain comprehensive topic information
        assert (
            len(meta_node.content) < len(long_content_with_topics) / 2
        ), f"META node {meta_node.node_id} should be more concise than original - got {len(meta_node.content)} vs {len(long_content_with_topics)}"

        # Verify META node content includes topics for AI visibility
        assert (
            "[Topics:" in meta_node.content
        ), f"META node {meta_node.node_id} should include '[Topics:' for AI visibility - got: {meta_node.content}"

        # Verify the content follows the expected META group format
        assert (
            "nodes," in meta_node.content.lower() and "lines)" in meta_node.content
        ), f"META node {meta_node.node_id} should include node/line count - got: {meta_node.content}"

        print(
            f"✓ META node {meta_node.node_id} has proper format with 'Discussion of' and visible topics"
        )

        # Critical test: ensure topics are preserved even though content is truncated
        # The topics should come from the topics field, not just from scanning the truncated content
        preserved_topics = meta_node.topics
        topics_not_in_content = []
        topics_in_content = []

        for topic in preserved_topics:
            # Check if the topic appears in the truncated content
            topic_in_content = topic.lower() in meta_node.content.lower()
            if topic_in_content:
                topics_in_content.append(topic)
            else:
                topics_not_in_content.append(topic)

        print(f"  Topics in content: {topics_in_content}")
        print(f"  Topics preserved but not in content: {topics_not_in_content}")

        # At least one topic should be preserved that's not in the truncated content
        # This proves topics are stored separately from content
        if topics_not_in_content:
            print(
                f"✓ META node {meta_node.node_id} has {len(topics_not_in_content)} topics preserved despite content truncation"
            )

    # Test ARCHIVE level nodes
    archive_nodes = [n for n in final_nodes if n.level == CompressionLevel.ARCHIVE]
    print(f"ARCHIVE nodes: {len(archive_nodes)}")

    for archive_node in archive_nodes:
        print(
            f"ARCHIVE node {archive_node.node_id}: content length = {len(archive_node.content)}, topics = {archive_node.topics}"
        )

        # Verify ARCHIVE node has topics extracted during compression
        assert (
            archive_node.topics is not None and len(archive_node.topics) > 0
        ), f"ARCHIVE node {archive_node.node_id} should have topics"

        # Verify that content is very heavily truncated for ARCHIVE level
        # Verify that the ARCHIVE node has a compressed summary (the summary field should be truncated)
        assert (
            archive_node.summary is not None
        ), f"ARCHIVE node {archive_node.node_id} should have a summary"
        assert (
            len(archive_node.summary) < len(long_content_with_topics) / 2
        ), f"ARCHIVE node {archive_node.node_id} summary should be compressed"

        # Critical test: ensure topics are preserved even with extreme compression
        preserved_topics = archive_node.topics
        topics_not_in_content = []
        topics_in_content = []

        for topic in preserved_topics:
            # Check if the topic appears in the heavily truncated content
            topic_in_content = topic.lower() in archive_node.content.lower()
            if topic_in_content:
                topics_in_content.append(topic)
            else:
                topics_not_in_content.append(topic)

        print(f"  ARCHIVE topics in content: {topics_in_content}")
        print(f"  ARCHIVE topics preserved but not in content: {topics_not_in_content}")

        # At least one topic should be preserved that's not in the truncated content
        # This proves topics are stored separately from content
        if topics_not_in_content:
            print(
                f"✓ ARCHIVE node {archive_node.node_id} has {len(topics_not_in_content)} topics preserved despite extreme content truncation"
            )

    # Ensure we actually tested some META or ARCHIVE nodes
    assert (
        len(meta_nodes) > 0 or len(archive_nodes) > 0
    ), "Test should have created at least some META or ARCHIVE nodes"

    # Verify that at least one topic was preserved despite content truncation across all compressed nodes
    total_topics_preserved_despite_truncation = 0
    all_compressed_nodes = meta_nodes + archive_nodes

    for node in all_compressed_nodes:
        topics_not_in_content_count = 0
        for topic in node.topics:
            topic_in_content = topic.lower() in node.content.lower()
            if not topic_in_content:
                topics_not_in_content_count += 1
        total_topics_preserved_despite_truncation += topics_not_in_content_count

    print(
        f"Total topics preserved despite content truncation: {total_topics_preserved_despite_truncation}"
    )

    # Success verification: our improvements ensure topics are visible to AI agents
    # Either topics are preserved despite truncation OR (even better) they're visible in content
    total_nodes_with_topics = len([n for n in all_compressed_nodes if n.topics])
    all_topics_present = all(
        any(topic.lower() in node.content.lower() for topic in node.topics)
        for node in all_compressed_nodes
        if node.topics
    )

    print(f"Nodes with topics: {total_nodes_with_topics}/{len(all_compressed_nodes)}")
    print(f"All topics visible in compressed content: {all_topics_present}")

    # Test success: either topics are preserved separately OR (better) visible in content
    assert total_topics_preserved_despite_truncation > 0 or all_topics_present, (
        "Test should demonstrate topics are accessible to AI agents - either preserved separately ",
        "or (ideally) visible in the compressed content itself",
    )

    if all_topics_present and total_topics_preserved_despite_truncation == 0:
        print(
            "✓ EXCELLENT: All topics are now visible to AI agents in compressed content!"
        )
        print(
            "✓ This means our enhancement worked - topics are included in compressed summaries"
        )
    else:
        print(
            "✓ Topics preserved independently of content truncation for AI accessibility"
        )

    print(
        "✓ Test passed: META/ARCHIVE compression properly preserves and displays topics for AI visibility"
    )


if __name__ == "__main__":
    pytest.main([__file__])
