"""Advanced hierarchical compression system for Phase 4."""

import logging
from typing import Dict, Any, List, Optional, Tuple

from .models import (
    ConversationNode,
    CompressionLevel,
    CompressionResult,
    HierarchyThresholds,
    MetaGroup,
)
from .compression import SimpleCompressor


logger = logging.getLogger(__name__)


class AdvancedHierarchyManager:
    """Manages the 4-level hierarchical compression system."""

    def __init__(
        self,
        base_compressor: SimpleCompressor,
        thresholds: Optional[HierarchyThresholds] = None,
    ):
        """Initialize the advanced hierarchy manager."""
        self.base_compressor = base_compressor
        self.thresholds = thresholds or HierarchyThresholds()

    def analyze_compression_needs(
        self, nodes: List[ConversationNode]
    ) -> Dict[CompressionLevel, List[ConversationNode]]:
        """Analyze which nodes need compression and to what levels."""
        if not nodes:
            return {}

        # Sort nodes by sequence number
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number)

        # Group nodes by current compression level
        nodes_by_level = {
            CompressionLevel.FULL: [],
            CompressionLevel.SUMMARY: [],
            CompressionLevel.META: [],
            CompressionLevel.ARCHIVE: [],
        }

        for node in sorted_nodes:
            nodes_by_level[node.level].append(node)

        compression_needs = {}
        total_nodes = len(sorted_nodes)

        # 1. FULL → SUMMARY compression
        full_nodes = nodes_by_level[CompressionLevel.FULL]
        if len(full_nodes) > self.thresholds.summary_threshold:
            # Keep most recent nodes as FULL, compress older ones to SUMMARY
            nodes_to_compress = full_nodes[: -self.thresholds.summary_threshold]
            if nodes_to_compress:
                compression_needs[CompressionLevel.SUMMARY] = nodes_to_compress

        # 2. SUMMARY → META compression
        summary_nodes = nodes_by_level[CompressionLevel.SUMMARY]
        if len(summary_nodes) > self.thresholds.meta_threshold:
            # Group older SUMMARY nodes into META groups
            nodes_to_group = summary_nodes[: -self.thresholds.meta_threshold]
            if nodes_to_group:
                compression_needs[CompressionLevel.META] = nodes_to_group

        # 3. META → ARCHIVE compression
        meta_nodes = nodes_by_level[CompressionLevel.META]
        if len(meta_nodes) > self.thresholds.archive_threshold:
            # Compress older META groups into ARCHIVE
            nodes_to_archive = meta_nodes[: -self.thresholds.archive_threshold]
            if nodes_to_archive:
                compression_needs[CompressionLevel.ARCHIVE] = nodes_to_archive

        logger.info(
            f"Compression analysis: {total_nodes} total nodes, "
            f"needs compression: {sum(len(nodes) for nodes in compression_needs.values())} nodes"
        )

        return compression_needs

    def compress_to_summary(
        self, nodes: List[ConversationNode]
    ) -> List[CompressionResult]:
        """Compress FULL nodes to SUMMARY level."""
        results = []

        for node in nodes:
            if node.level != CompressionLevel.FULL:
                continue

            # Use the base compressor for individual node compression
            base_result = self.base_compressor.compress_node(node)

            # Enhance the result with better summarization
            enhanced_summary = self._create_enhanced_summary(node)

            result = CompressionResult(
                original_node_id=node.node_id,
                compressed_content=enhanced_summary,
                compression_ratio=len(enhanced_summary) / len(node.content)
                if node.content
                else 1.0,
                topics_extracted=base_result.topics_extracted,
                metadata={
                    **base_result.metadata,
                    "compression_level": "SUMMARY",
                    "original_line_count": node.line_count,
                    "enhanced_summary": True,
                },
            )

            results.append(result)

        return results

    def compress_to_meta(
        self, nodes: List[ConversationNode]
    ) -> List[Tuple[MetaGroup, List[int]]]:
        """Group SUMMARY nodes into META level groups."""
        if not nodes:
            return []

        # Sort by sequence number
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number)

        groups = []
        current_group = []

        for node in sorted_nodes:
            if node.level != CompressionLevel.SUMMARY:
                continue

            current_group.append(node)

            # Create group when we reach optimal size
            if (
                len(current_group) >= self.thresholds.meta_group_size
                and len(current_group) <= self.thresholds.meta_group_max
            ):
                group = self._create_meta_group(current_group)
                node_ids = [n.node_id for n in current_group]
                groups.append((group, node_ids))
                current_group = []

        # Handle remaining nodes in final group
        if current_group:
            group = self._create_meta_group(current_group)
            node_ids = [n.node_id for n in current_group]
            groups.append((group, node_ids))

        return groups

    def compress_to_archive(
        self, nodes: List[ConversationNode]
    ) -> List[CompressionResult]:
        """Compress META nodes to ARCHIVE level."""
        results = []

        for node in nodes:
            if node.level != CompressionLevel.META:
                continue

            # Create high-level archive summary
            archive_summary = self._create_archive_summary(node)

            result = CompressionResult(
                original_node_id=node.node_id,
                compressed_content=archive_summary,
                compression_ratio=len(archive_summary) / len(node.content)
                if node.content
                else 1.0,
                topics_extracted=node.topics or [],
                metadata={
                    "compression_level": "ARCHIVE",
                    "original_level": "META",
                    "archive_compression": True,
                    "original_line_count": node.line_count,
                },
            )

            results.append(result)

        return results

    def _create_enhanced_summary(self, node: ConversationNode) -> str:
        """Create an enhanced summary for SUMMARY level."""
        content = node.content or ""

        # Extract first sentence or first 50 words, whichever is shorter
        sentences = content.split(". ")
        first_sentence = sentences[0] if sentences else ""

        words = content.split()
        first_50_words = " ".join(words[:50])

        # Use the shorter option
        base_summary = (
            first_sentence
            if len(first_sentence) <= len(first_50_words)
            else first_50_words
        )

        # Add line count and type information
        node_type = "User" if node.node_type.value == "user" else "AI"
        line_count = node.line_count or len(content.split("\n"))

        enhanced = f"{node_type}: {base_summary}"
        if line_count > 1:
            enhanced += f" ({line_count} lines)"

        # Add topic information if available
        if node.topics:
            topics_str = ", ".join(node.topics[:3])  # First 3 topics
            enhanced += f" [Topics: {topics_str}]"

        return enhanced

    def _create_meta_group(self, nodes: List[ConversationNode]) -> MetaGroup:
        """Create a META group from a list of SUMMARY nodes."""
        if not nodes:
            raise ValueError("Cannot create META group from empty node list")

        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number)
        first_node = sorted_nodes[0]
        last_node = sorted_nodes[-1]

        # Collect topics from all nodes
        all_topics = []
        for node in sorted_nodes:
            if node.topics:
                all_topics.extend(node.topics)

        # Get most common topics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        main_topics = [
            topic
            for topic, count in sorted(
                topic_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

        # Calculate total lines
        total_lines = sum(node.line_count or 0 for node in sorted_nodes)

        # Create group summary
        node_count = len(sorted_nodes)
        summary = self._create_group_summary(sorted_nodes, main_topics)

        return MetaGroup(
            start_node_id=first_node.node_id,
            end_node_id=last_node.node_id,
            start_sequence=first_node.sequence_number,
            end_sequence=last_node.sequence_number,
            node_count=node_count,
            total_lines=total_lines,
            main_topics=main_topics,
            summary=summary,
            timestamp_range=(first_node.timestamp, last_node.timestamp),
        )

    def _create_group_summary(
        self, nodes: List[ConversationNode], main_topics: List[str]
    ) -> str:
        """Create a summary for a group of nodes."""
        node_count = len(nodes)
        total_lines = sum(node.line_count or 0 for node in nodes)

        # Get sequence range
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number)
        start_seq = sorted_nodes[0].sequence_number
        end_seq = sorted_nodes[-1].sequence_number

        # Create summary
        summary = f"Nodes {start_seq}-{end_seq}: "

        if main_topics:
            topics_str = ", ".join(main_topics[:3])
            summary += f"Discussion of {topics_str}"
        else:
            summary += "Conversation segment"

        summary += f" ({node_count} nodes, {total_lines} lines)"

        return summary

    def _create_archive_summary(self, node: ConversationNode) -> str:
        """Create a high-level archive summary."""
        content = node.content or ""

        # Extract key information for archive level
        # This would be very high-level - major decisions, outcomes, etc.
        words = content.split()
        if len(words) > 20:
            # Very aggressive compression for archive
            archive_content = " ".join(words[:15]) + "..."
        else:
            archive_content = content

        # Add metadata
        line_count = node.line_count or 0
        if line_count > 0:
            archive_content += f" (archived from {line_count} lines)"

        return archive_content


class AdvancedCompressionManager:
    """Manages the complete advanced hierarchy compression process."""

    def __init__(
        self,
        base_compressor: SimpleCompressor,
        thresholds: Optional[HierarchyThresholds] = None,
    ):
        """Initialize the advanced compression manager."""
        self.hierarchy_manager = AdvancedHierarchyManager(base_compressor, thresholds)

    # Backward compatibility properties and methods for old CompressionManager interface
    @property
    def recent_node_limit(self) -> int:
        """Get the recent node limit from thresholds for backward compatibility."""
        return self.hierarchy_manager.thresholds.summary_threshold

    def identify_nodes_to_compress(
        self, nodes: List[ConversationNode]
    ) -> List[ConversationNode]:
        """Identify nodes that need compression (backward compatibility)."""
        compression_needs = self.hierarchy_manager.analyze_compression_needs(nodes)
        # Return nodes that need compression to SUMMARY level
        return compression_needs.get(CompressionLevel.SUMMARY, [])

    def compress_nodes(
        self, nodes: List[ConversationNode]
    ) -> List[CompressionResult]:
        """Compress nodes using SUMMARY level compression (backward compatibility)."""
        return self.hierarchy_manager.compress_to_summary(nodes)

    async def process_hierarchy_compression(
        self, nodes: List[ConversationNode], storage
    ) -> Dict[str, Any]:
        """Process all levels of hierarchy compression."""
        results = {
            "summary_compressed": 0,
            "meta_groups_created": 0,
            "archive_compressed": 0,
            "total_processed": 0,
        }

        try:
            # Analyze what compression is needed
            compression_needs = self.hierarchy_manager.analyze_compression_needs(nodes)

            # 1. Compress FULL → SUMMARY
            if CompressionLevel.SUMMARY in compression_needs:
                summary_results = self.hierarchy_manager.compress_to_summary(
                    compression_needs[CompressionLevel.SUMMARY]
                )

                for result in summary_results:
                    await storage.compress_node(
                        node_id=result.original_node_id,
                        conversation_id=nodes[
                            0
                        ].conversation_id,  # Assume same conversation
                        compression_level=CompressionLevel.SUMMARY,
                        summary=result.compressed_content,
                        metadata=result.metadata,
                    )

                results["summary_compressed"] = len(summary_results)

            # 2. Group SUMMARY → META
            if CompressionLevel.META in compression_needs:
                meta_groups = self.hierarchy_manager.compress_to_meta(
                    compression_needs[CompressionLevel.META]
                )

                for meta_group, node_ids in meta_groups:
                    # Create a new META node that represents the group
                    await storage.create_meta_group_node(
                        conversation_id=nodes[0].conversation_id,
                        meta_group=meta_group,
                        grouped_node_ids=node_ids,
                    )

                results["meta_groups_created"] = len(meta_groups)

            # 3. Compress META → ARCHIVE
            if CompressionLevel.ARCHIVE in compression_needs:
                archive_results = self.hierarchy_manager.compress_to_archive(
                    compression_needs[CompressionLevel.ARCHIVE]
                )

                for result in archive_results:
                    await storage.compress_node(
                        node_id=result.original_node_id,
                        conversation_id=nodes[0].conversation_id,
                        compression_level=CompressionLevel.ARCHIVE,
                        summary=result.compressed_content,
                        metadata=result.metadata,
                    )

                results["archive_compressed"] = len(archive_results)

            results["total_processed"] = (
                results["summary_compressed"]
                + results["meta_groups_created"]
                + results["archive_compressed"]
            )

            return results

        except Exception as e:
            logger.error(f"Error in hierarchy compression: {str(e)}", exc_info=True)
            return {"error": str(e), **results}
