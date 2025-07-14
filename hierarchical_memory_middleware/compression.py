"""Simple compression logic for Phase 1 implementation."""

import re
from typing import Dict, Any, List

from .models import ConversationNode, CompressionLevel, CompressionResult


class SimpleCompressor:
    """Phase 1 simple compression - just first 8 words."""

    def __init__(self, max_words: int = 8):
        """Initialize with maximum words for truncation."""
        self.max_words = max_words

    def compress_node(self, node: ConversationNode) -> CompressionResult:
        """Compress a node to its first N words."""
        # Extract first N words
        words = self._extract_words(node.content)
        truncated_words = words[:self.max_words]
        compressed_content = " ".join(truncated_words)

        # Add ellipsis if truncated
        if len(words) > self.max_words:
            compressed_content += "..."

        # Calculate compression ratio
        original_length = len(node.content)
        compressed_length = len(compressed_content)
        compression_ratio = compressed_length / original_length if original_length > 0 else 1.0

        # Extract basic topics (simple keyword extraction)
        topics = self._extract_simple_topics(node.content)


        return CompressionResult(
            original_node_id=node.id,
            compressed_content=compressed_content,
            compression_ratio=compression_ratio,
            topics_extracted=topics,
            metadata={
                "original_words": len(words),
                "compressed_words": len(truncated_words),
                "truncated": len(words) > self.max_words,
                "compression_method": "first_n_words",
                "max_words": self.max_words
            }
        )

    def should_compress(self, node: ConversationNode, conversation_length: int) -> bool:
        """Determine if a node should be compressed based on simple rules."""
        # Don't compress if already compressed
        if node.level != CompressionLevel.FULL:
            return False

        # For Phase 1: compress nodes that are older than recent_limit
        # This will be controlled by the conversation manager
        return True

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text, handling basic cleanup."""
        # Remove extra whitespace and split by whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned.split() if cleaned else []

    def _extract_simple_topics(self, text: str) -> List[str]:
        """Extract simple topics using basic keyword extraction."""
        # For Phase 1: very simple topic extraction
        words = self._extract_words(text.lower())
        
        # Simple keyword filtering
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Extract potential keywords (words longer than 3 chars, not stopwords)
        keywords = [
            word for word in words
            if len(word) > 3 and word not in stopwords and word.isalpha()
        ]

        # Return top 5 most frequent keywords
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and return top 5
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:5]]


class CompressionManager:
    """Manages compression logic for the conversation."""

    def __init__(self, compressor: SimpleCompressor, recent_node_limit: int = 10):
        """Initialize with compressor and configuration."""
        self.compressor = compressor
        self.recent_node_limit = recent_node_limit

    def identify_nodes_to_compress(
        self, 
        nodes: List[ConversationNode]
    ) -> List[ConversationNode]:
        """Identify which nodes should be compressed."""
        # Sort nodes by sequence number
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number)

        # Keep recent nodes uncompressed, compress older ones
        total_nodes = len(sorted_nodes)
        if total_nodes <= self.recent_node_limit:
            return []  # Don't compress anything if under limit

        # Compress all but the most recent nodes
        nodes_to_compress = sorted_nodes[:-self.recent_node_limit]

        # Only compress nodes that are at FULL level
        return [
            node for node in nodes_to_compress
            if node.level == CompressionLevel.FULL
        ]

    def compress_nodes(
        self,
        nodes: List[ConversationNode]
    ) -> List[CompressionResult]:
        """Compress a list of nodes."""
        results = []
        for node in nodes:
            if self.compressor.should_compress(node, len(nodes)):
                result = self.compressor.compress_node(node)
                results.append(result)
        return results
