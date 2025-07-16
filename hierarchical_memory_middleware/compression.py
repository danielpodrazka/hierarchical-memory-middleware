"""Compression logic with TF-IDF topic extraction for Phase 5 implementation."""

import re
from typing import Dict, Any, List, Optional
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .models import ConversationNode, CompressionLevel, CompressionResult


class TfidfTopicExtractor:
    """Extracts topics from text using TF-IDF analysis."""

    def __init__(self, max_features: int = 100, max_topics: int = 5, min_df: int = 1, max_df: float = 0.95):
        """Initialize TF-IDF topic extractor.

        Args:
            max_features: Maximum number of features for TF-IDF
            max_topics: Maximum number of topics to return
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
        """
        self.max_features = max_features
        self.max_topics = max_topics
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.document_corpus = []

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        return text.strip()

    def _get_stopwords(self) -> set:
        """Get expanded set of English stopwords."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'myself', 'yourself',
            'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }

    def add_document(self, text: str) -> None:
        """Add a document to the corpus for TF-IDF training."""
        preprocessed = self._preprocess_text(text)
        if preprocessed:
            self.document_corpus.append(preprocessed)

    def fit(self) -> None:
        """Fit the TF-IDF vectorizer on the document corpus."""
        if not self.document_corpus:
            # If no corpus, create a minimal vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words=list(self._get_stopwords()),
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 2)
            )
            return

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=list(self._get_stopwords()),
            min_df=max(1, min(self.min_df, len(self.document_corpus) // 4)),
            max_df=self.max_df,
            ngram_range=(1, 2)
        )

        # Fit the vectorizer
        try:
            self.vectorizer.fit(self.document_corpus)
        except ValueError:
            # Fallback for edge cases
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words=list(self._get_stopwords()),
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1)
            )
            self.vectorizer.fit(self.document_corpus)

    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using TF-IDF."""
        if not self.vectorizer:
            # Fallback to simple keyword extraction
            return self._fallback_topic_extraction(text)

        preprocessed = self._preprocess_text(text)
        if not preprocessed:
            return []

        try:
            # Transform the text
            tfidf_matrix = self.vectorizer.transform([preprocessed])

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]

            # Create term-score pairs
            term_scores = list(zip(feature_names, scores))

            # Filter out zero scores and sort by score
            term_scores = [(term, score) for term, score in term_scores if score > 0]
            term_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top terms
            topics = [term for term, score in term_scores[:self.max_topics]]
            return topics

        except Exception:
            # Fallback to simple extraction if TF-IDF fails
            return self._fallback_topic_extraction(text)

    def _fallback_topic_extraction(self, text: str) -> List[str]:
        """Fallback topic extraction using simple word frequency."""
        words = self._preprocess_text(text).split()
        stopwords = self._get_stopwords()

        # Filter words
        keywords = [
            word for word in words
            if len(word) > 3 and word not in stopwords and word.isalpha()
        ]

        # Count frequencies
        word_counts = Counter(keywords)

        # Return top keywords
        return [word for word, count in word_counts.most_common(self.max_topics)]


class TfidfCompressor:
    """Phase 5 compression with TF-IDF topic extraction."""

    def __init__(self, max_words: int = 8, topic_extractor: Optional[TfidfTopicExtractor] = None):
        """Initialize with maximum words for truncation and TF-IDF topic extractor."""
        self.max_words = max_words
        self.topic_extractor = topic_extractor or TfidfTopicExtractor()
        self._corpus_fitted = False

    def compress_node(self, node: ConversationNode) -> CompressionResult:
        """Compress a node to its first N words."""
        # Extract first N words
        words = self._extract_words(node.content)
        truncated_words = words[: self.max_words]
        compressed_content = " ".join(truncated_words)

        # Add ellipsis if truncated
        if len(words) > self.max_words:
            compressed_content += "..."

        # Calculate compression ratio
        original_length = len(node.content)
        compressed_length = len(compressed_content)
        compression_ratio = (
            compressed_length / original_length if original_length > 0 else 1.0
        )

        # Extract TF-IDF topics
        topics = self.topic_extractor.extract_topics(node.content)

        return CompressionResult(
            original_node_id=node.node_id,
            compressed_content=compressed_content,
            compression_ratio=compression_ratio,
            topics_extracted=topics,
            metadata={
                "original_words": len(words),
                "compressed_words": len(truncated_words),
                "truncated": len(words) > self.max_words,
                "compression_method": "first_n_words",
                "max_words": self.max_words,
            },
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
        cleaned = re.sub(r"\s+", " ", text.strip())
        return cleaned.split() if cleaned else []

    def build_corpus(self, nodes: List[ConversationNode]) -> None:
        """Build the TF-IDF corpus from existing conversation nodes."""
        for node in nodes:
            self.topic_extractor.add_document(node.content)
        
        # Fit the vectorizer
        self.topic_extractor.fit()
        self._corpus_fitted = True

    def ensure_corpus_fitted(self, nodes: List[ConversationNode]) -> None:
        """Ensure the TF-IDF corpus is fitted before extraction."""
        if not self._corpus_fitted:
            self.build_corpus(nodes)

    def compress_node_with_corpus(self, node: ConversationNode, corpus_nodes: List[ConversationNode]) -> CompressionResult:
        """Compress a node ensuring TF-IDF corpus is trained on provided nodes."""
        self.ensure_corpus_fitted(corpus_nodes)
        return self.compress_node(node)

    def update_corpus(self, new_node: ConversationNode) -> None:
        """Add a new node to the corpus and refit if necessary."""
        self.topic_extractor.add_document(new_node.content)
        # Only refit if we already had a fitted corpus
        if self._corpus_fitted:
            self.topic_extractor.fit()


class CompressionManager:
    """Manages compression logic for the conversation."""

    def __init__(self, compressor: TfidfCompressor, recent_node_limit: int = 10):
        """Initialize with TF-IDF compressor and configuration."""
        self.compressor = compressor
        self.recent_node_limit = recent_node_limit

    def identify_nodes_to_compress(
        self, nodes: List[ConversationNode]
    ) -> List[ConversationNode]:
        """Identify which nodes should be compressed."""
        # Sort nodes by sequence number
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number)

        # Keep recent nodes uncompressed, compress older ones
        total_nodes = len(sorted_nodes)
        if total_nodes <= self.recent_node_limit:
            return []  # Don't compress anything if under limit

        # Compress all but the most recent nodes
        nodes_to_compress = sorted_nodes[: -self.recent_node_limit]

        # Only compress nodes that are at FULL level
        return [
            node for node in nodes_to_compress if node.level == CompressionLevel.FULL
        ]

    def compress_nodes(self, nodes: List[ConversationNode], all_nodes: Optional[List[ConversationNode]] = None) -> List[CompressionResult]:
        """Compress a list of nodes with TF-IDF corpus training."""
        # Use all_nodes for corpus training, or fall back to the nodes being compressed
        corpus_nodes = all_nodes or nodes
        
        # Ensure TF-IDF corpus is trained
        self.compressor.ensure_corpus_fitted(corpus_nodes)
        
        results = []
        for node in nodes:
            if self.compressor.should_compress(node, len(nodes)):
                result = self.compressor.compress_node(node)
                results.append(result)
        return results


# Backward compatibility alias
SimpleCompressor = TfidfCompressor
