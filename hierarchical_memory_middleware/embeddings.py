"""Embeddings module for semantic search functionality.

This module provides embedding generation using sentence-transformers for local,
cost-free semantic search. The default model (all-MiniLM-L6-v2) is small (~80MB),
fast, and produces 384-dimensional embeddings.

Usage:
    from hierarchical_memory_middleware.embeddings import get_embedder, EmbeddingModel

    embedder = get_embedder()  # Uses default model
    embedding = embedder.embed("Hello, world!")
    embeddings = embedder.embed_batch(["Hello", "World"])
"""

import logging
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Embedding dimension for the default model (all-MiniLM-L6-v2)
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingModel(Enum):
    """Supported embedding models."""

    # Sentence Transformers models (local, free)
    MINILM_L6_V2 = "all-MiniLM-L6-v2"  # 384 dims, ~80MB, fast
    MINILM_L12_V2 = "all-MiniLM-L12-v2"  # 384 dims, ~120MB, slightly better quality
    MPNET_BASE_V2 = "all-mpnet-base-v2"  # 768 dims, ~420MB, best quality

    # OpenAI models (API-based, paid)
    OPENAI_ADA_002 = "text-embedding-ada-002"  # 1536 dims
    OPENAI_3_SMALL = "text-embedding-3-small"  # 1536 dims
    OPENAI_3_LARGE = "text-embedding-3-large"  # 3072 dims


# Model dimension mapping
MODEL_DIMENSIONS = {
    EmbeddingModel.MINILM_L6_V2: 384,
    EmbeddingModel.MINILM_L12_V2: 384,
    EmbeddingModel.MPNET_BASE_V2: 768,
    EmbeddingModel.OPENAI_ADA_002: 1536,
    EmbeddingModel.OPENAI_3_SMALL: 1536,
    EmbeddingModel.OPENAI_3_LARGE: 3072,
}


class Embedder:
    """Base embedder interface."""

    def __init__(self, model: EmbeddingModel):
        self.model = model
        self.dimension = MODEL_DIMENSIONS[model]

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """Embedder using sentence-transformers (local, free)."""

    def __init__(self, model: EmbeddingModel = EmbeddingModel.MINILM_L6_V2):
        super().__init__(model)
        self._model = None

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading sentence-transformer model: {self.model.value}")
                self._model = SentenceTransformer(self.model.value)
                logger.info(f"Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install 'hierarchical-memory-middleware[embeddings]'"
                )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


class OpenAIEmbedder(Embedder):
    """Embedder using OpenAI's embedding API (paid)."""

    def __init__(self, model: EmbeddingModel = EmbeddingModel.OPENAI_3_SMALL):
        super().__init__(model)
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import os
                from openai import OpenAI

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")

                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI embeddings. "
                    "Install with: pip install 'hierarchical-memory-middleware[embeddings]'"
                )
        return self._client

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        client = self._get_client()
        response = client.embeddings.create(input=text, model=self.model.value)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model.value)
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


# Global embedder instance (lazy initialized)
_embedder: Optional[Embedder] = None


def get_embedder(
    model: EmbeddingModel = EmbeddingModel.MINILM_L6_V2,
    force_new: bool = False,
) -> Embedder:
    """Get or create the global embedder instance.

    Args:
        model: The embedding model to use
        force_new: If True, create a new embedder even if one exists

    Returns:
        Configured embedder instance
    """
    global _embedder

    if _embedder is None or force_new:
        if model.value.startswith("text-embedding"):
            _embedder = OpenAIEmbedder(model)
        else:
            _embedder = SentenceTransformerEmbedder(model)

    return _embedder


def get_embedding_dimension(model: EmbeddingModel = EmbeddingModel.MINILM_L6_V2) -> int:
    """Get the dimension of embeddings for a given model."""
    return MODEL_DIMENSIONS[model]


def is_embeddings_available() -> bool:
    """Check if embedding dependencies are available."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False
