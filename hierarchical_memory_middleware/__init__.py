"""
Hierarchical Memory Middleware

A middleware system that enables infinite AI agent conversations
through intelligent hierarchical compression.

Supports multiple LLM providers:
- Claude Agent SDK (recommended): Uses Claude Pro/Max subscription via CLI auth
- Anthropic API: Direct Claude API access
- OpenAI, Gemini, Moonshot, DeepSeek, Together: Various providers via API

Features:
- Hierarchical compression (FULL -> SUMMARY -> META -> ARCHIVE)
- Semantic search via DuckDB VSS (requires [embeddings] extra)
- Keyword and hybrid search modes
- System prompt / scratchpad persistence
"""

__version__ = "0.1.0"

# Core exports
from .config import Config
from .middleware.conversation_manager import HierarchicalConversationManager
from .middleware.claude_agent_sdk_manager import ClaudeAgentSDKConversationManager
from .middleware import create_conversation_manager
from .storage import DuckDBStorage
from .compression import SimpleCompressor, CompressionManager
from .model_manager import ModelManager, ClaudeAgentSDKMarker
from .models import (
    ConversationNode,
    ConversationState,
    CompressionLevel,
    NodeType,
    CompressionResult,
    SearchResult,
    ModelProvider,
    ModelConfig,
)

# Optional embeddings support (requires [embeddings] extra)
try:
    from .embeddings import (
        get_embedder,
        get_embedding_dimension,
        is_embeddings_available,
        EmbeddingModel,
        Embedder,
    )
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False
    # Define stubs for type hints
    get_embedder = None
    get_embedding_dimension = None
    is_embeddings_available = lambda: False
    EmbeddingModel = None
    Embedder = None

__all__ = [
    # Configuration
    "Config",
    # Conversation managers
    "HierarchicalConversationManager",
    "ClaudeAgentSDKConversationManager",
    "create_conversation_manager",  # Factory function (recommended)
    # Storage
    "DuckDBStorage",
    # Compression
    "SimpleCompressor",
    "CompressionManager",
    # Models
    "ModelManager",
    "ModelProvider",
    "ModelConfig",
    "ClaudeAgentSDKMarker",
    # Data types
    "ConversationNode",
    "ConversationState",
    "CompressionLevel",
    "NodeType",
    "CompressionResult",
    "SearchResult",
    # Embeddings (optional)
    "get_embedder",
    "get_embedding_dimension",
    "is_embeddings_available",
    "EmbeddingModel",
    "Embedder",
    # Version
    "__version__",
]
