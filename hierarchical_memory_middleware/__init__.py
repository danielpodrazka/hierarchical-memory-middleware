"""
Hierarchical Memory Middleware

A middleware system that enables infinite AI agent conversations
through intelligent hierarchical compression.
"""

__version__ = "0.1.0"

# Core exports
from .config import Config
from .middleware.conversation_manager import HierarchicalConversationManager

__all__ = [
    "Config",
    "HierarchicalConversationManager",
    "__version__",
]
