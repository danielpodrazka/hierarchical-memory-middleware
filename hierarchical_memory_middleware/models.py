"""Data models for hierarchical memory system."""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import json
from pydantic import BaseModel, Field, field_validator


class CompressionLevel(Enum):
    """Compression levels for conversation nodes."""

    FULL = 0  # Recent nodes - complete content
    SUMMARY = 1  # Older nodes - 1-sentence summaries
    META = 2  # Groups of summaries (20-40 nodes each group)
    ARCHIVE = 3  # Very old - high-level context


class NodeType(Enum):
    """Types of conversation nodes."""

    USER = "user"
    AI = "ai"


class ConversationNode(BaseModel):
    """A single conversation node storing user message and AI response."""

    node_id: int
    conversation_id: str
    node_type: NodeType
    content: str  # For USER: the message. For AI: all content combined
    timestamp: datetime
    sequence_number: int  # Order within conversation
    line_count: int  # Number of lines in the full content

    # Compression fields
    level: CompressionLevel = CompressionLevel.FULL
    summary: Optional[str] = None
    summary_metadata: Optional[Dict[str, Any]] = None
    parent_summary_node_id: Optional[int] = None

    # Node-specific fields
    tokens_used: Optional[int] = None
    expandable: bool = True

    # For AI nodes: structured breakdown of what it contains
    ai_components: Optional[Dict[str, Any]] = (
        None  # {"assistant_text": str, "tool_calls": [...], "tool_results": [...], "errors": [...]}
    )

    # Semantic fields for better retrieval
    topics: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None

    # Relationship fields
    relates_to_node_id: Optional[int] = None  # For follow-ups, corrections, etc.

    @field_validator("summary_metadata", "ai_components", mode="before")
    def parse_json_fields(cls, v):
        """Parse JSON strings to dictionaries for metadata fields."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v

    @field_validator("topics", mode="before")
    def parse_topics(cls, v):
        """Ensure topics is always a list, parsing from JSON if needed."""
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v


class ConversationState(BaseModel):
    """Overall state of a conversation with statistics."""

    conversation_id: str
    name: Optional[str] = None  # Human-readable name for the conversation
    total_nodes: int
    compression_stats: Dict[CompressionLevel, int]
    current_goal: Optional[str] = None
    key_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: Optional[datetime] = None


class CompressionResult(BaseModel):
    """Result of compressing a conversation node."""

    original_node_id: int
    compressed_content: str
    compression_ratio: float
    topics_extracted: List[str]
    metadata: Dict[str, Any]


class SearchResult(BaseModel):
    """Result from searching conversation memory."""

    node: ConversationNode
    relevance_score: float
    match_type: str  # 'content', 'summary', 'topic'
    matched_text: str


class HierarchyThresholds(BaseModel):
    """Thresholds for different compression levels in the advanced hierarchy system."""

    # Number of nodes that trigger compression to next level
    summary_threshold: int = 10  # Recent nodes before compression starts
    meta_threshold: int = 50  # Summary nodes before META grouping
    archive_threshold: int = 200  # META groups before ARCHIVE

    # Group sizes for META level compression
    meta_group_size: int = 20  # Minimum nodes per META group
    meta_group_max: int = 40  # Maximum nodes per META group


class MetaGroup(BaseModel):
    """Represents a group of nodes compressed into META level."""

    start_node_id: int
    end_node_id: int
    start_sequence: int
    end_sequence: int
    node_count: int
    total_lines: int
    main_topics: List[str] = Field(default_factory=list)
    summary: str
    timestamp_range: Tuple[datetime, datetime]
    compression_level: CompressionLevel = CompressionLevel.META

    @field_validator("timestamp_range", mode="before")
    def parse_timestamp_range(cls, v):
        """Ensure timestamp_range is properly parsed."""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (v[0], v[1])
        return v


# ===================================================================
# MODEL PROVIDER CONFIGURATION MODELS
# ===================================================================


class ModelProvider(Enum):
    """Supported model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOONSHOT = "moonshot"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GEMINI = "gemini"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    provider: ModelProvider
    model_name: str = Field(..., description="The actual model name used by the API")
    api_key_env: str = Field(
        ..., description="Environment variable name for the API key"
    )
    base_url: Optional[str] = Field(None, description="Custom base URL for the API")
    default_temperature: float = Field(
        0.3, ge=0.0, le=2.0, description="Default temperature for the model"
    )
    max_tokens: Optional[int] = Field(
        None, gt=0, description="Maximum tokens for the model"
    )
    context_window: Optional[int] = Field(
        None, gt=0, description="Context window size in tokens"
    )
    supports_streaming: bool = Field(
        True, description="Whether the model supports streaming"
    )
    supports_functions: bool = Field(
        True, description="Whether the model supports function calling"
    )
    cost_per_input_token: Optional[float] = Field(
        None, ge=0, description="Cost per input token in USD"
    )
    cost_per_output_token: Optional[float] = Field(
        None, ge=0, description="Cost per output token in USD"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model metadata"
    )

    @field_validator("model_name")
    def validate_model_name(cls, v):
        """Ensure model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("api_key_env")
    def validate_api_key_env(cls, v):
        """Ensure API key environment variable name is valid."""
        if not v or not v.strip():
            raise ValueError("API key environment variable name cannot be empty")
        if not v.isupper():
            raise ValueError("API key environment variable should be uppercase")
        return v.strip()


class ModelInstance(BaseModel):
    """Runtime information about a model instance."""

    config: ModelConfig
    is_available: bool = Field(
        default=False, description="Whether the model is currently available"
    )
    last_checked: Optional[datetime] = Field(
        None, description="When availability was last checked"
    )
    error_message: Optional[str] = Field(
        None, description="Last error message if unavailable"
    )
    usage_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Usage statistics"
    )

    def update_availability(self, available: bool, error: Optional[str] = None) -> None:
        """Update the availability status of this model instance."""
        self.is_available = available
        self.last_checked = datetime.now()
        self.error_message = error

    def add_usage_stats(self, tokens_used: int, cost: Optional[float] = None) -> None:
        """Add usage statistics for this model."""
        if "total_tokens" not in self.usage_stats:
            self.usage_stats["total_tokens"] = 0
        if "total_requests" not in self.usage_stats:
            self.usage_stats["total_requests"] = 0
        if "total_cost" not in self.usage_stats:
            self.usage_stats["total_cost"] = 0.0

        self.usage_stats["total_tokens"] += tokens_used
        self.usage_stats["total_requests"] += 1
        if cost is not None:
            self.usage_stats["total_cost"] += cost


class ModelValidationResult(BaseModel):
    """Result of validating a model configuration."""

    model_name: str
    is_valid: bool
    is_available: bool = False
    error_message: Optional[str] = None
    missing_requirements: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ModelRegistry(BaseModel):
    """Registry of all available model configurations."""

    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)

    def register_model(self, model_name: str, config: ModelConfig) -> None:
        """Register a new model configuration."""
        self.models[model_name] = config
        self.last_updated = datetime.now()

    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self.models.get(model_name)

    def list_models_by_provider(
        self, provider: ModelProvider
    ) -> Dict[str, ModelConfig]:
        """List all models for a specific provider."""
        return {
            name: config
            for name, config in self.models.items()
            if config.provider == provider
        }

    def get_available_providers(self) -> List[ModelProvider]:
        """Get list of all providers that have registered models."""
        return list(set(config.provider for config in self.models.values()))

    def validate_all_models(self) -> List[ModelValidationResult]:
        """Validate all registered models."""
        from .model_manager import (
            ModelManager,
        )  # Import here to avoid circular imports

        results = []
        for model_name, config in self.models.items():
            try:
                is_available = ModelManager.validate_model_access(model_name)
                results.append(
                    ModelValidationResult(
                        model_name=model_name, is_valid=True, is_available=is_available
                    )
                )
            except Exception as e:
                results.append(
                    ModelValidationResult(
                        model_name=model_name, is_valid=False, error_message=str(e)
                    )
                )
        return results


# ===================================================================
# DEFAULT MODEL CONFIGURATIONS
# ===================================================================

# Default model registry with pre-configured models
DEFAULT_MODEL_REGISTRY = ModelRegistry(
    models={
        # Anthropic Claude models
        "claude-sonnet-4": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
            context_window=200000,
            supports_functions=True,
            metadata={"family": "claude-3.5", "tier": "premium"},
        ),
        "claude-3-5-haiku": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-5-haiku-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            context_window=200000,
            supports_functions=True,
            metadata={"family": "claude-3.5", "tier": "fast"},
        ),
        # OpenAI models
        "gpt-4o": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            context_window=128000,
            supports_functions=True,
            cost_per_input_token=0.0025,
            cost_per_output_token=0.01,
            metadata={"family": "gpt-4", "tier": "premium"},
        ),
        "gpt-4o-mini": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            context_window=128000,
            supports_functions=True,
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            metadata={"family": "gpt-4", "tier": "efficient"},
        ),
        "kimi-k2-0711-preview": ModelConfig(
            provider=ModelProvider.MOONSHOT,
            model_name="kimi-k2-0711-preview",
            api_key_env="MOONSHOT_API_KEY",
            base_url="https://api.moonshot.ai/v1",
            context_window=128000,
            supports_functions=True,
            default_temperature=0.3,
            metadata={"family": "moonshot-v1", "tier": "extended"},
        ),
        "moonshot-v1-128k": ModelConfig(
            provider=ModelProvider.MOONSHOT,
            model_name="moonshot-v1-128k",
            api_key_env="MOONSHOT_API_KEY",
            base_url="https://api.moonshot.ai/v1",
            context_window=128000,
            supports_functions=True,
            default_temperature=0.3,
            metadata={"family": "moonshot-v1", "tier": "long_context"},
        ),
        # Together AI models
        "llama-3-8b-instruct": ModelConfig(
            provider=ModelProvider.TOGETHER,
            model_name="meta-llama/Llama-3-8b-chat-hf",
            api_key_env="TOGETHER_API_KEY",
            base_url="https://api.together.xyz/v1",
            context_window=8192,
            supports_functions=False,
            metadata={"family": "llama-3", "size": "8b"},
        ),
        "llama-3-70b-instruct": ModelConfig(
            provider=ModelProvider.TOGETHER,
            model_name="meta-llama/Llama-3-70b-chat-hf",
            api_key_env="TOGETHER_API_KEY",
            base_url="https://api.together.xyz/v1",
            context_window=8192,
            supports_functions=False,
            metadata={"family": "llama-3", "size": "70b"},
        ),
        # DeepSeek models
        "deepseek-chat": ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com/v1",
            context_window=32000,
            supports_functions=True,
            metadata={"family": "deepseek", "specialty": "general"},
        ),
        "deepseek-coder": ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-coder",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com/v1",
            context_window=32000,
            supports_functions=True,
            metadata={"family": "deepseek", "specialty": "coding"},
        ),
        # Google Gemini models
        "gemini-2-5-pro": ModelConfig(
            provider=ModelProvider.GEMINI,
            model_name="gemini-2.5-pro",
            api_key_env="GEMINI_API_KEY",
            context_window=2000000,
            supports_functions=True,
            supports_streaming=True,
            default_temperature=0.3,
            metadata={"family": "gemini-2.5", "tier": "premium", "thinking": True},
        ),
        "gemini-2-5-flash": ModelConfig(
            provider=ModelProvider.GEMINI,
            model_name="gemini-2.5-flash",
            api_key_env="GEMINI_API_KEY",
            context_window=1000000,
            supports_functions=True,
            supports_streaming=True,
            default_temperature=0.3,
            metadata={"family": "gemini-2.5", "tier": "efficient", "thinking": True},
        ),
        "gemini-2-0-flash": ModelConfig(
            provider=ModelProvider.GEMINI,
            model_name="gemini-2.0-flash",
            api_key_env="GEMINI_API_KEY",
            context_window=1000000,
            supports_functions=True,
            supports_streaming=True,
            default_temperature=0.3,
            metadata={"family": "gemini-2.0", "tier": "fast"},
        ),
        "gemini-1-5-pro": ModelConfig(
            provider=ModelProvider.GEMINI,
            model_name="gemini-1.5-pro",
            api_key_env="GEMINI_API_KEY",
            context_window=2000000,
            supports_functions=True,
            supports_streaming=True,
            default_temperature=0.3,
            metadata={"family": "gemini-1.5", "tier": "multimodal"},
        ),
        "gemini-1-5-flash": ModelConfig(
            provider=ModelProvider.GEMINI,
            model_name="gemini-1.5-flash",
            api_key_env="GEMINI_API_KEY",
            context_window=1000000,
            supports_functions=True,
            supports_streaming=True,
            default_temperature=0.3,
            metadata={"family": "gemini-1.5", "tier": "efficient"},
        ),
    }
)
