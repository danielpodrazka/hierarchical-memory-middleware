"""Model management system for different LLM providers."""

import os
from typing import Dict, Any, Optional, Union, List

from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import Pydantic models from models.py
from .models import (
    ModelProvider,
    ModelConfig, 
    ModelInstance,
    ModelValidationResult,
    ModelRegistry,
    DEFAULT_MODEL_REGISTRY
)


class ModelManager:
    """Manages different LLM providers and their configurations using Pydantic models."""

    # Use the default model registry from models.py
    _registry: ModelRegistry = DEFAULT_MODEL_REGISTRY

    @classmethod
    def get_registry(cls) -> ModelRegistry:
        """Get the current model registry."""
        return cls._registry

    @classmethod
    def set_registry(cls, registry: ModelRegistry) -> None:
        """Set a custom model registry."""
        cls._registry = registry

    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return cls._registry.get_model(model_name)

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """List all available models with their providers."""
        return {
            model_name: config.provider.value
            for model_name, config in cls._registry.models.items()
        }

    @classmethod
    def list_models_by_provider(cls, provider: ModelProvider) -> Dict[str, ModelConfig]:
        """List all models for a specific provider."""
        return cls._registry.list_models_by_provider(provider)

    @classmethod
    def get_available_providers(cls) -> List[ModelProvider]:
        """Get list of all providers that have registered models."""
        return cls._registry.get_available_providers()

    @classmethod
    def register_model(cls, model_name: str, config: ModelConfig) -> None:
        """Register a new model configuration."""
        cls._registry.register_model(model_name, config)

    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> Union[Model, str]:
        """Create a PydanticAI compatible model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Model instance compatible with PydanticAI
            
        Raises:
            ValueError: If model is not supported or API key is missing
        """
        config = cls.get_model_config(model_name)
        if not config:
            available_models = list(cls._registry.models.keys())
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Available models: {available_models}"
            )

        # Get API key from environment
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Please set {config.api_key_env} "
                f"in your environment or .env file"
            )

        # Handle Anthropic models (PydanticAI's default)
        if config.provider == ModelProvider.ANTHROPIC:
            # PydanticAI handles Anthropic models natively
            return config.model_name

        # Handle OpenAI-compatible models
        elif config.provider in [
            ModelProvider.OPENAI,
            ModelProvider.MOONSHOT,
            ModelProvider.TOGETHER,
            ModelProvider.DEEPSEEK
        ]:
            # Create AsyncOpenAI client with custom base URL if needed
            client_kwargs = {"api_key": api_key}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url

            async_client = AsyncOpenAI(**client_kwargs)

            # Create OpenAI provider with the async client
            provider = OpenAIProvider(openai_client=async_client)

            # Create model settings for temperature and max_tokens
            from pydantic_ai.settings import ModelSettings
            settings = {}
            
            temperature = kwargs.get("temperature", config.default_temperature)
            if temperature is not None:
                settings["temperature"] = temperature
                
            max_tokens = kwargs.get("max_tokens", config.max_tokens)
            if max_tokens is not None:
                settings["max_tokens"] = max_tokens

            model_settings = ModelSettings(**settings) if settings else None

            # Create PydanticAI OpenAI model with provider
            return OpenAIModel(
                model_name=config.model_name,
                provider=provider,
                settings=model_settings
            )

        else:
            raise ValueError(f"Provider {config.provider} is not yet implemented")

    @classmethod
    def validate_model_access(cls, model_name: str) -> bool:
        """Validate that a model can be accessed (API key exists)."""
        config = cls.get_model_config(model_name)
        if not config:
            return False
        return bool(os.getenv(config.api_key_env))

    @classmethod
    def get_missing_api_keys(cls) -> Dict[str, str]:
        """Get list of models with missing API keys."""
        missing = {}
        for model_name, config in cls._registry.models.items():
            if not os.getenv(config.api_key_env):
                missing[model_name] = config.api_key_env
        return missing

    @classmethod
    def validate_all_models(cls) -> List[ModelValidationResult]:
        """Validate all registered models."""
        return cls._registry.validate_all_models()

    @classmethod
    def create_model_instance(cls, model_name: str) -> ModelInstance:
        """Create a model instance with availability and usage tracking."""
        config = cls.get_model_config(model_name)
        if not config:
            raise ValueError(f"Model '{model_name}' is not supported")
        
        instance = ModelInstance(config=config)
        
        # Check availability
        try:
            is_available = cls.validate_model_access(model_name)
            instance.update_availability(is_available)
        except Exception as e:
            instance.update_availability(False, str(e))
        
        return instance

    @classmethod
    def create_client_for_provider(
        cls, provider: ModelProvider, api_key: str, base_url: Optional[str] = None
    ) -> AsyncOpenAI:
        """Create a raw AsyncOpenAI client for a specific provider."""
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return AsyncOpenAI(**client_kwargs)

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        config = cls.get_model_config(model_name)
        if not config:
            return None
        
        return {
            "name": model_name,
            "provider": config.provider.value,
            "model_name": config.model_name,
            "context_window": config.context_window,
            "supports_functions": config.supports_functions,
            "supports_streaming": config.supports_streaming,
            "cost_per_input_token": config.cost_per_input_token,
            "cost_per_output_token": config.cost_per_output_token,
            "metadata": config.metadata,
            "is_available": cls.validate_model_access(model_name)
        }

    @classmethod
    def get_models_by_capability(cls, capability: str) -> List[str]:
        """Get models that support a specific capability."""
        matching_models = []
        
        for model_name, config in cls._registry.models.items():
            if capability == "functions" and config.supports_functions:
                matching_models.append(model_name)
            elif capability == "streaming" and config.supports_streaming:
                matching_models.append(model_name)
            elif capability == "long_context" and config.context_window and config.context_window > 32000:
                matching_models.append(model_name)
        
        return matching_models

    @classmethod
    def estimate_cost(
        cls, model_name: str, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        """Estimate the cost for using a model with given token counts."""
        config = cls.get_model_config(model_name)
        if not config or not config.cost_per_input_token or not config.cost_per_output_token:
            return None
        
        input_cost = input_tokens * config.cost_per_input_token
        output_cost = output_tokens * config.cost_per_output_token
        return input_cost + output_cost
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAI(**client_kwargs)
