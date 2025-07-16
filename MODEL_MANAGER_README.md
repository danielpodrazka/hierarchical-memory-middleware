# Model Manager - Multi-Provider LLM Support

The Hierarchical Memory Middleware supports multiple LLM providers through a unified model manager system. You can easily switch between different models and providers by simply changing the model name in your configuration.

## 🚀 Quick Start with Moonshot

To use Moonshot models, follow these steps:

1. **Set your API key** in `.env`:
   ```bash
   MOONSHOT_API_KEY=your_moonshot_api_key_here
   ```

2. **Update your model configuration** in `.env`:
   ```bash
   WORK_MODEL=moonshot-v1-128k
   SUMMARY_MODEL=moonshot-v1-128k
   ```

3. **Use it in your code** - no changes needed!
   ```python
   # Your existing code works unchanged
   config = Config.from_env()
   manager = HierarchicalConversationManager(config)
   ```

## 📋 Supported Models

### Anthropic Claude
- `claude-4-sonnet` (Claude 3.5 Sonnet)
- `claude-4-haiku` (Claude 3.5 Haiku)
- **API Key**: `ANTHROPIC_API_KEY`

### OpenAI
- `gpt-4o`
- `gpt-4o-mini`
- **API Key**: `OPENAI_API_KEY`

### Moonshot (Kimi)
- `moonshot-v1-8k` (8k context)
- `moonshot-v1-32k` (32k context)
- `moonshot-v1-128k` (128k context)
- **API Key**: `MOONSHOT_API_KEY`
- **Base URL**: `https://api.moonshot.ai/v1`

### Together AI
- `llama-3-8b-instruct`
- `llama-3-70b-instruct`
- **API Key**: `TOGETHER_API_KEY`
- **Base URL**: `https://api.together.xyz/v1`

### DeepSeek
- `deepseek-chat`
- `deepseek-coder`
- **API Key**: `DEEPSEEK_API_KEY`
- **Base URL**: `https://api.deepseek.com/v1`

## 🔧 Configuration Examples

### Moonshot Configuration
```bash
# .env file
WORK_MODEL=moonshot-v1-128k
SUMMARY_MODEL=moonshot-v1-128k
MOONSHOT_API_KEY=your_moonshot_api_key
```

### Mixed Provider Setup
```bash
# Use Claude for main work, GPT-4o-mini for summaries
WORK_MODEL=claude-4-sonnet
SUMMARY_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
```

### DeepSeek for Coding
```bash
WORK_MODEL=deepseek-coder
SUMMARY_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_deepseek_key
```

## 💻 Programmatic Usage

### Basic Usage
```python
from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.middleware.conversation_manager import HierarchicalConversationManager

# Simple - just specify the model name
config = Config(work_model="moonshot-v1-128k")
manager = HierarchicalConversationManager(config)

# Everything else works the same!
conversation_id = await manager.start_conversation()
response = await manager.chat("Hello!")
```

### Advanced Model Manager Usage
```python
from hierarchical_memory_middleware.model_manager import ModelManager

# List all available models
models = ModelManager.list_available_models()
print(models)  # {'moonshot-v1-128k': 'moonshot', 'claude-4-sonnet': 'anthropic', ...}

# Check if you have the required API keys
if ModelManager.validate_model_access("moonshot-v1-128k"):
    print("✅ Moonshot is ready to use!")
else:
    print("❌ Please set MOONSHOT_API_KEY")

# Get missing API keys
missing = ModelManager.get_missing_api_keys()
for model, key_name in missing.items():
    print(f"Model {model} needs {key_name}")
```

### Create Models Directly
```python
# Create a model instance directly
model = ModelManager.create_model("moonshot-v1-128k", temperature=0.5)

# Use with PydanticAI Agent
from pydantic_ai import Agent
agent = Agent(model=model, system_prompt="You are a helpful assistant")
```

## 🔍 Model Provider Details

### How It Works
The model manager automatically:
1. **Detects the provider** based on the model name
2. **Loads the appropriate API key** from environment variables
3. **Creates the correct client** (OpenAI-compatible for most providers)
4. **Configures the base URL** for non-OpenAI providers
5. **Returns a PydanticAI-compatible model instance**

### Adding New Models
To add support for a new model:

```python
from hierarchical_memory_middleware.model_manager import ModelManager, ModelConfig, ModelProvider

# Register a new model
ModelManager.register_model(
    "new-model-name",
    ModelConfig(
        provider=ModelProvider.OPENAI,  # or create new provider
        model_name="actual-api-model-name",
        api_key_env="NEW_API_KEY",
        base_url="https://api.newprovider.com/v1"  # if needed
    )
)
```

## 🐛 Troubleshooting

### Common Issues

**Error: "Model 'xyz' is not supported"**
- Check the model name spelling
- Use `ModelManager.list_available_models()` to see supported models

**Error: "API key not found"**
- Make sure you've set the correct environment variable
- Check `.env.example` for the exact variable name needed
- Verify your `.env` file is in the correct location

**Error: "Provider X is not yet implemented"**
- Currently supported: Anthropic, OpenAI, Moonshot, Together AI, DeepSeek
- File an issue for additional provider support

### Debugging
```python
# Check what's available
from hierarchical_memory_middleware.model_manager import ModelManager

print("Available models:", ModelManager.list_available_models())
print("Missing API keys:", ModelManager.get_missing_api_keys())

# Test a specific model
try:
    model = ModelManager.create_model("moonshot-v1-128k")
    print("✅ Model created successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
```

## 🚀 Example Usage Script

Run the included example script to test your setup:

```bash
python example_usage.py
```

This script will:
- Show all available models
- Check which ones have API keys configured
- Run a sample conversation with Moonshot (if configured)
- Test multiple models if available

## 🔮 Future Extensions

The model manager is designed to be easily extensible. Planned additions:
- Support for more providers (Cohere, Mistral, etc.)
- Model-specific parameter tuning
- Automatic failover between models
- Cost tracking per provider
- Model performance analytics

---

*The model manager maintains full backward compatibility - existing code will continue to work unchanged!*
