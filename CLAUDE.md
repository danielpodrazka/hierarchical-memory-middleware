# CLAUDE.md - Project Context for AI Assistants

This file provides context for AI assistants (like Claude Code) working on the Hierarchical Memory Middleware project.

## Project Overview

**Hierarchical Memory Middleware** is a Python library that provides hierarchical compression for AI conversations, enabling effectively unlimited conversation length by intelligently compressing older messages while maintaining access to full details via MCP tools.

### Key Value Proposition

- **For Claude Max/Pro subscribers**: Use your subscription instead of API credits
- **Infinite conversations**: Automatic compression of older messages
- **Memory tools**: AI can search and expand compressed content

## Architecture

### Two Conversation Managers

1. **ClaudeAgentSDKConversationManager** (Default, Recommended)
   - Uses Claude Agent SDK (spawns Claude CLI subprocess)
   - Works with Claude Pro/Max subscription (no API credits needed)
   - Memory tools provided via stdio MCP subprocess
   - Location: `hierarchical_memory_middleware/middleware/claude_agent_sdk_manager.py`

2. **HierarchicalConversationManager** (API-based)
   - Uses PydanticAI for LLM orchestration
   - Requires API keys and credits
   - Memory tools via HTTP MCP server
   - Location: `hierarchical_memory_middleware/middleware/conversation_manager.py`

### Factory Function

```python
from hierarchical_memory_middleware.middleware import create_conversation_manager

# Auto-selects based on model name:
# - "claude-agent-*" -> ClaudeAgentSDKConversationManager
# - Others -> HierarchicalConversationManager
manager = create_conversation_manager()
```

### Key Directories

```
hierarchical_memory_middleware/
├── middleware/                    # Conversation managers
│   ├── __init__.py               # Factory function
│   ├── claude_agent_sdk_manager.py  # Claude Agent SDK manager
│   └── conversation_manager.py    # PydanticAI-based manager
├── mcp_server/                   # MCP server implementations
│   ├── stdio_memory_server.py    # Stdio server for Agent SDK
│   ├── memory_server.py          # HTTP server for API models
│   └── run_server.py             # Standalone server runner
├── storage/                      # DuckDB storage layer
├── compression/                  # Compression algorithms
├── config.py                     # Configuration management
├── models.py                     # Model definitions and registry
├── model_manager.py              # Model validation and access
└── cli.py                        # Command-line interface
```

### Compression Levels

```
FULL → SUMMARY → META → ARCHIVE
```

- **FULL**: Recent messages, complete content
- **SUMMARY**: Older messages, truncated with TF-IDF topics
- **META**: Groups of summaries
- **ARCHIVE**: Highly compressed historical context

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_specific.py -v

# Format code
uv run black .
uv run ruff check --fix .

# Type checking
uv run mypy .

# Run CLI chat
uv run python -m hierarchical_memory_middleware.cli chat

# Run CLI with specific model
uv run python -m hierarchical_memory_middleware.cli chat --model claude-agent-sonnet
```

## Configuration

### Environment Variables

```bash
# Model selection (default: claude-agent-sonnet)
WORK_MODEL=claude-agent-sonnet

# Claude Agent SDK settings
AGENT_PERMISSION_MODE=default      # default, acceptEdits, bypassPermissions
AGENT_USE_SUBSCRIPTION=true        # Use subscription instead of API credits

# Compression thresholds
RECENT_NODE_LIMIT=10               # Nodes kept at FULL level
SUMMARY_THRESHOLD=20               # When to compress to SUMMARY
META_SUMMARY_THRESHOLD=50          # When to create META groups
ARCHIVE_THRESHOLD=200              # When to archive

# Storage
DB_PATH=./conversations.db

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### Model Names

- **Claude Agent SDK**: `claude-agent-opus`, `claude-agent-sonnet`, `claude-agent-haiku`
- **Anthropic API**: `claude-sonnet-4`, `claude-opus-4`, `claude-haiku-35`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`
- **Others**: See `models.py` for full list

## Memory Tools (MCP)

### For Claude Agent SDK (stdio subprocess)

Tools automatically available to the AI:
- `mcp__memory__expand_node(node_id)` - Get full content of a node
- `mcp__memory__search_memory(query, limit)` - Search conversation history
- `mcp__memory__get_memory_stats()` - Get conversation statistics
- `mcp__memory__get_recent_nodes(count)` - Get recent messages

### For API Models (HTTP server)

Requires running MCP server:
```bash
python -m hierarchical_memory_middleware.mcp_server.run_server
```

## Key Implementation Details

### Subscription Mode Workaround

The Claude Agent SDK doesn't officially support using subscriptions instead of API credits, but clearing the `ANTHROPIC_API_KEY` environment variable forces the CLI to use OAuth credentials from `~/.claude/.credentials.json`:

```python
# In claude_agent_sdk_manager.py
if self.use_subscription:
    env["ANTHROPIC_API_KEY"] = ""  # Force OAuth usage
```

### Stdio Memory Server

For Claude Agent SDK, memory tools are provided via a stdio subprocess:

```python
mcp_servers["memory"] = {
    "command": sys.executable,
    "args": [
        "-m", "hierarchical_memory_middleware.mcp_server.stdio_memory_server",
        "--conversation-id", self.conversation_id,
        "--db-path", self.config.db_path,
    ],
    "env": env,
}
```

## Common Tasks

### Adding a New Model

1. Add to `DEFAULT_MODEL_REGISTRY` in `models.py`
2. Add API key env var to `Config.from_env()` if needed
3. Update `model_manager.py` validation if needed

### Adding a New Memory Tool

1. For Agent SDK: Add to `stdio_memory_server.py`
2. For API models: Add to `memory_server.py`
3. Update allowed tools list in respective manager

### Modifying Compression Logic

- TF-IDF compression: `compression/tfidf_compressor.py`
- Compression triggers: `advanced_hierarchy.py`
- Thresholds: `config.py` and `models.py`

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=hierarchical_memory_middleware

# Run specific test
uv run pytest tests/test_claude_agent_sdk.py -v
```

## Debugging

### Enable Debug Logging

```bash
DEBUG_MODE=true python -m hierarchical_memory_middleware.cli chat
```

### Check Memory State

```python
# In a conversation
manager = create_conversation_manager()
await manager.start_conversation()

# Get stats
stats = await manager.get_conversation_stats()
print(stats)

# Search memory
results = await manager.search_memory("topic")
print(results)
```

## Important Notes

- Claude Agent SDK requires Claude CLI to be installed and authenticated (`claude login`)
- The stdio memory server logs to `/tmp/hmm_stdio_server.log`
- DuckDB database is created automatically at the configured `db_path`
- Compression happens automatically when thresholds are exceeded
