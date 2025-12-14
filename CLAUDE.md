# CLAUDE.md - Project Context for AI Assistants

This file provides context for AI assistants (like Claude Code) working on the Hierarchical Memory Middleware project.

## Project Overview

**Hierarchical Memory Middleware** is a Python library that enables infinite AI conversations through intelligent hierarchical compression. It preserves access to full conversation details via MCP tools while automatically compressing older messages.

### Key Value Proposition

- **Claude Pro/Max subscribers**: Use your subscription instead of API credits via Claude Agent SDK
- **Infinite conversations**: Automatic FULL → SUMMARY → META → ARCHIVE compression
- **Memory tools**: AI can search, expand compressed content, and maintain a scratchpad

## Architecture

### Two Conversation Managers

The system uses a factory pattern to select the appropriate manager:

```python
from hierarchical_memory_middleware import create_conversation_manager

# Auto-selects based on model name:
# - "claude-agent-*" -> ClaudeAgentSDKConversationManager
# - Others -> HierarchicalConversationManager
manager = create_conversation_manager()
```

1. **ClaudeAgentSDKConversationManager** (Default)
   - Uses Claude CLI subprocess with OAuth authentication
   - Works with Claude Pro/Max subscription (no API credits)
   - Memory tools via stdio MCP subprocess
   - Location: `middleware/claude_agent_sdk_manager.py`

2. **HierarchicalConversationManager** (API-based)
   - Uses PydanticAI for LLM orchestration
   - Requires API keys and credits
   - Memory tools via HTTP MCP server
   - Location: `middleware/conversation_manager.py`

### Directory Structure

```
hierarchical_memory_middleware/
├── middleware/                    # Conversation managers
│   ├── __init__.py               # Factory: create_conversation_manager()
│   ├── claude_agent_sdk_manager.py  # Claude CLI (Pro/Max subscription)
│   └── conversation_manager.py    # PydanticAI (API keys)
├── mcp_server/                   # MCP server implementations
│   ├── stdio_memory_server.py    # stdio transport for Agent SDK
│   ├── memory_server.py          # HTTP transport for API models
│   └── run_server.py             # Standalone server runner
├── storage.py                    # DuckDB storage with VSS extension
├── compression.py                # TfidfCompressor, CompressionManager
├── advanced_hierarchy.py         # AdvancedCompressionManager
├── embeddings.py                 # Optional semantic search
├── models.py                     # Data models & DEFAULT_MODEL_REGISTRY
├── model_manager.py              # Model validation & access
├── config.py                     # Config class with from_env()
├── db_utils.py                   # Database connection utilities
├── mcp_manager.py                # External MCP server management
└── cli.py                        # Typer CLI with Rich formatting
```

### Compression Levels

```
FULL (recent 10) → SUMMARY (50) → META (200) → ARCHIVE
```

- **FULL**: Complete content, all tool calls preserved
- **SUMMARY**: First N words + TF-IDF extracted topics
- **META**: Groups of 20-40 summaries with theme extraction
- **ARCHIVE**: Highly compressed historical context

### Memory Tools (MCP)

Available to AI during conversations:

| Tool | Description |
|------|-------------|
| `expand_node(node_id)` | Get full content of compressed node |
| `search_memory(query, limit, mode)` | Search history (keyword/semantic/hybrid) |
| `get_memory_stats()` | Compression statistics |
| `get_recent_nodes(count)` | Recent messages in full |
| `get_system_prompt()` | Read scratchpad |
| `set_system_prompt(content)` | Replace scratchpad |
| `append_to_system_prompt(content)` | Add to scratchpad |
| `yield_to_human(reason)` | Signal need for human input (agentic mode) |
| `backfill_embeddings()` | Generate embeddings for semantic search |

### Agentic Mode

Agentic mode enables autonomous AI operation with explicit handoff control. The AI works continuously until it calls `yield_to_human()` or the user interrupts.

**CLI Flag:**
```bash
uv run python -m hierarchical_memory_middleware.cli chat --agentic
```

**How it works:**

1. User sends initial request
2. System sends "continue" message automatically after each AI response
3. AI keeps working until it calls `yield_to_human(reason)`
4. System pauses for human input
5. User responds, cycle repeats

**Key Implementation Details:**

- System prompt injection (`claude_agent_sdk_manager.py:589-606`) instructs AI about agentic mode
- CLI chat loop (`cli.py:764-977`) handles auto-continue vs yield detection
- `yield_to_human` tool (`stdio_memory_server.py:322-347`) is a signal, not a control flow tool

**Interrupt Handling:**
- Single `Ctrl+C`: Sets `interrupted=True`, `yielded_to_human=True`, pauses for input
- Double `Ctrl+C`: Exits session

**When AI should call `yield_to_human()`:**
- Task is complete
- Needs clarification or decision
- Blocked on missing information
- Reached a natural checkpoint

## Development Commands

```bash
# Install dependencies
uv sync
uv sync --dev          # Include dev dependencies
uv sync --extra embeddings  # Include semantic search

# Run tests
uv run pytest
uv run pytest tests/test_storage.py -v  # Specific test
uv run pytest --cov=hierarchical_memory_middleware  # With coverage

# Format code
uv run black .
uv run ruff check --fix .

# Type checking
uv run mypy .

# Run CLI
uv run python -m hierarchical_memory_middleware.cli chat
uv run python -m hierarchical_memory_middleware.cli chat --model gpt-4o

# Run standalone MCP server (for API models)
uv run python -m hierarchical_memory_middleware.mcp_server.run_server
```

## Configuration

### Environment Variables (.env)

```bash
# Model selection (default: claude-agent-sonnet)
WORK_MODEL=claude-agent-sonnet

# Claude Agent SDK settings
AGENT_PERMISSION_MODE=default      # default, acceptEdits, bypassPermissions
AGENT_USE_SUBSCRIPTION=true        # Use subscription instead of API credits
AGENT_ALLOWED_TOOLS=               # Comma-separated allowed tools

# API keys (for API-based models)
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
MOONSHOT_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
TOGETHER_API_KEY=your_key

# Compression thresholds
RECENT_NODE_LIMIT=10
SUMMARY_THRESHOLD=20
META_SUMMARY_THRESHOLD=50
ARCHIVE_THRESHOLD=200

# Storage
DB_PATH=./conversations.db
MCP_PORT=8000

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
LOG_FILE=hierarchical_memory.log
```

### Model Names

**Claude Agent SDK** (subscription):
- `claude-agent-opus`, `claude-agent-sonnet`, `claude-agent-haiku`

**Anthropic API**: `claude-sonnet-4`, `claude-3-5-haiku`

**OpenAI**: `gpt-4o`, `gpt-4o-mini`

**Google**: `gemini-2-5-pro`, `gemini-2-5-flash`, `gemini-2-0-flash`, `gemini-1-5-pro`, `gemini-1-5-flash`

**Moonshot**: `kimi-k2-0711-preview`, `moonshot-v1-128k`

**DeepSeek**: `deepseek-chat`, `deepseek-coder`

**Together**: `llama-3-8b-instruct`, `llama-3-70b-instruct`

## Common Tasks

### Adding a New Model

1. Add to `DEFAULT_MODEL_REGISTRY` in `models.py`:
   ```python
   "new-model": ModelConfig(
       provider=ModelProvider.PROVIDER_NAME,
       model_name="actual-api-model-name",
       api_key_env="PROVIDER_API_KEY",
       context_window=128000,
       supports_functions=True,
   ),
   ```
2. Add API key env var to `Config.from_env()` if needed
3. Update `model_manager.py` validation if needed

### Adding a New Memory Tool

1. **For Claude Agent SDK**: Add to `mcp_server/stdio_memory_server.py`
2. **For API models**: Add to `mcp_server/memory_server.py`
3. Update allowed tools list in `claude_agent_sdk_manager.py` if needed

### Modifying Compression Logic

- **TF-IDF compression**: `compression.py` - `TfidfCompressor` class
- **Compression triggers**: `advanced_hierarchy.py` - `AdvancedCompressionManager`
- **Thresholds**: `config.py` (Config dataclass) and `models.py` (HierarchyThresholds)

## Key Implementation Details

### Subscription Mode (Claude Agent SDK)

The Claude Agent SDK doesn't officially expose subscription mode, but clearing `ANTHROPIC_API_KEY` forces CLI to use OAuth from `~/.claude/.credentials.json`:

```python
# In claude_agent_sdk_manager.py
if self.use_subscription:
    env["ANTHROPIC_API_KEY"] = ""  # Force OAuth usage
```

### Stdio Memory Server

For Claude Agent SDK, memory tools run as a subprocess:

```python
mcp_servers["memory"] = {
    "command": sys.executable,
    "args": ["-m", "hierarchical_memory_middleware.mcp_server.stdio_memory_server",
             "--conversation-id", self.conversation_id,
             "--db-path", self.config.db_path],
    "env": env,
}
```

### DuckDB with VSS

Storage uses DuckDB's VSS extension for vector similarity search:
- Installed/loaded automatically if semantic search is enabled
- Falls back to brute-force if VSS fails to load
- Embeddings generated via sentence-transformers (default: all-MiniLM-L6-v2)

## Testing

```bash
# All tests
uv run pytest

# With verbose output
uv run pytest -v

# Specific test file
uv run pytest tests/test_storage.py

# With coverage report
uv run pytest --cov=hierarchical_memory_middleware --cov-report=html

# Skip slow tests
uv run pytest -m "not slow"
```

### Test Files

- `test_storage.py` - DuckDB storage operations
- `test_compression.py` - TF-IDF compression
- `test_advanced_hierarchy.py` - Multi-level compression
- `test_conversation_manager.py` - PydanticAI manager
- `test_claude_agent_sdk.py` - Claude Agent SDK manager
- `test_mcp_server.py` - MCP server tools
- `test_config.py` - Configuration loading

## Debugging

### Enable Debug Logging

```bash
DEBUG_MODE=true uv run python -m hierarchical_memory_middleware.cli chat
```

### Check Stdio Server Logs

```bash
tail -f /tmp/hmm_stdio_server.log
```

### Inspect Database

```python
import duckdb
conn = duckdb.connect("conversations.db")
conn.execute("SELECT * FROM nodes LIMIT 10").fetchall()
conn.execute("SELECT * FROM conversations").fetchall()
```

## Important Notes

- Claude Agent SDK requires Claude CLI installed and authenticated (`claude login`)
- DuckDB database created automatically at configured `db_path`
- Compression happens automatically when thresholds exceeded
- System prompt/scratchpad persists per conversation in database
- Embeddings are optional - install with `uv sync --extra embeddings`
