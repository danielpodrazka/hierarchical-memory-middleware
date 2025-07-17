# Simple MCP Server Integration Plan

## Goal

Add support for external MCP servers (starting with text-editor) to the hierarchical memory middleware through a simple configuration file. Each server runs on its own port.

## What We're Building

A minimal integration that:
1. **Loads a config file** with text-editor server settings
2. **Starts the text-editor MCP server** as an HTTP process
3. **Connects pydantic-ai** to both memory and text-editor servers
4. **Manages the lifecycle** (start/stop) of the external server

## Current vs. Target State

### Current
```python
# Only memory MCP server
manager = HierarchicalConversationManager(
    config,
    mcp_server_url="http://localhost:8000/mcp"  # memory server
)
```

### Target
```python
# Memory + text-editor MCP servers
manager = HierarchicalConversationManager(
    config,
    mcp_server_url="http://localhost:8000/mcp",  # memory server
    external_mcp_servers=[text_editor_client]      # external servers
)
```

## Implementation

### 1. Configuration File

**File**: `~/.config/hierarchical_memory_middleware/mcp_servers.json`

```json
{
  "text-editor": {
    "command": "/home/daniel/pp/venvs/mcp-text-editor/bin/python",
    "args": [
      "/home/daniel/pp/mcp-text-editor/src/text_editor/server.py",
      "--transport", "streamable-http",
      "--port", "8001"
    ],
    "env": {
      "SKIM_MAX_LINES": "200",
      "MAX_SELECT_LINES": "100",
      "PYTHON_VENV": "/home/daniel/pp/venvs/llm-memory-middleware/bin/python",
      "PYTHONPATH": "/home/daniel/pp/llm-memory-middleware"
    },
    "port": 8001,
    "tool_prefix": "text-editor",
    "enabled": true
  }
}
```

### 2. Simple Configuration Loading

**File**: `hierarchical_memory_middleware/config.py`

```python
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

class ExternalMCPServer(BaseModel):
    command: str
    args: List[str]
    env: Dict[str, str] = {}
    port: int
    tool_prefix: str = ""
    enabled: bool = True

class Config:
    # ... existing config ...
    
    @classmethod
    def load_external_mcp_servers(cls) -> Dict[str, ExternalMCPServer]:
        """Load external MCP server configurations."""
        config_path = Path.home() / ".config" / "hierarchical_memory_middleware" / "mcp_servers.json"
        
        if not config_path.exists():
            return {}
            
        with open(config_path) as f:
            data = json.load(f)
            
        servers = {}
        for name, config in data.items():
            servers[name] = ExternalMCPServer(**config)
            
        return servers
```

### 3. Simple MCP Manager

**File**: `hierarchical_memory_middleware/mcp_manager.py`

```python
import asyncio
import logging
import subprocess
import os
from typing import Dict, List, Optional
from pydantic_ai.mcp import MCPServerStreamableHTTP
from .config import ExternalMCPServer

class SimpleMCPManager:
    """Simple manager for external MCP servers."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.clients: Dict[str, MCPServerStreamableHTTP] = {}
        self.logger = logging.getLogger(__name__)

    async def start_server(self, name: str, config: ExternalMCPServer) -> Optional[MCPServerStreamableHTTP]:
        """Start an MCP server and return the client."""
        if not config.enabled:
            return None
            
        try:
            # Start the server process
            env = {**os.environ, **config.env}
            process = subprocess.Popen(
                [config.command] + config.args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes[name] = process
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            # Create the pydantic-ai client
            client = MCPServerStreamableHTTP(
                url=f"http://127.0.0.1:{config.port}/mcp",
                tool_prefix=config.tool_prefix
            )
            
            self.clients[name] = client
            self.logger.info(f"Started MCP server '{name}' on port {config.port}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to start server '{name}': {e}")
            return None

    def stop_all(self):
        """Stop all MCP servers."""
        for name, process in self.processes.items():
            try:
                process.terminate()
                self.logger.info(f"Stopped MCP server '{name}'")
            except Exception as e:
                self.logger.error(f"Error stopping server '{name}': {e}")
                
        self.processes.clear()
        self.clients.clear()

    def get_clients(self) -> List[MCPServerStreamableHTTP]:
        """Get all active MCP clients."""
        return list(self.clients.values())
```

### 4. Update Conversation Manager

**File**: `hierarchical_memory_middleware/middleware/conversation_manager.py`

```python
class HierarchicalConversationManager:
    def __init__(
        self,
        config: Config,
        storage: Optional[DuckDBStorage] = None,
        mcp_server_url: Optional[str] = None,
        external_mcp_servers: Optional[List[MCPServerStreamableHTTP]] = None,  # NEW
    ):
        # ... existing initialization ...

        # Build MCP server list
        mcp_servers = []
        
        # Add memory server if provided
        if mcp_server_url:
            memory_server = MCPServerStreamableHTTP(
                url=mcp_server_url,
                tool_prefix="memory",
                process_tool_call=self._log_tool_call if config.log_tool_calls else None,
            )
            mcp_servers.append(memory_server)
            
        # Add external servers
        if external_mcp_servers:
            mcp_servers.extend(external_mcp_servers)

        # Initialize agent
        agent_kwargs = {
            "model": model_instance,
            "system_prompt": system_prompt,
            "history_processors": [self._hierarchical_memory_processor],
        }
        
        if mcp_servers:
            agent_kwargs["mcp_servers"] = mcp_servers

        self.work_agent = Agent(**agent_kwargs)
        self.has_mcp_tools = bool(mcp_servers)
```

### 5. Update CLI Integration

**File**: `hierarchical_memory_middleware/cli.py`

```python
from .mcp_manager import SimpleMCPManager

async def _chat_session(
    # ... existing parameters ...
):
    # ... existing code ...
    
    # Load and start external MCP servers
    external_servers = config.load_external_mcp_servers()
    mcp_manager = SimpleMCPManager()
    external_clients = []
    
    for name, server_config in external_servers.items():
        client = await mcp_manager.start_server(name, server_config)
        if client:
            external_clients.append(client)
    
    try:
        # Initialize conversation manager with all servers
        manager = HierarchicalConversationManager(
            config,
            mcp_server_url=mcp_server_url,  # memory server
            external_mcp_servers=external_clients  # external servers
        )
        
        # ... existing chat session code ...
        
    finally:
        # Clean up external servers
        mcp_manager.stop_all()
```

## Implementation Steps

### Step 1: Configuration
- [ ] Add `ExternalMCPServer` model to `config.py`
- [ ] Add `load_external_mcp_servers()` method
- [ ] Create example `mcp_servers.json` config file

### Step 2: MCP Manager
- [ ] Create `SimpleMCPManager` class
- [ ] Implement `start_server()` method
- [ ] Implement `stop_all()` method
- [ ] Add basic error handling

### Step 3: Integration
- [ ] Update `HierarchicalConversationManager.__init__()`
- [ ] Add `external_mcp_servers` parameter
- [ ] Combine memory and external servers in agent initialization

### Step 4: CLI Integration
- [ ] Update `_chat_session()` to load external servers
- [ ] Start external servers before conversation
- [ ] Stop external servers after conversation

### Step 5: Testing
- [ ] Test with text-editor server
- [ ] Verify both memory and text-editor tools work
- [ ] Test graceful shutdown

**Total time: ~3 hours**

## Port Management

Each MCP server needs its own port:
- **Memory server**: Port 8000 (existing)
- **Text-editor server**: Port 8001
- **Future servers**: Port 8002, 8003, etc.

The config file specifies the port for each server, and we pass it in the server startup arguments.

## Success Criteria

- [ ] Can start text-editor MCP server via config
- [ ] Both `memory_*` and `text-editor_*` tools available in conversation
- [ ] Can edit files through conversation
- [ ] Servers start/stop cleanly
- [ ] Easy to add new servers to config file

