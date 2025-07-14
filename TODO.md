# Hierarchical Memory Agent: Implementation Specification v2

## Executive Summary

A middleware system that enables infinite AI agent conversations through intelligent hierarchical compression, preserving full conversation fidelity while maintaining optimal context windows. Built as a conversation orchestration layer using PydanticAI for LLM interfaces and MCP for memory browsing tools.

## Core Architecture

### System Overview

```
User Input
    ↓
[Hierarchical Memory Middleware]
    ├── Compress old conversations into summaries
    ├── Maintain hierarchical structure (4 levels)
    ├── Prepare optimal context for LLM
    └── Store everything in DuckDB
    ↓
LLM (via PydanticAI)
    ↓
Response
    ↓
[MCP Memory Tools] (Read-only, called by LLM during response)
    ├── expand_node(id)
    ├── search_memory(query)
    └── browse_hierarchy(level)
```

### Key Components

```python
HierarchicalMemorySystem/
├── middleware/
│   ├── conversation_manager.py    # Main orchestration
│   ├── hierarchy_builder.py       # Manages compression levels
│   └── context_optimizer.py      # Prepares optimal context
├── storage/
│   ├── duckdb_store.py           # Persistent storage
│   └── models.py                 # Data models
├── mcp_server/
│   ├── memory_tools.py           # Read-only MCP tools
│   └── server.py                 # MCP server setup
├── llm/
│   ├── agents.py                 # PydanticAI agents
│   └── summarizer.py             # Compression logic
└── config.py                     # Configuration
```

## Data Models

### Node Definition

A **conversation node** is the fundamental unit of conversation storage. Each node represents a single message in the conversation and can be one of two types:

1. **User Node**: Contains a user message and optionally file attachments
2. **AI Assistant Node**: Contains an AI response and optionally includes tool calls with their results

Nodes are stored sequentially and can be compressed at different levels while maintaining full retrievability through the MCP expand_node() tool.

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class CompressionLevel(Enum):
    FULL = 0          # Recent nodes - complete content
    SUMMARY = 1       # Older nodes - 1-sentence summaries
    META = 2          # Groups of summaries (20-40 nodes each group)
    ARCHIVE = 3       # Very old - high-level context

class NodeType(Enum):
    USER = "user"
    AI = "ai"

@dataclass
class ConversationNode:
    id: int
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
    parent_summary_id: Optional[int] = None

    # Node-specific fields
    tokens_used: Optional[int] = None
    importance_score: Optional[float] = None
    expandable: bool = True

    # For AI nodes: structured breakdown of what it contains
    ai_components: Optional[Dict[str, Any]] = None  # {"assistant_text": str, "tool_calls": [...], "tool_results": [...], "errors": [...]}

    # Semantic fields for better retrieval
    topics: List[str] = None
    embedding: Optional[List[float]] = None

    # Relationship fields
    relates_to_node_id: Optional[int] = None  # For follow-ups, corrections, etc.

@dataclass
class ConversationState:
    conversation_id: str
    total_nodes: int
    compression_stats: Dict[CompressionLevel, int]
    current_goal: Optional[str] = None
    key_decisions: List[Dict[str, Any]] = None
    last_updated: datetime = None
```

## Core Implementation

### 1. Middleware Layer

```python
from pydantic_ai import Agent
from typing import List, Dict, Optional

class HierarchicalConversationManager:
    def __init__(self, config: Config):
        # Two agents: expensive for work, cheap for summaries
        self.work_agent = Agent(config.work_model)  # e.g., claude-3.5-sonnet
        self.summary_agent = Agent(config.summary_model)  # e.g., claude-3.5-haiku
        
        # Storage and memory management
        self.storage = DuckDBStorage(config.db_path)
        self.hierarchy = HierarchyBuilder(config.hierarchy_config)
        
        # Configuration
        self.config = config
        self.conversation_id = None
        
    async def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Initialize or resume a conversation"""
        if conversation_id and self.storage.conversation_exists(conversation_id):
            self.conversation_id = conversation_id
            self.hierarchy.load_conversation(conversation_id)
        else:
            self.conversation_id = str(uuid.uuid4())
            self.hierarchy.initialize_new_conversation()
            
        return self.conversation_id
    
    async def chat(self, user_message: str) -> str:
        """Main conversation interface"""
        # 1. Prepare hierarchical context
        context = await self._prepare_optimized_context(user_message)
        
        # 2. Generate response
        response = await self.work_agent.run(
            user_message,
            message_history=context.messages,
            system_prompt=context.system_prompt
        )
        
        # 3. Store the node
        node = await self.storage.save_node(
            user_message=user_message,
            ai_response=response.data,
            conversation_id=self.conversation_id,
            tokens_used=response.usage.total_tokens
        )

        # 4. Update hierarchy if needed
        await self._update_hierarchy(node)
        
        return response.data
    
    async def _prepare_optimized_context(self, current_message: str) -> Context:
        """Build optimal context from hierarchy"""
        # Get base context from hierarchy
        hierarchy_state = self.hierarchy.get_current_state()
        
        # Optionally use summary agent to determine relevance
        if self.config.use_smart_context:
            relevance_check = await self.summary_agent.run(
                f"What context is most relevant for: {current_message}",
                context={"available_summaries": hierarchy_state.summaries}
            )
            
        return Context(
            messages=self._build_message_history(hierarchy_state, relevance_check),
            system_prompt=self._build_system_prompt(hierarchy_state)
        )
```

### 2. Hierarchy Management

```python
class HierarchyBuilder:
    def __init__(self, config: HierarchyConfig):
        self.config = config
        self.storage = None  # Injected
        self.summary_agent = None  # Injected
        
    async def compress_old_nodes(self, nodes: List[ConversationNode]) -> List[ConversationNode]:
        """Compress nodes into summaries"""
        compressed = []

        for node in nodes:
            if self._should_compress(node):
                summary = await self._create_summary(node)
                node.level = CompressionLevel.SUMMARY
                node.summary = summary.text
                node.summary_metadata = {
                    "importance": summary.importance,
                    "topics": summary.topics,
                    "has_decision": summary.has_decision,
                    "has_code": summary.has_code
                }
                compressed.append(node)

        return compressed
    
    async def _create_summary(self, node: ConversationNode) -> Summary:
        """Create intelligent summary of a conversation node"""
        prompt = f"""
        Summarize this conversation node in 1-2 sentences:

        Node Type: {node.node_type.value}
        Content: {node.content[:1000]}...

        Focus on: decisions made, problems solved, code changes, key insights.
        If this contains critical information, mark as [IMPORTANT].
        Extract key topics discussed.
        """

        result = await self.summary_agent.run(prompt)
        return self._parse_summary_response(result.data)
```

### 3. MCP Memory Tools

```python
from mcp import MCPServer, mcp_tool

class MemoryBrowsingServer(MCPServer):
    def __init__(self, storage: DuckDBStorage):
        super().__init__("hierarchical-memory")
        self.storage = storage
        
    @mcp_tool()
    async def expand_node(self, node_id: int) -> Dict:
        """Retrieve full content of a summarized conversation node"""
        node = await self.storage.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        return {
            "node_id": node.id,
            "node_type": node.node_type.value,
            "content": node.content,
            "timestamp": node.timestamp.isoformat(),
            "sequence_number": node.sequence_number,
            "line_count": node.line_count,
            "summary": node.summary,
            "metadata": node.summary_metadata,
            "ai_components": node.ai_components,
            "relates_to": node.relates_to_node_id
        }

    @mcp_tool()
    async def search_memory(self, query: str, limit: int = 10) -> List[Dict]:
        """Search across all conversation history"""
        results = await self.storage.semantic_search(query, limit)

        return [{
            "node_id": node.id,
            "node_type": node.node_type.value,
            "summary": node.summary or self._truncate(node.content),
            "timestamp": node.timestamp.isoformat(),
            "line_count": node.line_count,
            "relevance_score": score
        } for node, score in results]

    @mcp_tool()
    async def browse_hierarchy(self, level: int = 0) -> Dict:
        """Browse conversation at specific compression level"""
        nodes = await self.storage.get_nodes_by_level(CompressionLevel(level))

        return {
            "level": level,
            "level_name": CompressionLevel(level).name,
            "node_count": len(nodes),
            "nodes": [self._node_to_browse_format(n) for n in nodes[:50]]
        }

    @mcp_tool()
    async def get_conversation_stats(self) -> Dict:
        """Get overview of conversation memory state"""
        stats = await self.storage.get_conversation_stats()

        return {
            "total_nodes": stats.total_nodes,
            "user_nodes": stats.user_nodes,
            "ai_nodes": stats.ai_nodes,
            "compression_levels": {
                level.name: count
                for level, count in stats.level_counts.items()
            },
            "total_tokens_saved": stats.tokens_saved,
            "compression_ratio": stats.compression_ratio,
            "conversation_duration_days": stats.duration_days
        }
```

### 4. Storage Layer

```python
import duckdb
from typing import List, Optional, Dict, Tuple

class DuckDBStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()
        
    def _init_schema(self):
        """Create database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Compression fields
                level INTEGER DEFAULT 0,
                summary TEXT,
                summary_metadata JSON,
                parent_summary_id INTEGER,
                
                -- Tracking fields
                tokens_used INTEGER,
                importance_score REAL,
                tool_calls JSON,
                expandable BOOLEAN DEFAULT TRUE,
                
                -- Semantic fields
                topics JSON,
                embedding FLOAT[],
                
                -- Indexes
                INDEX idx_conversation (conversation_id),
                INDEX idx_level (level),
                INDEX idx_timestamp (timestamp),
                INDEX idx_parent (parent_summary_id)
            )
        """)
        
    async def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[Turn, float]]:
        """Search using embeddings and return relevance scores"""
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)
        
        # Use DuckDB's array similarity functions
        results = self.conn.execute("""
            SELECT *, 
                   array_cosine_similarity(embedding, ?::FLOAT[]) as score
            FROM nodes
            WHERE embedding IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """, (query_embedding, limit)).fetchall()
        
        return [(self._row_to_node(row), row[-1]) for row in results]
```

## Configuration

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model configuration
    work_model: str = "claude-4-sonnet"
    summary_model: str = "claude-4-sonnet"
    embedding_model: str = "text-embedding-3-small"
    
    # Hierarchy configuration  
    recent_node_limit: int = 10
    summary_threshold: int = 20
    meta_summary_threshold: int = 50
    archive_threshold: int = 200
    
    # Storage
    db_path: str = "./conversations.db"
    backup_interval_hours: int = 24
    
    # MCP server
    mcp_port: int = 8000
    enable_mcp_tools: bool = True
```

## Usage Example

```python
# Initialize the system
system = HierarchicalConversationManager(Config())

# Start a conversation
conversation_id = await system.start_conversation()

# Chat naturally - system handles compression automatically
response = await system.chat("Let's build an authentication system")

# ... many messages later ...

response = await system.chat("What approach did we decide on for token refresh?")
# System automatically includes relevant compressed context
# LLM can call expand_node() via MCP if it needs more detail

# Check memory state
stats = await system.get_stats()
print(f"Conversation has {stats['total_nodes']} nodes ({stats['user_nodes']} user, {stats['ai_nodes']} AI)")
print(f"Compression saved {stats['tokens_saved']} tokens")
```

### Conversation Flow Example

**1. Original conversation (early nodes):**
```
Node 1 (user): Let's build a user authentication system for my web app

Node 2 (ai): I'll help you build a secure authentication system. Let's start with the requirements...
[Assistant response with multiple paragraphs explaining different auth approaches]

Node 3 (user): I want JWT tokens with refresh functionality
Also, should I use bcrypt for password hashing?

Node 4 (ai): Great questions! For JWT with refresh tokens, here's the approach...
[tool call: web_search('bcrypt vs argon2 password hashing 2024')]
[tool result: Found comparison of bcrypt vs Argon2. Argon2 is now recommended...]
Based on current research, I'd recommend Argon2 over bcrypt for new projects...
[Full implementation details with code examples]

Node 5 (user): How should I handle token expiration on the frontend?

Node 6 (ai): For handling token expiration, you'll want to implement automatic refresh logic...
[Tool call failed: Connection timeout]
Let me continue with the explanation without the external lookup...
[Complete explanation with code examples - 45 lines total]

... (many more nodes) ...
```

**2. Later conversation with old nodes summarized:**
```
Node 1-2: [IMPORTANT] User asked about JWT auth system. Assistant provided comprehensive overview of authentication approaches and security considerations. (47 lines)

Node 3-4: Discussed JWT refresh tokens and password hashing. Researched bcrypt vs Argon2, decided on Argon2. Included implementation examples. (62 lines)

Node 5-6: Explained frontend token expiration handling with automatic refresh flow and error recovery strategies. (45 lines)

Nodes 7-12: Implemented backend endpoints, discussed database schema, added validation middleware. (128 lines)

Recent nodes 23-26:
Node 23 (user): Should we add OAuth2 integration?

Node 24 (ai): OAuth2 would be a great addition. Let me explain how to integrate it...
[Detailed OAuth2 implementation guide]

Node 25 (user): What about multi-factor authentication?
Actually, wait - I'm getting confused about the token refresh flow we discussed earlier

Node 26 (ai): Let me get the exact details from our earlier conversation.
```

**3. Assistant uses MCP to expand old node:**
```
Node 26 (ai): Let me get the exact details from our earlier conversation.

[MCP Call: expand_node(6)]
Retrieved full Node 6:
Node Type: ai
Line Count: 45
Content: For handling token expiration, you'll want to implement automatic refresh logic. Here's the complete flow:
1. Store both access token (15min expiry) and refresh token (7 days)
2. Include access token in Authorization header for API calls
3. When you get a 401 response, automatically call /refresh endpoint
4. Update stored tokens and retry the original request
5. If refresh fails, redirect to login

Here's the JavaScript implementation:
```javascript
// Token refresh interceptor
axios.interceptors.response.use(
  response => response,
  async error => {
    if (error.response?.status === 401 && !error.config._retry) {
      error.config._retry = true;
      try {
        const newToken = await refreshToken();
        error.config.headers['Authorization'] = `Bearer ${newToken}`;
        return axios.request(error.config);
      } catch (refreshError) {
        redirectToLogin();
        return Promise.reject(refreshError);
      }
    }
    return Promise.reject(error);
  }
);
```

// Additional helper functions
[... rest of the 45-line implementation ...]

Now I can give you a complete answer about the refresh flow based on node 6...
```
## Development Phases

### Phase 1: Core Middleware
- [ ] Basic conversation manager with PydanticAI
- [ ] DuckDB storage implementation
- [ ] Simple compression (no multi-level hierarchy yet, instead of actual summaries, it just shows the first 8 words of the node)
- [ ] Basic chat interface
- [ ] Test setup with pytest

### Phase 2: MCP Integration
- [ ] MCP server for memory browsing
- [ ] read-only tool `expand(idx:int)`: just expanding single nodes
- [ ] Integration with conversation flow

### Phase 2: Hierarchical Compression
- [ ] Multi-level hierarchy system
- [ ] AI summarization of nodes
- [ ] Automatic compression triggers
- [ ] New tool `find(text:str)` (regex search across all nodes or from A node to B node)
- [ ] New tool `show_summaries(start_node:int,end_node:int)`

### Phase 4: Advanced Features
- [ ] Semantic search with embeddings
- [ ] Meta-summary generation
- [ ] Performance optimization
- [ ] Basic web UI for visualization

### Phase 5: Production Ready
- [ ] Documentation
- [ ] Deployment packages
- [ ] Example applications

## Success Metrics

1. **Conversation Length**: Support 10,000+ node conversations
2. **Token Efficiency**: 10x reduction in context size
3. **Information Fidelity**: 100% retrievability of any conversation node

