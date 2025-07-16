# Hierarchical Memory Agent: Implementation Specification v2

## Recent Updates (July 2025)

**Phase 1 Core Implementation Completed:**
- ✅ **Fixed critical storage issues**: Resolved in-memory database connection problems that prevented data persistence across operations
- ✅ **All tests passing**: 20/20 tests now pass with 54% overall coverage (91% storage coverage)
- ✅ **Robust database handling**: Fixed connection management for both in-memory and file-based databases
- ✅ **Complete data model implementation**: ConversationNode, CompressionLevel, and storage operations fully functional
- ✅ **Working compression system**: Basic word truncation and topic extraction implemented
- ✅ **PyArrow storage optimization**: Migrated from manual data conversion to PyArrow-based serialization with Pydantic field validators for cleaner, more maintainable code

**Current Status**: Phase 1 is complete! Storage layer is optimized and all core functionality is working. Ready to begin Phase 2 (MCP Integration).

**Next Priorities:**
1. **PydanticAI Integration**: Complete the conversation manager with actual LLM integration. Use claude-sonnet-4-20250514 as default.
2. **MCP Server Setup**: Implement read-only memory browsing tool (`expand_node`) using fastmcp. Use web search to look up example implementations. Make sure the server can run in streaming-http transport.
3. **CLI Testing**: Validate the command-line interface works end-to-end
4. **Integration Testing**: Create tests that verify the full conversation flow

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
    ├── find(query)
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

## Usage Example

See `chat_demo.py`

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
- [x] Configuration system with environment variables ✅
- [x] Basic package structure and dependencies ✅
- [x] Test setup with pytest ✅
- [x] DuckDB connection utilities ✅
- [x] Data models (ConversationNode, CompressionLevel, etc.) ✅
- [x] DuckDB storage schema and operations ✅
- [x] Simple compression (basic word truncation and topic extraction) ✅
- [x] PyArrow storage optimization with Pydantic field validators ✅
- [x] Basic conversation manager with PydanticAI integration
- [x] CLI interface testing and validation


### Phase 2: MCP Integration
- [x] MCP server for memory browsing
- [x] read-only tool `expand(idx:int)`: just expanding single nodes
- [x] MCP server+client setup using pydanticAI that integrates the MCP and the middleware

### Phase 3
- [x] Automatic compression triggers
- [x] New tool `find(text:str)` (regex search across all nodes or from A node to B node)

### Phase 4: Advanced Hierarchy
- [x] Multi-level hierarchy system (see diagrams.md for more details)
- [x] New tool `show_summaries(start_node:int,end_node:int)`

### Phase 5: tf-idf topics
- [ ] tf-idf topics included in the compressed nodes in the topics attribute. they should also appear in the display next to each enhanced summary display.

### Phase 6: User Command Line Interface
- [ ] Improve on the current chat_demo.py. It should at least support every MCP server function.

### Phase 7: Production Ready
- [ ] Documentation
- [ ] Deployment packages
- [ ] Example applications
- [ ] End-to-end integration tests

## Success Metrics

1. **Conversation Length**: Support 10,000+ node conversations
2. **Token Efficiency**: 10x reduction in context size
3. **Information Fidelity**: 100% retrievability of any conversation node

