# Hierarchical Memory Middleware - Visual Diagrams

## 1. Conversation Flow Sequence

This sequence diagram shows how a conversation works with the hierarchical memory middleware:

```mermaid
sequenceDiagram
    participant User
    participant Middleware as Hierarchical Memory<br/>Middleware
    participant Storage as DuckDB<br/>Storage
    participant LLM as PydanticAI<br/>Agent
    participant MCP as MCP Memory<br/>Tools

    User->>Middleware: "What was our token refresh strategy?"
    
    Note over Middleware,Storage: Prepare optimized context
    Middleware->>Storage: Get recent nodes (FULL level)
    Storage-->>Middleware: Last 10 nodes
    Middleware->>Storage: Get compressed summaries
    Storage-->>Middleware: Older nodes as summaries
    
    Note over Middleware: Build context with:<br/>- Recent full nodes<br/>- Compressed summaries<br/>- System prompt
    
    Middleware->>LLM: Send optimized context + user message
    
    Note over LLM: Realizes it needs more detail<br/>about older conversation
    
    LLM->>MCP: expand_node(6)
    MCP->>Storage: Get full node 6 content
    Storage-->>MCP: Full 45-line token refresh explanation
    MCP-->>LLM: Complete node 6 details
    
    LLM-->>Middleware: "Based on node 6, here's our strategy..."
    
    Note over Middleware,Storage: Store new conversation
    Middleware->>Storage: Save user node
    Middleware->>Storage: Save AI response node
    
    Note over Middleware,Storage: Check compression triggers
    Middleware->>Storage: Compress old nodes if needed
    
    Middleware-->>User: Complete answer with details from node 6
```

## 2. Hierarchical Compression System

This flowchart shows how conversation nodes move through compression levels:

```mermaid
flowchart TD
    Start([New Conversation Node]) --> Recent{Recent?<br/>< 10 nodes}
    
    Recent -->|Yes| Full["🟢 FULL LEVEL<br/>Complete content<br/>All details preserved"]
    Recent -->|No| CheckAge{Age?<br/>< 50 nodes}
    
    CheckAge -->|Yes| Summary["🟡 SUMMARY LEVEL<br/>1-2 sentence summary<br/>+ line count<br/>Key topics preserved"]
    CheckAge -->|No| CheckOlder{Age?<br/>< 200 nodes}
    
    CheckOlder -->|Yes| Meta["🟠 META LEVEL<br/>Group summaries<br/>20-40 nodes per group<br/>High-level themes"]
    CheckOlder -->|No| Archive["🔴 ARCHIVE LEVEL<br/>Very compressed<br/>Major decisions only<br/>Long-term context"]
    
    Full --> Trigger1{"Conversation<br/>grows?"}
    Summary --> Trigger2{"Conversation<br/>grows?"}
    Meta --> Trigger3{"Conversation<br/>grows?"}
    
    Trigger1 -->|Yes| Summary
    Trigger2 -->|Yes| Meta
    Trigger3 -->|Yes| Archive
    
    Full --> MCP1["🔍 expand_node()<br/>Returns full content"]
    Summary --> MCP2["🔍 expand_node()<br/>Returns full content"]
    Meta --> MCP3["🔍 expand_node()<br/>Returns group details"]
    Archive --> MCP4["🔍 expand_node()<br/>Returns archived content"]
```

## 3. System Architecture Overview

This diagram shows the complete system architecture:

```mermaid
graph TB
    subgraph "User Interface"
        User[👤 User]
    end
    
    subgraph "Hierarchical Memory Middleware"
        CM["🧠 Conversation Manager<br/>• Context optimization<br/>• Compression triggers<br/>• Response orchestration"]
        HB["📊 Hierarchy Builder<br/>• Smart summarization<br/>• Importance scoring<br/>• Level management"]
    end
    
    subgraph "AI Agents (PydanticAI)"
        Work["🤖 Work Agent<br/>(i.e. Claude Sonnet)<br/>Main conversations"]
        Sum["📝 Summary Agent<br/>Compression tasks"]
    end
    
    subgraph "Storage Layer"
        DB["🗄️ DuckDB<br/>• Conversation nodes<br/>• Compression levels<br/>• Embeddings<br/>• Metadata"]
    end
    
    subgraph "MCP Memory Tools"
        MCP["🔧 Memory Tools<br/>• expand_node(id)<br/>• search_memory(query)<br/>• browse_hierarchy(level)<br/>• get_stats()"]
    end
    
    User -.->|"Chat message"| CM
    CM -->|"Prepare context"| DB
    CM -->|"Generate response"| Work
    Work <-->|"MCP calls during response"| MCP
    MCP <--> DB
    CM -->|"Compress old nodes"| Sum
    Sum --> HB
    HB --> DB
    CM -.->|"Response"| User
```

## 4. Memory Compression in Action

This shows how a long conversation gets compressed over time:

```mermaid
timeline
    title Conversation Memory Evolution

    section Early Stage (10 Nodes)
        Node 1-10 : Full Content
               : 🟢 All details preserved
               : Complete tool calls
               : Full code examples

    section Growing Stage (20 Nodes)
        Node 1-10 : Compressed to Summaries
               : 🟡 "JWT auth system design (47 lines)"
               : 🟡 "Token refresh implementation (62 lines)"
        Node 11-20 : Full Content
                 : 🟢 Recent interactions
                 : 🟢 Current context

    section Mature Stage (50 Nodes)
        Node 1-20 : Meta Summaries
               : 🟠 "Authentication system design & implementation"
               : 🟠 "Database schema and validation (20 nodes, 180 lines)"
        Node 21-40 : Summary Level
                 : 🟡 OAuth2 integration discussion
                 : 🟡 MFA implementation details
        Node 41-50 : Full Content
                 : 🟢 Current working context

    section Extended Stage (200 Nodes)
        Node 1-50 : Archive Level
               : 🔴 "Built complete auth system with JWT, OAuth2, MFA"
        Node 51-150 : Meta Summaries
                  : 🟠 Feature additions and optimizations
                  : 🟠 Testing and deployment discussions
        Node 151-190 : Summary Level
                   : 🟡 Performance tuning conversations
                   : 🟡 Bug fixes and improvements
        Node 191-200 : Full Content
                   : 🟢 Current discussion

    section Long-term Stage (1000+ Nodes)
        Node 1-200 : Archive Level
               : 🔴 "Complete project lifecycle: auth → features → deployment"
        Node 201-800 : Meta Summaries
                  : 🟠 Major feature development phases
                  : 🟠 Architecture decisions and refactoring
        Node 801-990 : Summary Level
                   : 🟡 Recent feature discussions
                   : 🟡 Code reviews and optimizations
        Node 991-1000 : Full Content
                    : 🟢 Active conversation
```

## 5. Token Efficiency Visualization

This shows the dramatic token savings achieved through hierarchical compression:

```mermaid
pie title Token Usage Distribution
    "Recent Nodes (Full)" : 8000
    "Summary Level" : 1200
    "Meta Level" : 400
    "Archive Level" : 200
    "System Prompts" : 1200
```

```mermaid
xychart-beta
    title "Token Usage: Traditional vs Hierarchical"
    x-axis ["10 Nodes", "50 Nodes", "100 Nodes", "500 Nodes", "1000 Nodes"]
    y-axis "Tokens Used" 0 --> 150000
    line [8000, 40000, 80000, 400000, 800000]
    line [8000, 12000, 15000, 25000, 35000]
```

---

*These diagrams illustrate how the Hierarchical Memory Middleware enables infinite AI conversations while maintaining optimal performance and perfect recall through intelligent compression and MCP-based expansion.*
