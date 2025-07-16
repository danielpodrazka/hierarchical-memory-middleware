# 🧪 Testing the Hierarchical Memory Chat System

This guide explains how to test the hierarchical memory conversation system that we just fixed.

## 🚀 Quick Start

### 1. Basic Chat Test

Run the interactive chat with MCP memory browsing tools (default):

```bash
python chat_demo.py
```

This will:
- Start a conversation with ID `test-chat-session-001`
- Create a local database file `chat_demo.db` for persistence
- **Start MCP server with memory browsing tools**
- Allow you to chat interactively with tool access
- Show memory browsing capabilities

### 2. Disabling MCP Server (Basic Mode)

```bash
python chat_demo.py --no-mcp
```

This uses the conversation manager directly without MCP tools.
**Note**: By default, the chat demo runs with MCP server integration enabled, giving you access to memory browsing tools.

## 💬 Chat Features

### Basic Chat
Just type your messages and the AI will respond using the hierarchical memory system.

### Special Commands
- `/search <query>` - Search through conversation history
- `/summary` - Show conversation statistics
- `/recent` - Show recent messages
- `/help` - Show available commands
- `/quit` or `/exit` - Exit the chat

## 🔍 What to Test

### 1. Basic Conversation
```
👤 You: Hello! Can you help me with Python programming?
🤖 Assistant: Of course! I'd be happy to help you with Python programming...

👤 You: What are list comprehensions?
🤖 Assistant: List comprehensions are a concise way to create lists...
```

### 2. Memory Search
```
👤 You: /search Python
🔍 Searching for: 'Python'
   Found 2 results:
   1. 👤 (score: 0.80) Hello! Can you help me with Python programming?
   2. 🤖 (score: 0.80) Of course! I'd be happy to help you with Python...
```

### 3. Conversation Resumption
1. Chat for a few messages
2. Exit with `/quit`
3. Run `python chat_demo.py` again
4. Notice it says "✅ Resumed existing conversation"
5. Use `/recent` to see your previous messages

### 4. Memory Persistence
The conversation is saved to `chat_demo.db`, so you can:
- Stop and restart the script
- Your conversation history will be preserved
- Search through past messages
- See conversation statistics

## 🧩 System Architecture Being Tested

This test script validates:

### ✅ Fixed Issues
1. **Node-based storage** - User and AI messages are stored as separate nodes
2. **Conversation resumption** - Conversations can be stopped and resumed
3. **Memory browsing** - Search through conversation history
4. **Hierarchical compression** - Memory management for long conversations

### 🔧 Technical Components
- `save_conversation_node()` - Our newly implemented method
- `HierarchicalConversationManager` - Main conversation logic
- `DuckDBStorage` - Database operations
- `MemoryMCPServer` - MCP server integration with memory browsing tools (default)

## 📊 Expected Behavior

### First Run (New Conversation)
```
🚀 Starting chat tester...
📄 Database: ./chat_demo.db
🤖 Model: claude-sonnet-4-20250514
🔗 Conversation ID: test-chat-session-001

🆕 Started new conversation

📊 Conversation Summary:
   Total nodes: 0
   Recent nodes: 0
   Compressed nodes: 0
```

### Subsequent Runs (Resumed Conversation)
```
🚀 Starting chat tester...
📄 Database: ./chat_demo.db
🤖 Model: claude-sonnet-4-20250514
🔗 Conversation ID: test-chat-session-001

✅ Resumed existing conversation

📊 Conversation Summary:
   Total nodes: 4
   Recent nodes: 4
   Compressed nodes: 0

📜 Recent messages:
   1. 👤 [2025-01-14T10:30] Hello! Can you help me with Python?
   2. 🤖 [2025-01-14T10:30] Of course! I'd be happy to help...
```

## 🐛 Troubleshooting

### Database Issues
If you get database errors:
```bash
# Remove the test database and start fresh
rm chat_demo.db
python chat_demo.py
```

### Model Configuration
The script uses `claude-sonnet-4-20250514` by default. If you need to change it:
1. Edit line 40 in `chat_demo.py`
2. Change `work_model` to your preferred model

### Memory Issues
If search results seem incorrect:
1. Use `/summary` to check conversation statistics
2. Try `/recent` to see recent messages
3. Check the database file was created: `ls -la chat_demo.db`

## 📈 Testing Scenarios

### Scenario 1: Basic Functionality
1. Start chat
2. Send 3-5 messages
3. Use `/search` with keywords from your messages
4. Use `/summary` to see statistics
5. Exit and restart - verify resumption works

### Scenario 2: Memory Search
1. Chat about different topics (Python, cooking, travel)
2. Use `/search Python` - should find Python-related messages
3. Use `/search cooking` - should find cooking-related messages
4. Use `/search random` - should show "No results found"

### Scenario 3: Long Conversation
1. Send 10+ messages to trigger potential compression
2. Check `/summary` to see if compression occurs
3. Verify search still works across compressed and uncompressed messages

## 🏆 Success Criteria

The system is working correctly if:

- ✅ **73/73 tests pass** (already verified)
- ✅ **Conversations can be started and resumed**
- ✅ **Messages are saved as separate user/AI nodes**
- ✅ **Search finds relevant messages**
- ✅ **Database persistence works across restarts**
- ✅ **Memory statistics are accurate**
- ✅ **No AttributeError about save_conversation_turn**

Enjoy testing the hierarchical memory system! 🎉
