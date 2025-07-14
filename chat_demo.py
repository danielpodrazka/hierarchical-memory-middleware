#!/usr/bin/env python3
"""
Test script for the hierarchical memory conversation system.

This script demonstrates:
1. Starting/resuming a conversation with a fixed ID
2. Interactive chatting with hierarchical memory
3. Memory browsing capabilities
4. Optional MCP server integration
"""

import asyncio
import logging
import sys
from typing import Optional

from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.middleware.conversation_manager import (
    HierarchicalConversationManager,
)
from hierarchical_memory_middleware.mcp_server.memory_server import MemoryMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fixed conversation ID for testing (so you can restart and resume)
TEST_CONVERSATION_ID = "test-chat-session-001"


class ChatTester:
    """Interactive chat tester for the hierarchical memory system."""

    def __init__(self, use_mcp_server: bool = False):
        """Initialize the chat tester."""
        # Create config (load from .env file, then override test-specific settings)
        self.config = Config.from_env()
        # Override specific settings for test environment
        self.config.db_path = "./test_chat.db"  # Local file for persistence
        self.config.recent_node_limit = 5
        self.config.summary_threshold = 20

        if use_mcp_server:
            # Use the full MCP server
            self.server = MemoryMCPServer(self.config)
            self.conversation_manager = self.server.conversation_manager
        else:
            # Use conversation manager directly
            self.conversation_manager = HierarchicalConversationManager(self.config)
            self.server = None

        self.conversation_id = None

    async def start(self):
        """Start or resume the test conversation."""
        print(f"🚀 Starting chat tester...")
        print(f"📄 Database: {self.config.db_path}")
        print(f"🤖 Model: {self.config.work_model}")
        print(f"🔗 Conversation ID: {TEST_CONVERSATION_ID}")
        print()

        # Start/resume conversation
        self.conversation_id = await self.conversation_manager.start_conversation(
            TEST_CONVERSATION_ID
        )

        if self.conversation_id == TEST_CONVERSATION_ID:
            print("✅ Resumed existing conversation")
        else:
            print("🆕 Started new conversation")

        # Show conversation summary
        await self.show_conversation_summary()
        print()

    async def show_conversation_summary(self):
        """Display conversation statistics and recent messages."""
        try:
            summary = await self.conversation_manager.get_conversation_summary()
            print("📊 Conversation Summary:")
            print(f"   Total nodes: {summary.get('total_nodes', 0)}")
            print(f"   Recent nodes: {summary.get('recent_nodes', 0)}")
            print(f"   Compressed nodes: {summary.get('compressed_nodes', 0)}")

            # Show recent messages
            if summary.get("total_nodes", 0) > 0:
                await self.show_recent_messages(limit=3)
        except Exception as e:
            print(f"❌ Error getting summary: {e}")

    async def show_recent_messages(self, limit: int = 5):
        """Show recent conversation messages."""
        try:
            results = await self.conversation_manager.search_memory("", limit=limit)
            if results:
                print(f"\n📜 Recent messages:")
                for i, result in enumerate(results[:limit], 1):
                    node_type = result["node_type"]
                    content = result["content"]
                    timestamp = result["timestamp"]

                    # Truncate long content
                    if len(content) > 100:
                        content = content[:97] + "..."

                    icon = "👤" if node_type == "user" else "🤖"
                    print(f"   {i}. {icon} [{timestamp[:16]}] {content}")
        except Exception as e:
            print(f"❌ Error getting recent messages: {e}")

    async def search_memory(self, query: str):
        """Search conversation memory."""
        try:
            print(f"🔍 Searching for: '{query}'")
            results = await self.conversation_manager.search_memory(query, limit=5)

            if not results:
                print("   No results found.")
                return

            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                node_type = result["node_type"]
                content = result["content"]
                score = result["relevance_score"]

                icon = "👤" if node_type == "user" else "🤖"
                print(f"   {i}. {icon} (score: {score:.2f}) {content}")

        except Exception as e:
            print(f"❌ Error searching: {e}")

    async def chat_loop(self):
        """Main interactive chat loop."""
        print("💬 Chat started! Type your messages below.")
        print("   Special commands:")
        print("   - /search <query>  : Search conversation memory")
        print("   - /summary         : Show conversation summary")
        print("   - /recent          : Show recent messages")
        print("   - /quit or /exit   : Exit chat")
        print("   - /help            : Show this help")
        print()

        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue

                # Send message to conversation manager
                print("🤖 Assistant: ", end="", flush=True)
                response = await self.conversation_manager.chat(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.exception("Chat error")

    async def handle_command(self, command: str):
        """Handle special chat commands."""
        parts = command.split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ["/quit", "/exit"]:
            print("👋 Goodbye!")
            sys.exit(0)
        elif cmd == "/help":
            print("📖 Available commands:")
            print("   /search <query>  - Search conversation memory")
            print("   /summary         - Show conversation summary")
            print("   /recent          - Show recent messages")
            print("   /quit, /exit     - Exit chat")
        elif cmd == "/search":
            if args:
                await self.search_memory(args)
            else:
                print("❌ Usage: /search <query>")
        elif cmd == "/summary":
            await self.show_conversation_summary()
        elif cmd == "/recent":
            await self.show_recent_messages()
        else:
            print(f"❌ Unknown command: {cmd}. Type /help for available commands.")

        print()

    async def run(self):
        """Run the complete chat test."""
        try:
            await self.start()
            await self.chat_loop()
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            logger.exception("Fatal error")
        finally:
            # Cleanup
            if hasattr(self.conversation_manager.storage, "close"):
                self.conversation_manager.storage.close()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the hierarchical memory chat system"
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Use MCP server (default: use conversation manager directly)",
    )
    parser.add_argument(
        "--start-mcp-server",
        action="store_true",
        help="Start MCP server in background (for external MCP clients)",
    )

    args = parser.parse_args()

    # Create and run chat tester
    tester = ChatTester(use_mcp_server=args.mcp)

    if args.start_mcp_server:
        print("🌐 Starting MCP server in background...")
        # TODO: This would start the MCP server for external clients
        print("   (MCP server functionality would be implemented here)")
        print()

    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
