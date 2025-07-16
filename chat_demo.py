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
import json
import logging
import os
import sys

from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.middleware.conversation_manager import (
    HierarchicalConversationManager,
)
from hierarchical_memory_middleware.models import CompressionLevel, NodeType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fixed conversation ID for testing (so you can restart and resume)
TEST_CONVERSATION_ID = "3d1fe944-ea9d-4b48-a311-0852171f953c"
TEST_CONVERSATION_ID = "582019a5-3fb5-4fe1-aefb-0ca09c33a726"


class ChatTester:
    """Interactive chat tester for the hierarchical memory system."""

    def __init__(self, use_mcp_server: bool = True):
        """Initialize the chat tester."""
        # Create config (load from .env file, then override test-specific settings)
        self.config = Config.from_env()
        # Override specific settings for test environment
        self.config.recent_node_limit = 5
        self.config.summary_threshold = 20

        if use_mcp_server:
            # Connect to external MCP server
            mcp_server_url = f"http://127.0.0.1:{self.config.mcp_port}/mcp"
            self.conversation_manager = HierarchicalConversationManager(
                self.config, mcp_server_url=mcp_server_url
            )
            self.server_url = mcp_server_url
            self.needs_external_server = True
        else:
            # Use conversation manager directly without tools
            self.conversation_manager = HierarchicalConversationManager(self.config)
            self.needs_external_server = False

        self.conversation_id = None

        # Will be set up in start() method once we have conversation_id
        self.conversation_json_path = None

    async def start(self):
        """Start or resume the test conversation."""
        print(f"🚀 Starting chat tester...")
        print(f"📄 Database: {self.config.db_path}")
        print(f"🤖 Model: {self.config.work_model}")
        print(f"🔗 Conversation ID: {TEST_CONVERSATION_ID}")
        print()

        # Check for external MCP server if using MCP mode
        if self.needs_external_server:
            print(f"📡 Connecting to MCP server at {self.server_url}...")
            # The connection will be tested when we first try to use tools
            print(f"✅ Will use MCP tools from {self.server_url}")
            print(f"⚠️  Note: Make sure the MCP server is running with:")
            print(f"   python hierarchical_memory_middleware/mcp_server/run_server.py")
            print()
        # Start/resume conversation
        self.conversation_id = await self.conversation_manager.start_conversation(
            TEST_CONVERSATION_ID
        )

        # Set up conversation JSON file for real-time viewing
        conversations_dir = ".conversations"
        os.makedirs(conversations_dir, exist_ok=True)
        self.conversation_json_path = os.path.join(
            conversations_dir, f"{self.conversation_id}.json"
        )
        self.conversation_ai_view_json_path = os.path.join(
            conversations_dir, f"{self.conversation_id}_ai_view.json"
        )
        print(f"📄 Real-time conversation JSON: {self.conversation_json_path}")
        print(f"🤖 AI view conversation JSON: {self.conversation_ai_view_json_path}")

        if self.conversation_id == TEST_CONVERSATION_ID:
            print("✅ Resumed existing conversation")
        else:
            print("🆕 Started new conversation")

        # Show conversation summary
        await self.show_conversation_summary()

        # Save initial conversation state to JSON
        await self.save_conversation_to_json()
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
            results = await self.conversation_manager.find("", limit=limit)
            if results:
                print(f"\n📜 Recent messages:")
                for i, result in enumerate(results[:limit], 1):
                    node_type = result["node_type"]
                    # Use summary if available (for compressed nodes), otherwise use content
                    display_text = result.get("summary") or result["content"]
                    timestamp = result["timestamp"]
                    node_id = result.get("node_id", "unknown")

                    # Truncate long content
                    if len(display_text) > 500:
                        display_text = display_text[:500] + "..."

                    icon = "👤" if node_type == "user" else "🤖"
                    print(
                        f"   {i}. {icon} [Node {node_id}] [{timestamp[:16]}] {display_text}"
                    )
        except Exception as e:
            print(f"❌ Error getting recent messages: {e}")

    async def find(self, query: str):
        """Search conversation memory."""
        try:
            print(f"🔍 Searching for: '{query}'")
            results = await self.conversation_manager.find(query, limit=5)

            if not results:
                print("   No results found.")
                return

            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                node_type = result["node_type"]
                # Use summary if available (for compressed nodes), otherwise use content
                display_text = result.get("summary") or result["content"]
                score = result["relevance_score"]
                node_id = result.get("node_id", "unknown")

                icon = "👤" if node_type == "user" else "🤖"
                print(
                    f"   {i}. {icon} [Node {node_id}] (score: {score:.2f}) {display_text}"
                )

        except Exception as e:
            print(f"❌ Error searching: {e}")

    async def show_hierarchical_summaries(self, start_node: int, end_node: int):
        """Show hierarchical summaries for a range of nodes using the new Phase 4 functionality."""
        try:
            print(
                f"📊 Phase 4: Hierarchical Summaries for nodes {start_node}-{end_node}"
            )
            print("    (This demonstrates the new advanced hierarchy system)")
            print()

            # For now, we'll call the storage method directly since MCP tools require server setup
            # In a full MCP setup, this would use the show_summaries tool
            nodes = await self.conversation_manager.storage.get_nodes_in_range(
                conversation_id=self.conversation_id,
                start_node_id=start_node,
                end_node_id=end_node,
            )

            if not nodes:
                print(f"   ❌ No nodes found in range {start_node}-{end_node}")
                return

            # Group nodes by compression level
            nodes_by_level = {"FULL": [], "SUMMARY": [], "META": [], "ARCHIVE": []}

            for node in nodes:
                level_name = node.level.name
                nodes_by_level[level_name].append(node)

            # Display hierarchy
            total_nodes = len(nodes)
            total_lines = sum(node.line_count or 0 for node in nodes)

            print(
                f"   📈 Found {total_nodes} nodes ({total_lines} total lines) in range {start_node}-{end_node}"
            )
            print()

            # Show compression distribution
            compression_stats = {
                level: len(nodes) for level, nodes in nodes_by_level.items() if nodes
            }

            if compression_stats:
                print("   🗜️  Compression Distribution:")
                for level, count in compression_stats.items():
                    print(f"      {level}: {count} nodes")
                print()

            # Show each level in detail
            level_order = ["FULL", "SUMMARY", "META", "ARCHIVE"]
            level_descriptions = {
                "FULL": "🟢 Recent nodes with complete content",
                "SUMMARY": "🟡 Older nodes with 1-2 sentence summaries",
                "META": "🟠 Groups of summary nodes (20-40 nodes each)",
                "ARCHIVE": "🔴 Very compressed high-level context",
            }

            for level in level_order:
                if level in nodes_by_level and nodes_by_level[level]:
                    level_nodes = nodes_by_level[level]
                    print(f"   {level_descriptions[level]} ({len(level_nodes)} nodes):")

                    for i, node in enumerate(level_nodes[:5], 1):  # Show first 5 nodes
                        node_type_icon = (
                            "👤" if node.node_type.value == "user" else "🤖"
                        )

                        # Determine what content to show
                        if node.summary:
                            display_content = node.summary
                            content_type = "[Summary]"
                        else:
                            display_content = node.content
                            content_type = "[Full]"

                        # Truncate long content
                        if len(display_content) > 150:
                            display_content = display_content[:150] + "..."

                        print(
                            f"      {i}. {node_type_icon} Node {node.node_id} {content_type}: {display_content}"
                        )

                    if len(level_nodes) > 5:
                        print(f"      ... and {len(level_nodes) - 5} more nodes")
                    print()

            # Show META group information if any META nodes exist
            meta_nodes = nodes_by_level.get("META", [])
            if meta_nodes:
                print("   🔗 META Group Details:")
                for node in meta_nodes:
                    if node.summary_metadata:
                        try:
                            metadata = json.loads(node.summary_metadata)
                            meta_info = metadata.get("meta_group_info", {})
                            if meta_info:
                                print(
                                    f"      📦 Node {node.node_id}: Groups nodes {meta_info.get('start_node_id')}-{meta_info.get('end_node_id')}"
                                )
                                print(
                                    f"         Topics: {', '.join(meta_info.get('main_topics', [])[:3])}"
                                )
                                print(
                                    f"         Contains: {meta_info.get('node_count')} nodes, {meta_info.get('total_lines')} lines"
                                )
                        except (json.JSONDecodeError, KeyError):
                            print(
                                f"      📦 Node {node.node_id}: META group (details unavailable)"
                            )
                print()

            print(
                "   ✨ This demonstrates Phase 4: Advanced Hierarchy with 4-level compression!"
            )
            print("      FULL → SUMMARY → META → ARCHIVE")

        except Exception as e:
            print(f"   ❌ Error showing hierarchical summaries: {e}")
            import traceback

            traceback.print_exc()

    async def expand_node(self, node_id: int):
        """Expand a node to show its full content."""
        try:
            print(f"🔍 Expanding node {node_id}...")
            result = await self.conversation_manager.get_node_details(
                node_id, self.conversation_id
            )

            if result is None:
                print(
                    f"   ❌ Node {node_id} not found in conversation {self.conversation_id}"
                )
                return

            print(f"   ✅ Node {node_id} details:")
            print(f"      Type: {result.get('node_type', 'unknown')}")
            print(f"      Timestamp: {result.get('timestamp', 'unknown')}")
            print(f"      Sequence: {result.get('sequence_number', 'unknown')}")
            print(f"      Compression Level: {result.get('level', 'unknown')}")
            print(f"      Line Count: {result.get('line_count', 'unknown')}")
            print(f"      Tokens Used: {result.get('tokens_used', 'unknown')}")

            # Show topics if available
            topics = result.get("topics", [])
            if topics:
                print(f"      Topics: {', '.join(topics)}")

            # Show summary if available
            summary = result.get("summary", "")
            if summary:
                print(f"      Summary: {summary}")

            # Show content
            content = result.get("content", "")
            print(f"      Content:")
            # Split content into lines and indent
            for line in content.split("\n"):
                print(f"        {line}")

        except Exception as e:
            print(f"   ❌ Error expanding node {node_id}: {e}")

    async def save_conversation_to_json(self):
        """Save the current conversation to JSON files for real-time viewing.

        Saves two versions:
        1. Full conversation with all nodes
        2. AI agent view with only what the AI actually sees
        """
        try:
            # Get all conversation nodes for full view
            all_nodes = await self.conversation_manager.storage.get_conversation_nodes(
                self.conversation_id
            )

            # 1. Save full conversation data
            full_conversation_data = {
                "conversation_id": self.conversation_id,
                "total_nodes": len(all_nodes),
                "last_updated": all_nodes[-1].timestamp.isoformat()
                if all_nodes
                else None,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "node_type": node.node_type.value,
                        "content": node.content,
                        "summary": node.summary,
                        "timestamp": node.timestamp.isoformat(),
                        "sequence_number": node.sequence_number,
                        "line_count": node.line_count,
                        "compression_level": node.level.value,
                        "tokens_used": node.tokens_used,
                        "topics": node.topics
                        if node.topics and isinstance(node.topics, list)
                        else [],
                        "ai_components": node.ai_components,
                        "relates_to_node_id": node.relates_to_node_id,
                    }
                    for node in all_nodes
                ],
            }

            # Write full conversation to file
            with open(self.conversation_json_path, "w", encoding="utf-8") as f:
                json.dump(full_conversation_data, f, indent=2, ensure_ascii=False)

            # 2. Get the actual AI view data from the conversation manager
            ai_view_raw = self.conversation_manager.get_last_ai_view_data()

            if ai_view_raw:
                # We have AI view data from the last message processing
                ai_view_data = {
                    "conversation_id": self.conversation_id,
                    "description": "This shows exactly what the AI agent saw in the last message processing",
                    "last_updated": all_nodes[-1].timestamp.isoformat()
                    if all_nodes
                    else None,
                    "compressed_nodes_count": len(
                        ai_view_raw.get("compressed_nodes", [])
                    ),
                    "recent_nodes_count": len(ai_view_raw.get("recent_nodes", [])),
                    "recent_messages_from_input_count": len(
                        ai_view_raw.get("recent_messages_from_input", [])
                    ),
                    "total_messages_sent_to_ai": ai_view_raw.get(
                        "total_messages_sent_to_ai", 0
                    ),
                    "compressed_nodes": ai_view_raw.get("compressed_nodes", []),
                    "recent_nodes": ai_view_raw.get("recent_nodes", []),
                    "recent_messages_from_input": ai_view_raw.get(
                        "recent_messages_from_input", []
                    ),
                }
            else:
                # No AI view data available yet (e.g., no messages processed)
                ai_view_data = {
                    "conversation_id": self.conversation_id,
                    "description": "No AI view data available yet - send a message to see what the AI sees",
                    "last_updated": all_nodes[-1].timestamp.isoformat()
                    if all_nodes
                    else None,
                    "note": "The AI view is captured during message processing. It will be populated after you send a message.",
                }

            # Write AI view to file
            with open(self.conversation_ai_view_json_path, "w", encoding="utf-8") as f:
                json.dump(ai_view_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            # Don't let JSON export errors break the chat
            logger.debug(f"Error saving conversation to JSON: {e}")

    async def chat_loop(self):
        """Main interactive chat loop."""
        print("💬 Chat started! Type your messages below.")
        print("   Special commands:")
        print("   - /search <query>  : Search conversation memory")
        print("   - /expand <node_id>: Expand a node to see full content")
        print("   - /summary         : Show conversation summary")
        print("   - /recent          : Show recent messages")
        print(
            "   - /summaries <start> <end> : Show hierarchical summaries for node range (Phase 4)"
        )
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

                # Save conversation to JSON for real-time viewing
                await self.save_conversation_to_json()

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
            print("📚 Available commands:")
            print("   /search <query>  - Search conversation memory")
            print("   /expand <node_id>- Expand a node to see full content")
            print("   /summary         - Show conversation summary")
            print("   /recent          - Show recent messages")
            print(
                "   /summaries <start> <end> - Show hierarchical summaries for node range (Phase 4)"
            )
            print("   /quit, /exit     - Exit chat")
        elif cmd == "/search":
            if args:
                await self.find(args)
            else:
                print("❌ Usage: /search <query>")
        elif cmd == "/expand":
            if args:
                try:
                    node_id = int(args)
                    await self.expand_node(node_id)
                except ValueError:
                    print("❌ Invalid node_id. Please provide a number.")
            else:
                print("❌ Usage: /expand <node_id>")
        elif cmd == "/summary":
            await self.show_conversation_summary()
        elif cmd == "/recent":
            await self.show_recent_messages()
        elif cmd == "/summaries":
            if args:
                try:
                    parts = args.split()
                    if len(parts) == 2:
                        start_node = int(parts[0])
                        end_node = int(parts[1])
                        await self.show_hierarchical_summaries(start_node, end_node)
                    else:
                        print("❌ Usage: /summaries <start_node> <end_node>")
                except ValueError:
                    print("❌ Invalid node IDs. Please provide numbers.")
            else:
                print("❌ Usage: /summaries <start_node> <end_node>")
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
        "--no-mcp",
        action="store_true",
        help="Disable MCP server (default: use MCP server with memory tools)",
    )
    parser.add_argument(
        "--start-mcp-server",
        action="store_true",
        help="Start MCP server in background (for external MCP clients)",
    )

    args = parser.parse_args()

    # Create and run chat tester
    tester = ChatTester(use_mcp_server=not args.no_mcp)

    if args.start_mcp_server:
        print("🌐 Starting MCP server in background...")
        # TODO: This would start the MCP server for external clients
        print("   (MCP server functionality would be implemented here)")
        print()

    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
