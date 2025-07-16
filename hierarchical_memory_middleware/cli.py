"""Consolidated CLI interface for hierarchical memory middleware with MCP integration."""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.status import Status

from .config import Config
from .middleware.conversation_manager import HierarchicalConversationManager
from .models import CompressionLevel, NodeType

# Logging will be configured by the CLI callback (setup_cli_logging)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Hierarchical Memory Middleware CLI - Advanced conversational AI with hierarchical memory",
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def setup_cli_logging(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging for all commands",
    ),
):
    """Setup logging for all CLI commands."""
    config = Config.from_env()
    config.setup_logging()
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


def check_mcp_server_running(mcp_url: str) -> bool:
    """Check if MCP server is running by sending an initialization request."""
    try:
        import requests
        import json

        # Add proper Accept header for MCP Streamable HTTP transport
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

        # Send MCP initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": "test-init",
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        }

        response = requests.post(mcp_url, headers=headers, json=init_request, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


async def resolve_conversation_id(partial_id: str, db_path: str) -> str:
    """Resolve a partial conversation ID to a full ID, Docker-style."""
    try:
        from .storage import DuckDBStorage

        storage = DuckDBStorage(db_path)
        conversations = await storage.get_conversation_list()

        # Find conversations that start with the partial ID
        matches = [conv for conv in conversations if conv["id"].startswith(partial_id)]

        if len(matches) == 0:
            raise ValueError(
                f"No conversation found with ID starting with '{partial_id}'"
            )
        elif len(matches) == 1:
            return matches[0]["id"]
        else:
            # Multiple matches - show ambiguous error
            match_ids = [conv["id"][:12] + "..." for conv in matches]
            raise ValueError(
                f"Ambiguous conversation ID '{partial_id}'. Matches: {', '.join(match_ids)}"
            )
    except Exception as e:
        raise ValueError(f"Error resolving conversation ID: {e}")


async def resolve_conversation_name(name: str, db_path: str) -> str:
    """Resolve a conversation name to its full ID."""
    try:
        from .storage import DuckDBStorage

        storage = DuckDBStorage(db_path)
        conversations = await storage.get_conversation_list()

        # Find conversations that match the name
        matches = [conv for conv in conversations if conv.get("name") == name]

        if len(matches) == 0:
            raise ValueError(f"No conversation found with name '{name}'")
        elif len(matches) == 1:
            return matches[0]["id"]
        else:
            # Multiple matches - should not happen due to unique constraints
            raise ValueError(f"Multiple conversations found with name '{name}'")
    except Exception as e:
        raise ValueError(f"Error resolving conversation name: {e}")


async def resolve_conversation_identifier(identifier: str, db_path: str) -> str:
    """Resolve a conversation identifier (ID, partial ID, or name) to full ID."""
    # First try as partial ID
    try:
        return await resolve_conversation_id(identifier, db_path)
    except ValueError:
        pass

    # Then try as name
    try:
        return await resolve_conversation_name(identifier, db_path)
    except ValueError:
        pass

    # If neither works, raise an error
    raise ValueError(f"No conversation found with identifier '{identifier}'")


async def save_conversation_to_json(manager, conversation_id: str, export_dir: str):
    """Save conversation to JSON files for real-time viewing."""
    try:
        # Get all conversation nodes
        all_nodes = await manager.storage.get_conversation_nodes(conversation_id)

        # Save full conversation
        conversation_json_path = os.path.join(export_dir, f"{conversation_id}.json")
        full_data = {
            "conversation_id": conversation_id,
            "total_nodes": len(all_nodes),
            "last_updated": all_nodes[-1].timestamp.isoformat() if all_nodes else None,
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
                    "relates_to_node_id": node.relates_to_node_id,
                }
                for node in all_nodes
            ],
        }

        with open(conversation_json_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)

        # Save AI view
        ai_view_json_path = os.path.join(export_dir, f"{conversation_id}_ai_view.json")
        ai_view_raw = manager.get_last_ai_view_data()

        if ai_view_raw:
            ai_view_data = {
                "conversation_id": conversation_id,
                "description": "This shows exactly what the AI agent saw in the last message processing",
                "last_updated": all_nodes[-1].timestamp.isoformat()
                if all_nodes
                else None,
                "compressed_nodes_count": len(ai_view_raw.get("compressed_nodes", [])),
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
            ai_view_data = {
                "conversation_id": conversation_id,
                "description": "No AI view data available yet - send a message to see what the AI sees",
                "last_updated": all_nodes[-1].timestamp.isoformat()
                if all_nodes
                else None,
                "note": "The AI view is captured during message processing.",
            }

        with open(ai_view_json_path, "w", encoding="utf-8") as f:
            json.dump(ai_view_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.debug(f"Error saving conversation to JSON: {e}")


@app.command()
def chat(
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", "-c", help="Resume existing conversation"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Create or resume conversation by name"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
    recent_limit: Optional[int] = typer.Option(
        None, "--recent-limit", "-r", help="Recent nodes limit"
    ),
    mcp_port: Optional[int] = typer.Option(
        None, "--mcp-port", help="MCP server port (default from config)"
    ),
    export_dir: str = typer.Option(
        ".conversations",
        "--export-dir",
        help="Directory for real-time conversation exports",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Enable streaming responses (default: True)",
    ),
):
    """Start an interactive chat session with MCP memory tools."""
    asyncio.run(
        _chat_session(
            conversation_id, name, model, db_path, recent_limit, mcp_port, export_dir, stream
        )
    )


async def _chat_session(
    conversation_id: Optional[str],
    name: Optional[str],
    model: Optional[str],
    db_path: Optional[str],
    recent_limit: Optional[int],
    mcp_port: Optional[int],
    export_dir: str,
    stream: bool,
):
    """Run the interactive chat session with MCP integration."""
    # Load configuration
    config = Config.from_env()

    # Override with CLI arguments
    if model:
        config.work_model = model
    if db_path:
        config.db_path = db_path
    if recent_limit:
        config.recent_node_limit = recent_limit
    if mcp_port:
        config.mcp_port = mcp_port

    # Configure test-specific settings
    config.recent_node_limit = config.recent_node_limit or 5
    config.summary_threshold = config.summary_threshold or 20

    console.print("[bold green]🚀 Hierarchical Memory Middleware[/bold green]")
    console.print(f"📄 Database: {config.db_path}")
    console.print(f"🤖 Model: {config.work_model}")
    console.print(f"🔗 Recent nodes limit: {config.recent_node_limit}")
    console.print(f"📈 Summary threshold: {config.summary_threshold}")
    console.print(f"⚡ Streaming: {'Enabled' if stream else 'Disabled'}")

    # Setup MCP server (required)
    mcp_server_url = f"http://127.0.0.1:{config.mcp_port}/mcp"
    console.print(f"📡 MCP server: {mcp_server_url}")

    # Check if MCP server is running
    if not check_mcp_server_running(mcp_server_url):
        console.print("[red]❌ MCP server is not running![/red]")
        console.print("[yellow]Please start the MCP server first:[/yellow]")
        console.print(
            f"[yellow]   python -m hierarchical_memory_middleware.mcp_server.run_server[/yellow]"
        )
        sys.exit(1)

    console.print("[green]✅ MCP server is running[/green]")
    console.print()

    # Setup conversation export directory
    os.makedirs(export_dir, exist_ok=True)

    try:
        # Initialize conversation manager with MCP
        manager = HierarchicalConversationManager(config, mcp_server_url=mcp_server_url)

        # Resolve conversation identifier
        resolved_conversation_id = None
        if conversation_id and name:
            console.print(
                "[red]❌ Cannot specify both --conversation-id and --name[/red]"
            )
            sys.exit(1)
        elif conversation_id:
            # Use conversation ID (supports partial matching)
            try:
                resolved_conversation_id = await resolve_conversation_id(
                    conversation_id, config.db_path
                )
                console.print(
                    f"[dim]Resolved conversation ID: {resolved_conversation_id}[/dim]"
                )
            except ValueError as e:
                console.print(f"[red]❌ {e}[/red]")
                sys.exit(1)
        elif name:
            # Use conversation name
            try:
                resolved_conversation_id = await resolve_conversation_name(
                    name, config.db_path
                )
                console.print(
                    f"[dim]Resolved conversation '{name}' to ID: {resolved_conversation_id}[/dim]"
                )
            except ValueError:
                # Name not found, will create new conversation with this name
                console.print(
                    f"[yellow]Creating new conversation with name '{name}'[/yellow]"
                )
                resolved_conversation_id = None

        # Start or resume conversation
        conv_id = await manager.start_conversation(resolved_conversation_id)

        # Set conversation name if specified and it's a new conversation
        if name and resolved_conversation_id is None:
            await manager.set_conversation_name(conv_id, name)
            console.print(f"[green]✅ Named conversation '{name}' created[/green]")

        console.print(f"[blue]🔗 Conversation ID: {conv_id}[/blue]")
        if name:
            console.print(f"[blue]📝 Conversation Name: {name}[/blue]")
        console.print()

        # Setup conversation JSON export paths
        conversation_json_path = os.path.join(export_dir, f"{conv_id}.json")
        conversation_ai_view_json_path = os.path.join(
            export_dir, f"{conv_id}_ai_view.json"
        )
        console.print(
            f"[dim]📄 Real-time conversation JSON: {conversation_json_path}[/dim]"
        )
        console.print(f"[dim]🤖 AI view JSON: {conversation_ai_view_json_path}[/dim]")
        console.print()

        # Show conversation summary if resuming
        if conversation_id == conv_id:  # Successfully resumed
            await show_conversation_summary(manager)

        # Save initial conversation state
        await save_conversation_to_json(manager, conv_id, export_dir)

        # Show help
        console.print("[bold cyan]💬 Chat Commands:[/bold cyan]")
        console.print(
            "   [yellow]/search <query>[/yellow]      - Search conversation memory"
        )
        console.print(
            "   [yellow]/expand <node_id>[/yellow]    - Expand a node to see full content"
        )
        console.print(
            "   [yellow]/summary[/yellow]             - Show conversation summary"
        )
        console.print("   [yellow]/recent[/yellow]              - Show recent messages")
        console.print(
            "   [yellow]/summaries <start> <end>[/yellow] - Show hierarchical summaries (Phase 4)"
        )
        console.print(
            "   [yellow]/stats[/yellow]               - Show detailed statistics"
        )
        console.print("   [yellow]/help[/yellow]                - Show this help")
        console.print(
            "   [yellow]/quit[/yellow] or [yellow]/exit[/yellow]        - Exit chat"
        )
        console.print()

        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = typer.prompt("👤 You", type=str).strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    await handle_chat_command(user_input, manager, conv_id, export_dir)
                    continue

                # Generate AI response
                if stream:
                    # Streaming mode
                    console.print("[bold green]🤖 Assistant:[/bold green]")
                    console.print("┌─ [green]Response[/green] ──────────────────────────────────────────────────────────────────┐")

                    # Stream the response
                    response_chunks = []
                    async for chunk in manager.chat_stream(user_input):
                        console.print(chunk, end="", style="white")
                        response_chunks.append(chunk)

                    # Add bottom border
                    console.print()
                    console.print("└────────────────────────────────────────────────────────────────────────────────┘")
                    console.print()
                    response = ''.join(response_chunks)
                else:
                    # Non-streaming mode
                    console.print("[bold green]🤖 Thinking...[/bold green]")
                    response = await manager.chat(user_input)

                    # Display response in panel
                    console.print(
                        Panel(
                            Text(response, style="white"),
                            title="🤖 Assistant",
                            border_style="green",
                        )
                    )
                    console.print()

                # Save conversation state
                await save_conversation_to_json(manager, conv_id, export_dir)

            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Chat interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {str(e)}[/red]")
                logger.exception("Chat error")
                continue

    except Exception as e:
        console.print(f"[red]❌ Failed to initialize: {str(e)}[/red]")
        logger.exception("Initialization error")
        sys.exit(1)

    console.print("[blue]👋 Goodbye![/blue]")


async def show_conversation_summary(manager):
    """Show conversation summary with rich formatting."""
    try:
        summary = await manager.get_conversation_summary()
        if "error" not in summary:
            console.print(
                Panel(
                    f"📊 Total nodes: {summary.get('total_nodes', 0)}\n"
                    f"🔄 Recent nodes: {summary.get('recent_nodes', 0)}\n"
                    f"🗜️ Compressed nodes: {summary.get('compressed_nodes', 0)}",
                    title="📋 Conversation Summary",
                    border_style="blue",
                )
            )
        else:
            console.print(f"[red]❌ Error getting summary: {summary['error']}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Error getting summary: {e}[/red]")


async def handle_chat_command(command: str, manager, conv_id: str, export_dir: str):
    """Handle special chat commands."""
    parts = command.split(" ", 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ["/quit", "/exit"]:
        console.print("[blue]👋 Goodbye![/blue]")
        sys.exit(0)

    elif cmd == "/help":
        console.print("[bold cyan]📚 Available Commands:[/bold cyan]")
        console.print(
            "   [yellow]/search <query>[/yellow]      - Search conversation memory"
        )
        console.print(
            "   [yellow]/expand <node_id>[/yellow]    - Expand a node to see full content"
        )
        console.print(
            "   [yellow]/summary[/yellow]             - Show conversation summary"
        )
        console.print("   [yellow]/recent[/yellow]              - Show recent messages")
        console.print(
            "   [yellow]/summaries <start> <end>[/yellow] - Show hierarchical summaries (Phase 4)"
        )
        console.print(
            "   [yellow]/stats[/yellow]               - Show detailed statistics"
        )
        console.print(
            "   [yellow]/rename <new_name>[/yellow]        - Rename current conversation"
        )
        console.print(
            "   [yellow]/quit[/yellow], [yellow]/exit[/yellow]         - Exit chat"
        )

    elif cmd == "/search":
        if args:
            await search_memory(manager, args)
        else:
            console.print("[red]❌ Usage: /search <query>[/red]")

    elif cmd == "/expand":
        if args:
            try:
                node_id = int(args)
                await expand_node(manager, node_id, conv_id)
            except ValueError:
                console.print("[red]❌ Invalid node_id. Please provide a number.[/red]")
        else:
            console.print("[red]❌ Usage: /expand <node_id>[/red]")

    elif cmd == "/summary":
        await show_conversation_summary(manager)

    elif cmd == "/recent":
        await show_recent_messages(manager)

    elif cmd == "/summaries":
        if args:
            try:
                parts = args.split()
                if len(parts) == 2:
                    start_node = int(parts[0])
                    end_node = int(parts[1])
                    await show_hierarchical_summaries(
                        manager, conv_id, start_node, end_node
                    )
                else:
                    console.print(
                        "[red]❌ Usage: /summaries <start_node> <end_node>[/red]"
                    )
            except ValueError:
                console.print("[red]❌ Invalid node IDs. Please provide numbers.[/red]")
        else:
            console.print("[red]❌ Usage: /summaries <start_node> <end_node>[/red]")

    elif cmd == "/stats":
        await show_detailed_stats(manager)

    elif cmd == "/rename":
        if args:
            try:
                success = await manager.set_conversation_name(conv_id, args.strip())
                if success:
                    console.print(
                        f"[green]✅ Conversation renamed to '{args.strip()}'[/green]"
                    )
                else:
                    console.print(
                        f"[red]❌ Failed to rename conversation. Name '{args.strip()}' may already exist.[/red]"
                    )
            except Exception as e:
                console.print(f"[red]❌ Error renaming conversation: {e}[/red]")
        else:
            console.print("[red]❌ Usage: /rename <n>[/red]")

    else:
        console.print(
            f"[red]❌ Unknown command: {cmd}. Type /help for available commands.[/red]"
        )

    console.print()


async def search_memory(manager, query: str):
    """Search conversation memory with rich formatting."""
    try:
        console.print(f"[cyan]🔍 Searching for: '{query}'[/cyan]")
        results = await manager.find(query, limit=10)

        if not results:
            console.print("[yellow]📭 No results found.[/yellow]")
            return

        table = Table(title=f"🔍 Search Results for '{query}'", show_header=True)
        table.add_column("Node ID", style="cyan", width=8)
        table.add_column("Type", style="magenta", width=8)
        table.add_column("Content", style="white")
        table.add_column("Score", style="green", width=8)

        for result in results:
            node_type = result["node_type"]
            content = result.get("summary") or result["content"]
            score = result["relevance_score"]
            node_id = result.get("node_id", "unknown")

            # Truncate long content
            if len(content) > 100:
                content = content[:100] + "..."

            table.add_row(
                str(node_id),
                "👤 User" if node_type == "user" else "🤖 AI",
                content,
                f"{score:.2f}",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Error searching: {e}[/red]")


async def expand_node(manager, node_id: int, conv_id: str):
    """Expand a node to show full content."""
    try:
        console.print(f"[cyan]🔍 Expanding node {node_id}...[/cyan]")
        result = await manager.get_node_details(node_id, conv_id)

        if result is None:
            console.print(
                f"[red]❌ Node {node_id} not found in conversation {conv_id}[/red]"
            )
            return

        # Create detailed panel
        node_info = []
        node_info.append(f"**Type:** {result.get('node_type', 'unknown')}")
        node_info.append(f"**Timestamp:** {result.get('timestamp', 'unknown')}")
        node_info.append(f"**Sequence:** {result.get('sequence_number', 'unknown')}")
        node_info.append(f"**Compression Level:** {result.get('level', 'unknown')}")
        node_info.append(f"**Line Count:** {result.get('line_count', 'unknown')}")
        node_info.append(f"**Tokens Used:** {result.get('tokens_used', 'unknown')}")

        # Show topics if available
        topics = result.get("topics", [])
        if topics:
            node_info.append(f"**Topics:** {', '.join(topics)}")

        # Show summary if available
        summary = result.get("summary", "")
        if summary:
            node_info.append(f"**Summary:** {summary}")

        node_info.append("")
        node_info.append("**Content:**")
        node_info.append(result.get("content", ""))

        console.print(
            Panel(
                Markdown("\n".join(node_info)),
                title=f"📄 Node {node_id} Details",
                border_style="blue",
                expand=False,
            )
        )

    except Exception as e:
        console.print(f"[red]❌ Error expanding node {node_id}: {e}[/red]")


async def show_recent_messages(manager, limit: int = 10):
    """Show recent conversation messages."""
    try:
        results = await manager.find("", limit=limit)
        if not results:
            console.print("[yellow]📭 No recent messages found.[/yellow]")
            return

        table = Table(title="📜 Recent Messages", show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Node ID", style="cyan", width=8)
        table.add_column("Type", style="magenta", width=8)
        table.add_column("Time", style="blue", width=16)
        table.add_column("Content", style="white")

        for i, result in enumerate(results[:limit], 1):
            node_type = result["node_type"]
            content = result.get("summary") or result["content"]
            timestamp = result["timestamp"]
            node_id = result.get("node_id", "unknown")

            # Truncate long content
            if len(content) > 80:
                content = content[:80] + "..."

            table.add_row(
                str(i),
                str(node_id),
                "👤 User" if node_type == "user" else "🤖 AI",
                timestamp[:16],
                content,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Error getting recent messages: {e}[/red]")


async def show_hierarchical_summaries(
    manager, conv_id: str, start_node: int, end_node: int
):
    """Show hierarchical summaries for a range of nodes (Phase 4 functionality)."""
    try:
        console.print(
            f"[bold cyan]📊 Phase 4: Hierarchical Summaries for nodes {start_node}-{end_node}[/bold cyan]"
        )
        console.print("[dim]This demonstrates the advanced hierarchy system[/dim]")
        console.print()

        # Get nodes in range
        nodes = await manager.storage.get_nodes_in_range(
            conversation_id=conv_id,
            start_node_id=start_node,
            end_node_id=end_node,
        )

        if not nodes:
            console.print(
                f"[red]❌ No nodes found in range {start_node}-{end_node}[/red]"
            )
            return

        # Group nodes by compression level
        nodes_by_level = {"FULL": [], "SUMMARY": [], "META": [], "ARCHIVE": []}

        for node in nodes:
            level_name = node.level.name
            nodes_by_level[level_name].append(node)

        # Display summary
        total_nodes = len(nodes)
        total_lines = sum(node.line_count or 0 for node in nodes)

        console.print(
            Panel(
                f"📈 Found {total_nodes} nodes ({total_lines} total lines) in range {start_node}-{end_node}",
                title="📋 Range Summary",
                border_style="blue",
            )
        )

        # Show compression distribution
        compression_stats = {
            level: len(nodes) for level, nodes in nodes_by_level.items() if nodes
        }

        if compression_stats:
            table = Table(title="🗜️ Compression Distribution", show_header=True)
            table.add_column("Level", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Description", style="white")

            level_descriptions = {
                "FULL": "🟢 Recent nodes with complete content",
                "SUMMARY": "🟡 Older nodes with 1-2 sentence summaries",
                "META": "🟠 Groups of summary nodes (20-40 nodes each)",
                "ARCHIVE": "🔴 Very compressed high-level context",
            }

            for level, count in compression_stats.items():
                table.add_row(
                    level,
                    str(count),
                    level_descriptions.get(level, "Unknown level"),
                )

            console.print(table)

        # Show details for each level
        level_order = ["FULL", "SUMMARY", "META", "ARCHIVE"]
        for level in level_order:
            if level in nodes_by_level and nodes_by_level[level]:
                level_nodes = nodes_by_level[level]
                await show_level_details(level, level_nodes)

        # Show META group information
        meta_nodes = nodes_by_level.get("META", [])
        if meta_nodes:
            await show_meta_group_details(meta_nodes)

        console.print(
            "[bold green]✨ This demonstrates Phase 4: Advanced Hierarchy with 4-level compression![/bold green]"
        )
        console.print("[dim]FULL → SUMMARY → META → ARCHIVE[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Error showing hierarchical summaries: {e}[/red]")
        import traceback

        traceback.print_exc()


async def show_level_details(level: str, level_nodes):
    """Show details for a specific compression level."""
    level_descriptions = {
        "FULL": "🟢 Recent nodes with complete content",
        "SUMMARY": "🟡 Older nodes with 1-2 sentence summaries",
        "META": "🟠 Groups of summary nodes (20-40 nodes each)",
        "ARCHIVE": "🔴 Very compressed high-level context",
    }

    console.print(
        f"[bold]{level_descriptions[level]}[/bold] ({len(level_nodes)} nodes):"
    )

    for i, node in enumerate(level_nodes[:5], 1):  # Show first 5 nodes
        node_type_icon = "👤" if node.node_type.value == "user" else "🤖"

        # Determine content to show
        if node.summary:
            display_content = node.summary
            content_type = "[Summary]"
        else:
            display_content = node.content
            content_type = "[Full]"

        # Truncate long content
        if len(display_content) > 150:
            display_content = display_content[:150] + "..."

        console.print(
            f"   {i}. {node_type_icon} Node {node.node_id} {content_type}: {display_content}"
        )

    if len(level_nodes) > 5:
        console.print(f"   ... and {len(level_nodes) - 5} more nodes")
    console.print()


async def show_meta_group_details(meta_nodes):
    """Show details about META group nodes."""
    console.print("[bold]🔗 META Group Details:[/bold]")

    for node in meta_nodes:
        if node.summary_metadata:
            try:
                metadata = json.loads(node.summary_metadata)
                meta_info = metadata.get("meta_group_info", {})
                if meta_info:
                    console.print(
                        f"   📦 Node {node.node_id}: Groups nodes {meta_info.get('start_node_id')}-{meta_info.get('end_node_id')}"
                    )
                    console.print(
                        f"      Topics: {', '.join(meta_info.get('main_topics', [])[:3])}"
                    )
                    console.print(
                        f"      Contains: {meta_info.get('node_count')} nodes, {meta_info.get('total_lines')} lines"
                    )
            except (json.JSONDecodeError, KeyError):
                console.print(
                    f"   📦 Node {node.node_id}: META group (details unavailable)"
                )
        else:
            console.print(f"   📦 Node {node.node_id}: META group (no metadata)")

    console.print()


async def show_detailed_stats(manager):
    """Show detailed conversation statistics."""
    try:
        summary = await manager.get_conversation_summary()
        if "error" in summary:
            console.print(f"[red]❌ Error getting stats: {summary['error']}[/red]")
            return

        # Create comprehensive stats table
        table = Table(title="📊 Detailed Conversation Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description", style="white")

        table.add_row(
            "Total Nodes",
            str(summary.get("total_nodes", 0)),
            "Total conversation nodes in database",
        )
        table.add_row(
            "Recent Nodes",
            str(summary.get("recent_nodes", 0)),
            "Nodes at FULL compression level",
        )
        table.add_row(
            "Compressed Nodes",
            str(summary.get("compressed_nodes", 0)),
            "Nodes at SUMMARY/META/ARCHIVE levels",
        )

        # Add additional metrics if available
        if "compression_ratio" in summary:
            table.add_row(
                "Compression Ratio",
                f"{summary['compression_ratio']:.2%}",
                "Space saved through compression",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Error getting detailed stats: {e}[/red]")


# Add other commands from the original CLI


@app.command()
def summaries(
    conversation_id: str = typer.Argument(help="Conversation ID"),
    start_node: int = typer.Argument(help="Start node ID"),
    end_node: int = typer.Argument(help="End node ID"),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
    mcp_port: Optional[int] = typer.Option(None, "--mcp-port", help="MCP server port"),
):
    """Show hierarchical summaries for a range of nodes (Phase 4)."""
    asyncio.run(
        _show_summaries(conversation_id, start_node, end_node, db_path, mcp_port)
    )


async def _show_summaries(
    conversation_id: str,
    start_node: int,
    end_node: int,
    db_path: Optional[str],
    mcp_port: Optional[int],
):
    """Show hierarchical summaries implementation."""
    config = Config.from_env()
    if db_path:
        config.db_path = db_path
    if mcp_port:
        config.mcp_port = mcp_port

    try:
        conversation_id = await resolve_conversation_id(conversation_id, config.db_path)
        console.print(f"[dim]Resolved conversation ID: {conversation_id}[/dim]")
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)

    # Setup MCP
    mcp_server_url = f"http://127.0.0.1:{config.mcp_port}/mcp"
    if not check_mcp_server_running(mcp_server_url):
        console.print("[red]❌ MCP server is not running![/red]")
        console.print("[yellow]Please start the MCP server first[/yellow]")
        sys.exit(1)

    try:
        manager = HierarchicalConversationManager(config, mcp_server_url=mcp_server_url)
        await manager.start_conversation(conversation_id)
        await show_hierarchical_summaries(
            manager, conversation_id, start_node, end_node
        )
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@app.command()
def expand(
    conversation_id: str = typer.Argument(help="Conversation ID"),
    node_id: int = typer.Argument(help="Node ID to expand"),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
    mcp_port: Optional[int] = typer.Option(None, "--mcp-port", help="MCP server port"),
):
    """Expand a node to show its full content."""
    asyncio.run(_expand_node(conversation_id, node_id, db_path, mcp_port))


async def _expand_node(
    conversation_id: str,
    node_id: int,
    db_path: Optional[str],
    mcp_port: Optional[int],
):
    """Expand node implementation."""
    config = Config.from_env()
    if db_path:
        config.db_path = db_path
    if mcp_port:
        config.mcp_port = mcp_port

    try:
        conversation_id = await resolve_conversation_id(conversation_id, config.db_path)
        console.print(f"[dim]Resolved conversation ID: {conversation_id}[/dim]")
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)

    # Setup MCP
    mcp_server_url = f"http://127.0.0.1:{config.mcp_port}/mcp"
    if not check_mcp_server_running(mcp_server_url):
        console.print("[red]❌ MCP server is not running![/red]")
        sys.exit(1)

    try:
        manager = HierarchicalConversationManager(config, mcp_server_url=mcp_server_url)
        await manager.start_conversation(conversation_id)
        await expand_node(manager, node_id, conversation_id)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", "-c", help="Limit search to specific conversation"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
    mcp_port: Optional[int] = typer.Option(None, "--mcp-port", help="MCP server port"),
):
    """Search conversation memory across all conversations."""
    asyncio.run(_search_memory(query, conversation_id, limit, db_path, mcp_port))


async def _search_memory(
    query: str,
    conversation_id: Optional[str],
    limit: int,
    db_path: Optional[str],
    mcp_port: Optional[int],
):
    """Search memory implementation."""
    config = Config.from_env()
    if db_path:
        config.db_path = db_path
    if mcp_port:
        config.mcp_port = mcp_port

    if conversation_id:
        try:
            conversation_id = await resolve_conversation_id(
                conversation_id, config.db_path
            )
            console.print(f"[dim]Resolved conversation ID: {conversation_id}[/dim]")
        except ValueError as e:
            console.print(f"[red]❌ {e}[/red]")
            sys.exit(1)

    mcp_server_url = f"http://127.0.0.1:{config.mcp_port}/mcp"
    if not check_mcp_server_running(mcp_server_url):
        console.print("[red]❌ MCP server is not running![/red]")
        sys.exit(1)

    try:
        manager = HierarchicalConversationManager(config, mcp_server_url=mcp_server_url)
        if conversation_id:
            await manager.start_conversation(conversation_id)
        await search_memory(manager, query)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@app.command()
def list_conversations(
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
):
    """List all conversations in the database."""
    asyncio.run(_list_conversations(db_path))


async def _list_conversations(db_path: Optional[str]):
    """List conversations implementation."""
    config = Config.from_env()
    if db_path:
        config.db_path = db_path

    try:
        from .storage import DuckDBStorage

        storage = DuckDBStorage(config.db_path)

        # Get conversation list
        conversations = await storage.get_conversation_list()

        if not conversations:
            console.print("[yellow]📭 No conversations found.[/yellow]")
            return

        table = Table(title="📋 Conversations", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Created", style="blue")
        table.add_column("Last Updated", style="green")
        table.add_column("Nodes", style="yellow")
        table.add_column("Status", style="magenta")

        for conv in conversations:
            table.add_row(
                conv["id"][:8] + "...",
                conv.get("name", "<unnamed>"),
                conv["created"][:16] if conv["created"] else "Unknown",
                conv["last_updated"][:16] if conv["last_updated"] else "Unknown",
                str(conv["node_count"]),
                "Active" if conv["is_active"] else "Inactive",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@app.command()
def switch_conversation(
    identifier: str = typer.Argument(help="Conversation ID, partial ID, or name"),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
    mcp_port: Optional[int] = typer.Option(None, "--mcp-port", help="MCP server port"),
):
    """Switch to a conversation by ID, partial ID, or name and start chat session."""
    asyncio.run(_switch_conversation(identifier, db_path, mcp_port))


async def _switch_conversation(
    identifier: str,
    db_path: Optional[str],
    mcp_port: Optional[int],
):
    """Switch conversation implementation."""
    config = Config.from_env()
    if db_path:
        config.db_path = db_path
    if mcp_port:
        config.mcp_port = mcp_port

    try:
        conversation_id = await resolve_conversation_identifier(
            identifier, config.db_path
        )
        console.print(f"[green]✅ Switching to conversation: {conversation_id}[/green]")

        # Start chat session with resolved conversation
        await _chat_session(
            conversation_id,
            None,
            None,
            config.db_path,
            None,
            mcp_port,
            ".conversations",
            True,  # Enable streaming by default
        )
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")

        # Show available conversations
        console.print("\n[yellow]Available conversations:[/yellow]")
        await _list_conversations(config.db_path)


@app.command()
def export_conversation(
    conversation_id: str = typer.Argument(help="Conversation ID to export"),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
    mcp_port: Optional[int] = typer.Option(None, "--mcp-port", help="MCP server port"),
):
    """Export a conversation to JSON."""
    asyncio.run(_export_conversation(conversation_id, output_file, db_path, mcp_port))


async def _export_conversation(
    conversation_id: str,
    output_file: Optional[str],
    db_path: Optional[str],
    mcp_port: Optional[int],
):
    """Export conversation implementation."""
    config = Config.from_env()
    if db_path:
        config.db_path = db_path
    if mcp_port:
        config.mcp_port = mcp_port

    # Resolve partial conversation ID
    try:
        conversation_id = await resolve_conversation_id(conversation_id, config.db_path)
        console.print(f"[dim]Resolved conversation ID: {conversation_id}[/dim]")
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)

    # Setup MCP
    mcp_server_url = f"http://127.0.0.1:{config.mcp_port}/mcp"
    if not check_mcp_server_running(mcp_server_url):
        console.print("[red]❌ MCP server is not running![/red]")
        sys.exit(1)

    try:
        manager = HierarchicalConversationManager(config, mcp_server_url=mcp_server_url)
        await manager.start_conversation(conversation_id)

        summary = await manager.get_conversation_summary()

        if "error" in summary:
            console.print(f"[red]❌ Error: {summary['error']}[/red]")
            return

        # Export data
        export_data = {
            "conversation_id": conversation_id,
            "summary": summary,
            "exported_at": datetime.now().isoformat(),
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            console.print(f"[green]✅ Exported to {output_file}[/green]")
        else:
            console.print(json.dumps(export_data, indent=2))

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@app.command()
def server(
    action: str = typer.Argument(help="Action: start, stop, status"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="MCP server port"),
):
    """Manage MCP server."""
    config = Config.from_env()
    if port:
        config.mcp_port = port

    mcp_server_url = f"http://127.0.0.1:{config.mcp_port}/mcp"

    if action == "status":
        if check_mcp_server_running(mcp_server_url):
            console.print(
                f"[green]✅ MCP server is running on port {config.mcp_port}[/green]"
            )
        else:
            console.print(
                f"[red]❌ MCP server is not running on port {config.mcp_port}[/red]"
            )

    elif action == "start":
        console.print(
            f"[yellow]🚀 Starting MCP server on port {config.mcp_port}...[/yellow]"
        )
        console.print(
            f"[dim]Run: python -m hierarchical_memory_middleware.mcp_server.run_server[/dim]"
        )
        console.print(
            f"[yellow]Note: This command shows you how to start the server manually.[/yellow]"
        )

    elif action == "stop":
        console.print(
            f"[yellow]🛑 To stop the MCP server, use Ctrl+C in the server terminal[/yellow]"
        )

    else:
        console.print(
            f"[red]❌ Unknown action: {action}. Use: start, stop, status[/red]"
        )


@app.command()
def version():
    """Show version information."""
    try:
        from . import __version__

        console.print(
            f"[bold green]Hierarchical Memory Middleware v{__version__}[/bold green]"
        )
    except ImportError:
        console.print("[bold green]Hierarchical Memory Middleware[/bold green]")
        console.print("[dim]Version information not available[/dim]")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
