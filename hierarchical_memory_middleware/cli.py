"""CLI interface for testing hierarchical memory middleware."""

import asyncio
import json
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .config import Config
from .middleware.conversation_manager import HierarchicalConversationManager


app = typer.Typer(help="Hierarchical Memory Middleware CLI")
console = Console()


@app.command()
def chat(
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", "-c", help="Resume existing conversation"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
    recent_limit: Optional[int] = typer.Option(
        None, "--recent-limit", "-r", help="Recent nodes limit"
    ),
):
    """Start an interactive chat session."""
    asyncio.run(_chat_session(conversation_id, model, db_path, recent_limit))


async def _chat_session(
    conversation_id: Optional[str],
    model: Optional[str],
    db_path: Optional[str],
    recent_limit: Optional[int],
):
    """Run the interactive chat session."""
    # Load configuration
    config = Config.from_env_or_default()

    # Override with CLI arguments
    if model:
        config.work_model = model
    if db_path:
        config.db_path = db_path
    if recent_limit:
        config.recent_node_limit = recent_limit

    console.print("[bold green]Hierarchical Memory Middleware[/bold green]")
    console.print(f"Model: {config.work_model}")
    console.print(f"Database: {config.db_path}")
    console.print(f"Recent nodes limit: {config.recent_node_limit}")
    console.print()

    try:
        # Initialize conversation manager
        manager = HierarchicalConversationManager(config)

        # Start or resume conversation
        conv_id = await manager.start_conversation(conversation_id)
        console.print(f"[blue]Conversation ID: {conv_id}[/blue]")

        # Show conversation summary if resuming
        if conversation_id:
            summary = await manager.get_conversation_summary()
            if "error" not in summary:
                console.print(
                    Panel(
                        f"Total nodes: {summary['total_nodes']}\n"
                        f"Recent nodes: {summary['recent_nodes']}\n"
                        f"Compressed nodes: {summary['compressed_nodes']}",
                        title="Conversation Summary",
                        border_style="blue",
                    )
                )

        console.print(
            "[yellow]Type 'exit' to quit, 'summary' for stats, 'search <query>' to search memory[/yellow]"
        )
        console.print()

        while True:
            try:
                # Get user input
                user_input = typer.prompt("You")

                if user_input.lower() in ["exit", "quit", "bye"]:
                    break

                elif user_input.lower() == "summary":
                    summary = await manager.get_conversation_summary()
                    console.print(
                        Panel(
                            json.dumps(summary, indent=2),
                            title="Conversation Summary",
                            border_style="green",
                        )
                    )
                    continue

                elif user_input.lower().startswith("search "):
                    query = user_input[7:]  # Remove 'search ' prefix
                    results = await manager.search_memory(query)

                    if results:
                        table = Table(title=f"Search Results for '{query}'")
                        table.add_column("Node ID", style="cyan")
                        table.add_column("Type", style="magenta")
                        table.add_column("Content", style="white")
                        table.add_column("Score", style="green")

                        for result in results[:5]:  # Show top 5
                            table.add_row(
                                str(result["node_id"]),
                                result["node_type"],
                                result["content"][:100] + "..."
                                if len(result["content"]) > 100
                                else result["content"],
                                f"{result['relevance_score']:.2f}",
                            )

                        console.print(table)
                    else:
                        console.print("[yellow]No results found[/yellow]")
                    continue

                # Generate AI response
                with console.status("[bold green]Thinking..."):
                    response = await manager.chat(user_input)

                # Display response
                console.print(
                    Panel(
                        Text(response, style="white"),
                        title="Assistant",
                        border_style="green",
                    )
                )
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Chat interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                continue

    except Exception as e:
        console.print(f"[red]Failed to initialize: {str(e)}[/red]")
        sys.exit(1)

    console.print("[blue]Goodbye![/blue]")


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
    config = Config.from_env_or_default()
    if db_path:
        config.db_path = db_path

    try:
        from .storage import DuckDBStorage

        storage = DuckDBStorage(config.db_path)

        # Simple query to get conversations
        # For Phase 1, we'll just show basic info
        console.print(f"[blue]Database: {config.db_path}[/blue]")
        console.print(
            "[yellow]Note: Full conversation listing will be implemented in Phase 2[/yellow]"
        )

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def export_conversation(
    conversation_id: str = typer.Argument(help="Conversation ID to export"),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", "-d", help="Database path"
    ),
):
    """Export a conversation to JSON."""
    asyncio.run(_export_conversation(conversation_id, output_file, db_path))


async def _export_conversation(
    conversation_id: str, output_file: Optional[str], db_path: Optional[str]
):
    """Export conversation implementation."""
    config = Config.from_env_or_default()
    if db_path:
        config.db_path = db_path

    try:
        manager = HierarchicalConversationManager(config)
        await manager.start_conversation(conversation_id)

        # Get conversation data
        summary = await manager.get_conversation_summary()

        if "error" in summary:
            console.print(f"[red]Error: {summary['error']}[/red]")
            return

        # Export to file or stdout
        export_data = {
            "conversation_id": conversation_id,
            "summary": summary,
            "exported_at": "2025-01-14",  # Would use datetime in real implementation
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            console.print(f"[green]Exported to {output_file}[/green]")
        else:
            console.print(json.dumps(export_data, indent=2))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"Hierarchical Memory Middleware v{__version__}")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
