#!/usr/bin/env python3
"""Standalone script to run the hierarchical memory MCP server."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.mcp_server.memory_server import MemoryMCPServer


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the server."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )


async def main() -> None:
    """Main function to run the MCP server."""
    parser = argparse.ArgumentParser(
        description="Run the Hierarchical Memory MCP Server"
    )
    parser.add_argument(
        "--transport",
        choices=["http", "streamable-http", "stdio", "sse"],
        default="streamable-http",
        help="Transport protocol to use (default: streamable-http)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--db-path",
        help="Path to DuckDB database file (default: ./conversations.db)",
    )
    parser.add_argument(
        "--work-model",
        default="claude-sonnet-4-20250514",
        help="Model to use for work tasks (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument("--conversation-id", help="Resume a specific conversation ID")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Create configuration (load from .env file at project root)
        env_file_path = project_root / ".env"
        config = Config.from_env(env_file=str(env_file_path))
        # Override with command line arguments
        if args.db_path is not None:
            config.db_path = args.db_path
        config.work_model = args.work_model
        config.summary_model = args.work_model  # Use same model for now
        config.mcp_port = args.port  # Use the port from command line

        logger.info(f"Initializing MCP server with config: {config}")

        # Create and initialize the server
        server = MemoryMCPServer(config)

        # Start a conversation if specified
        if args.conversation_id:
            conv_id = await server.start_conversation(args.conversation_id)
            logger.info(f"Resumed conversation: {conv_id}")
        else:
            conv_id = await server.start_conversation()
            logger.info(f"Started new conversation: {conv_id}")

        # Run the server
        logger.info(
            f"Starting MCP server on {args.transport}://{args.host}:{args.port}"
        )
        logger.info(
            "Available tools: expand_node, search_memory, get_conversation_stats, get_recent_nodes"
        )
        logger.info("Press Ctrl+C to stop the server")

        if args.transport == "stdio":
            # For stdio, use sync run
            server.mcp.run(transport="stdio")
        else:
            # For HTTP transports, use async run
            await server.run_async(
                transport=args.transport, host=args.host, port=args.port
            )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Handle both async and sync execution contexts
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
