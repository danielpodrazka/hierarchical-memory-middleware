#!/usr/bin/env python3
"""Slack-enabled stdio MCP server for Claude Agent SDK integration.

This server extends the base memory server with Slack-specific tools
for fetching channel history using the Slack API.

Usage:
    python -m hierarchical_memory_middleware.mcp_server.stdio_slack_memory_server \
        --conversation-id <id> --db-path <path> \
        --slack-bot-token <token> --slack-channel-id <channel>
"""

import argparse
import logging
import sys
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP

# Suppress logging to stderr since stdio is used for communication
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/hmm_slack_stdio_server.log")],
)
logger = logging.getLogger(__name__)


def create_slack_memory_server(
    conversation_id: str,
    db_path: str,
    slack_bot_token: str,
    slack_channel_id: str,
) -> FastMCP:
    """Create a FastMCP server with memory tools + Slack tools.

    Args:
        conversation_id: The conversation ID to use for all operations
        db_path: Path to the DuckDB database
        slack_bot_token: Slack bot OAuth token (xoxb-...)
        slack_channel_id: Default Slack channel ID for history queries

    Returns:
        Configured FastMCP server instance
    """
    # Import the base memory server creator
    from .stdio_memory_server import create_memory_server

    # Create base memory server with all standard tools
    mcp = create_memory_server(conversation_id, db_path)

    # Now add Slack-specific tools
    # These tools use the Slack Web API to fetch channel history

    @mcp.tool()
    async def get_slack_channel_history(
        limit: int = 10,
        before_ts: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch recent messages from the Slack channel.

        Use this tool to get context about recent channel activity
        that happened before the current conversation.

        Args:
            limit: Maximum number of messages to fetch (default: 10, max: 100)
            before_ts: Optional timestamp - fetch messages before this time
            channel_id: Optional channel ID (defaults to current channel)

        Returns:
            List of recent messages with user, text, and timestamp
        """
        try:
            # Use httpx for async HTTP requests (lighter than slack_sdk for simple calls)
            import httpx

            # Validate limit
            limit = min(max(1, limit), 100)

            # Use provided channel or default
            target_channel = channel_id or slack_channel_id

            # Build request params
            params = {
                "channel": target_channel,
                "limit": limit,
            }
            if before_ts:
                params["latest"] = before_ts
                params["inclusive"] = "false"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://slack.com/api/conversations.history",
                    params=params,
                    headers={"Authorization": f"Bearer {slack_bot_token}"},
                    timeout=10.0,
                )
                data = response.json()

            if not data.get("ok"):
                error = data.get("error", "Unknown error")
                return {"error": f"Slack API error: {error}"}

            messages = data.get("messages", [])

            # Format messages for readability
            formatted_messages = []
            for msg in messages:
                # Skip bot messages and subtypes (joins, leaves, etc.)
                if msg.get("bot_id") or msg.get("subtype"):
                    continue

                formatted_messages.append({
                    "user": msg.get("user", "unknown"),
                    "text": msg.get("text", ""),
                    "timestamp": msg.get("ts"),
                    "thread_ts": msg.get("thread_ts"),
                })

            # Return in chronological order (oldest first)
            formatted_messages.reverse()

            return {
                "success": True,
                "channel_id": target_channel,
                "count": len(formatted_messages),
                "messages": formatted_messages,
            }

        except ImportError:
            return {"error": "httpx not installed. Install with: pip install httpx"}
        except Exception as e:
            logger.error(f"Error fetching Slack history: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_slack_thread_replies(
        thread_ts: str,
        limit: int = 50,
        channel_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch replies in a Slack thread.

        Use this tool to get the full context of a threaded conversation.

        Args:
            thread_ts: The timestamp of the parent message (thread_ts)
            limit: Maximum number of replies to fetch (default: 50, max: 100)
            channel_id: Optional channel ID (defaults to current channel)

        Returns:
            List of thread replies with user, text, and timestamp
        """
        try:
            import httpx

            limit = min(max(1, limit), 100)
            target_channel = channel_id or slack_channel_id

            params = {
                "channel": target_channel,
                "ts": thread_ts,
                "limit": limit,
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://slack.com/api/conversations.replies",
                    params=params,
                    headers={"Authorization": f"Bearer {slack_bot_token}"},
                    timeout=10.0,
                )
                data = response.json()

            if not data.get("ok"):
                error = data.get("error", "Unknown error")
                return {"error": f"Slack API error: {error}"}

            messages = data.get("messages", [])

            # Format messages
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "user": msg.get("user", "unknown"),
                    "text": msg.get("text", ""),
                    "timestamp": msg.get("ts"),
                    "is_parent": msg.get("ts") == thread_ts,
                })

            return {
                "success": True,
                "channel_id": target_channel,
                "thread_ts": thread_ts,
                "count": len(formatted_messages),
                "messages": formatted_messages,
            }

        except ImportError:
            return {"error": "httpx not installed. Install with: pip install httpx"}
        except Exception as e:
            logger.error(f"Error fetching Slack thread: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def search_slack_messages(
        query: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search for messages in Slack.

        Use this tool to find specific messages or topics discussed in Slack.
        Note: Requires the search:read scope on the bot token.

        Args:
            query: Search query (supports Slack search syntax)
            limit: Maximum number of results (default: 20, max: 100)

        Returns:
            List of matching messages with context
        """
        try:
            import httpx

            limit = min(max(1, limit), 100)

            params = {
                "query": query,
                "count": limit,
                "sort": "timestamp",
                "sort_dir": "desc",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://slack.com/api/search.messages",
                    params=params,
                    headers={"Authorization": f"Bearer {slack_bot_token}"},
                    timeout=15.0,
                )
                data = response.json()

            if not data.get("ok"):
                error = data.get("error", "Unknown error")
                # Common error: missing search:read scope
                if error == "missing_scope":
                    return {
                        "error": "Bot token doesn't have search:read scope. "
                        "Add this scope in Slack app settings to enable search."
                    }
                return {"error": f"Slack API error: {error}"}

            matches = data.get("messages", {}).get("matches", [])

            formatted_results = []
            for match in matches:
                formatted_results.append({
                    "user": match.get("user", "unknown"),
                    "text": match.get("text", ""),
                    "timestamp": match.get("ts"),
                    "channel_id": match.get("channel", {}).get("id"),
                    "channel_name": match.get("channel", {}).get("name"),
                    "permalink": match.get("permalink"),
                })

            return {
                "success": True,
                "query": query,
                "count": len(formatted_results),
                "results": formatted_results,
            }

        except ImportError:
            return {"error": "httpx not installed. Install with: pip install httpx"}
        except Exception as e:
            logger.error(f"Error searching Slack: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_slack_user_info(
        user_id: str,
    ) -> Dict[str, Any]:
        """Get information about a Slack user.

        Use this tool to look up user details like display name and real name.

        Args:
            user_id: The Slack user ID (e.g., U01234567)

        Returns:
            User information including name and display name
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://slack.com/api/users.info",
                    params={"user": user_id},
                    headers={"Authorization": f"Bearer {slack_bot_token}"},
                    timeout=10.0,
                )
                data = response.json()

            if not data.get("ok"):
                error = data.get("error", "Unknown error")
                return {"error": f"Slack API error: {error}"}

            user = data.get("user", {})
            profile = user.get("profile", {})

            return {
                "success": True,
                "user_id": user_id,
                "name": user.get("name"),
                "real_name": user.get("real_name"),
                "display_name": profile.get("display_name"),
                "email": profile.get("email"),
                "is_bot": user.get("is_bot", False),
            }

        except ImportError:
            return {"error": "httpx not installed. Install with: pip install httpx"}
        except Exception as e:
            logger.error(f"Error getting Slack user info: {e}")
            return {"error": str(e)}

    return mcp


def main():
    """Main entry point for the Slack-enabled stdio memory server."""
    parser = argparse.ArgumentParser(description="Slack-enabled stdio MCP memory server")
    parser.add_argument("--conversation-id", required=True, help="Conversation ID")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    parser.add_argument("--slack-bot-token", required=True, help="Slack bot OAuth token")
    parser.add_argument("--slack-channel-id", required=True, help="Default Slack channel ID")
    args = parser.parse_args()

    logger.info(
        f"Starting Slack-enabled stdio memory server for conversation: {args.conversation_id}"
    )

    try:
        mcp = create_slack_memory_server(
            conversation_id=args.conversation_id,
            db_path=args.db_path,
            slack_bot_token=args.slack_bot_token,
            slack_channel_id=args.slack_channel_id,
        )
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
