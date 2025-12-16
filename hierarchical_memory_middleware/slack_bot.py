"""Slack bot integration for Hierarchical Memory Middleware.

This module provides a Slack bot interface that wraps HMM + Claude Code,
enabling multi-user, async conversations via Slack with persistent memory.

Features:
- Socket Mode (no public URL needed)
- Persistent memory per channel/DM
- Streaming responses with progress updates
- Long-running task support
- Multi-user access with shared memory per channel

Usage:
    # Create a .env file with:
    # SLACK_BOT_TOKEN=xoxb-...
    # SLACK_APP_TOKEN=xapp-...

    # Or set environment variables directly, then run:
    hmm-slack  # or python -m hierarchical_memory_middleware.slack_bot
"""

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Try multiple locations: current dir, home dir, project dir
    for env_path in [
        Path.cwd() / ".env",
        Path.home() / ".env",
        Path(__file__).parent.parent / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        load_dotenv()  # Try default locations
except ImportError:
    pass  # dotenv not installed, rely on environment variables

logger = logging.getLogger(__name__)

# Check for slack_bolt before importing
try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_sdk.web.async_client import AsyncWebClient
    SLACK_BOLT_AVAILABLE = True
except ImportError:
    SLACK_BOLT_AVAILABLE = False
    AsyncApp = None
    AsyncSocketModeHandler = None
    AsyncWebClient = None


from .config import Config
from .model_manager import ModelManager
from .middleware.claude_agent_sdk_manager import (
    ClaudeAgentSDKConversationManager,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    SlackMCPConfig,
)


@dataclass
class SlackBotConfig:
    """Configuration for the Slack bot."""
    bot_token: str
    app_token: str
    # Optional customization
    response_thread: bool = True  # Reply in threads
    show_tool_calls: bool = True  # Show tool usage
    show_thinking: bool = True  # Show "thinking" status
    max_message_length: int = 3000  # Slack message limit (actual is 40k, but we chunk)
    update_interval_seconds: float = 2.0  # How often to update streaming message

    @classmethod
    def from_env(cls) -> "SlackBotConfig":
        """Create config from environment variables."""
        bot_token = os.environ.get("SLACK_BOT_TOKEN")
        app_token = os.environ.get("SLACK_APP_TOKEN")

        if not bot_token:
            raise ValueError("SLACK_BOT_TOKEN environment variable is required")
        if not app_token:
            raise ValueError("SLACK_APP_TOKEN environment variable is required (for Socket Mode)")

        return cls(
            bot_token=bot_token,
            app_token=app_token,
            response_thread=os.environ.get("SLACK_RESPONSE_THREAD", "true").lower() == "true",
            show_tool_calls=os.environ.get("SLACK_SHOW_TOOL_CALLS", "true").lower() == "true",
            show_thinking=os.environ.get("SLACK_SHOW_THINKING", "true").lower() == "true",
        )


class SlackHMMBot:
    """Slack bot that wraps HMM + Claude Code.

    Each channel/DM gets its own conversation with persistent memory.
    The bot responds to:
    - Direct messages
    - @mentions in channels
    - Slash commands (if configured)
    """

    def __init__(
        self,
        slack_config: SlackBotConfig,
        hmm_config: Optional[Config] = None,
        permission_mode: str = "default",
    ):
        """Initialize the Slack HMM bot.

        Args:
            slack_config: Slack bot configuration
            hmm_config: HMM configuration (uses defaults if not provided)
            permission_mode: Permission mode for Claude Code tools
                - "default": SDK handles permissions
                - "acceptEdits": Auto-approve file edits
                - "bypassPermissions": No prompts (for automation)
        """
        if not SLACK_BOLT_AVAILABLE:
            raise ImportError(
                "slack-bolt is required for Slack integration. "
                "Install it with: pip install slack-bolt aiohttp"
            )

        self.slack_config = slack_config
        self.hmm_config = hmm_config or Config()
        self.permission_mode = permission_mode

        # Initialize Slack app
        self.app = AsyncApp(token=slack_config.bot_token)

        # Track active conversations (channel_id -> conversation_manager)
        self._conversations: Dict[str, ClaudeAgentSDKConversationManager] = {}

        # Track active response tasks (for cancellation)
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # Track Slack client per conversation (for history tool)
        self._conversation_clients: Dict[str, AsyncWebClient] = {}
        self._conversation_channels: Dict[str, str] = {}

        # Bot user ID (filled on startup)
        self._bot_user_id: Optional[str] = None

        # Get model config for conversation managers
        self._model_config = ModelManager.get_model_config(self.hmm_config.work_model)

        # Setup event handlers
        self._setup_handlers()

        logger.info("SlackHMMBot initialized")

    def _setup_handlers(self):
        """Setup Slack event handlers."""

        # Handle direct messages
        @self.app.event("message")
        async def handle_message(event: dict, say: Callable, client: AsyncWebClient):
            await self._handle_message_event(event, say, client)

        # Handle app mentions (@bot)
        @self.app.event("app_mention")
        async def handle_mention(event: dict, say: Callable, client: AsyncWebClient):
            await self._handle_mention_event(event, say, client)

        # Handle slash command (optional)
        @self.app.command("/hmm")
        async def handle_slash_command(ack: Callable, command: dict, respond: Callable):
            await ack()  # Acknowledge within 3 seconds
            await self._handle_slash_command(command, respond)

    async def _get_or_create_conversation(
        self, channel_id: str, thread_ts: Optional[str] = None
    ) -> ClaudeAgentSDKConversationManager:
        """Get or create a conversation manager for a channel/thread.

        Args:
            channel_id: Slack channel ID
            thread_ts: Thread timestamp (if in a thread)

        Returns:
            Conversation manager for this context
        """
        # Use channel + thread as unique key (so threads have their own memory)
        conv_key = f"{channel_id}:{thread_ts}" if thread_ts else channel_id

        if conv_key not in self._conversations:
            # Create new conversation manager
            model_config = self._model_config

            # Slack-specific formatting instructions
            slack_instructions = """You are responding in Slack. You MUST use Slack's mrkdwn syntax, NOT standard Markdown.

CRITICAL - Slack mrkdwn differences from standard Markdown:
- Bold: Use *bold* NOT **bold** (single asterisks, not double!)
- Italic: Use _italic_ NOT *italic* (underscores, not asterisks!)
- Headers: NOT SUPPORTED - don't use # or ## - just use *bold text* on its own line
- Horizontal rules: NOT SUPPORTED - don't use --- or ***

Slack mrkdwn syntax:
- *bold* for emphasis
- _italic_ for secondary emphasis
- ~strikethrough~ for struck text
- `code` for inline code
- ```code blocks``` for multi-line code (no language specifier after ```)
- > for block quotes
- • or numbered lists (1. 2. 3.) for lists
- Links: <URL|display text>

Keep responses concise. Slack is a conversational medium - avoid walls of text.

CHANNEL CONTEXT: You automatically receive the last 2 messages from the channel before each user message (shown as "[Recent channel messages for context:]"). Use this context to understand the conversation flow and reference previous messages when relevant.

SLACK TOOLS: You have access to tools for fetching more Slack context:
- `get_slack_channel_history`: Fetch more messages from the channel (beyond the last 2)
- `get_slack_thread_replies`: Get all replies in a thread
- `search_slack_messages`: Search for messages across Slack (requires search:read scope)
- `get_slack_user_info`: Look up user details by user ID

Use these tools when you need more context about what was discussed in the channel."""

            # Create Slack MCP config to enable Slack tools
            slack_mcp_config = SlackMCPConfig(
                bot_token=self.slack_config.bot_token,
                channel_id=channel_id,
            )

            manager = ClaudeAgentSDKConversationManager(
                config=self.hmm_config,
                model_config=model_config,
                permission_mode=self.permission_mode,
                enable_memory_tools=True,
                agentic_mode=False,  # Slack is interactive, not agentic
                custom_instructions=slack_instructions,
                slack_mcp_config=slack_mcp_config,
            )

            # Start conversation with deterministic ID based on channel/thread
            # This allows persistent memory across bot restarts
            import hashlib
            conv_id = hashlib.sha256(conv_key.encode()).hexdigest()[:36]
            await manager.start_conversation(conversation_id=conv_id)

            self._conversations[conv_key] = manager
            self._conversation_channels[conv_key] = channel_id
            logger.info(f"Created new conversation for {conv_key}")

        return self._conversations[conv_key]

    async def _get_recent_messages(
        self,
        client: AsyncWebClient,
        channel_id: str,
        before_ts: Optional[str] = None,
        limit: int = 2,
    ) -> List[Dict[str, Any]]:
        """Fetch recent messages from a channel.

        Args:
            client: Slack web client
            channel_id: Channel to fetch from
            before_ts: Fetch messages before this timestamp (exclusive)
            limit: Maximum number of messages to return

        Returns:
            List of message dicts with 'user', 'text', 'ts' keys
        """
        try:
            # Fetch channel history
            kwargs = {
                "channel": channel_id,
                "limit": limit + 1,  # Fetch one extra to exclude current message
            }
            if before_ts:
                kwargs["latest"] = before_ts
                kwargs["inclusive"] = False  # Don't include the message at latest

            result = await client.conversations_history(**kwargs)
            messages = result.get("messages", [])

            # Filter out bot messages and the current message
            filtered = []
            for msg in messages:
                # Skip bot messages
                if msg.get("bot_id") or msg.get("subtype"):
                    continue
                # Skip the triggering message itself
                if before_ts and msg.get("ts") == before_ts:
                    continue
                filtered.append({
                    "user": msg.get("user", "unknown"),
                    "text": msg.get("text", ""),
                    "ts": msg.get("ts"),
                })
                if len(filtered) >= limit:
                    break

            # Return in chronological order (oldest first)
            return list(reversed(filtered))

        except Exception as e:
            logger.warning(f"Failed to fetch recent messages: {e}")
            return []

    async def _handle_message_event(
        self, event: dict, say: Callable, client: AsyncWebClient
    ):
        """Handle incoming message events (DMs)."""
        channel_type = event.get("channel_type", "")
        channel_id = event.get("channel", "")
        logger.debug(f"Message event received: channel_type={channel_type}, channel={channel_id}, text={event.get('text', '')[:50]}")

        # Ignore bot messages (including our own)
        if event.get("bot_id") or event.get("subtype"):
            logger.debug(f"Ignoring bot/subtype message: bot_id={event.get('bot_id')}, subtype={event.get('subtype')}")
            return

        # Ignore messages in channels (those need @mention)
        if channel_type not in ("im", "mpim"):  # Direct message or multi-person DM
            logger.debug(f"Ignoring channel message (need @mention): channel_type={channel_type}")
            return

        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")

        if not text.strip():
            return

        # Fetch recent channel context (last 2 messages before this one)
        recent_context = await self._get_recent_messages(
            client=client,
            channel_id=channel_id,
            before_ts=event.get("ts"),
            limit=2,
        )

        await self._process_message(
            channel_id=channel_id,
            user_id=user_id,
            text=text,
            thread_ts=thread_ts,
            say=say,
            client=client,
            recent_context=recent_context,
        )

    async def _handle_mention_event(
        self, event: dict, say: Callable, client: AsyncWebClient
    ):
        """Handle @mentions in channels."""
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")

        # Remove the @mention from the text
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        if not text:
            await say(
                "Hi! How can I help you? Just @ mention me with your question.",
                thread_ts=thread_ts if self.slack_config.response_thread else None,
            )
            return

        # Fetch recent channel context (last 2 messages before this one)
        recent_context = await self._get_recent_messages(
            client=client,
            channel_id=channel_id,
            before_ts=event.get("ts"),
            limit=2,
        )

        await self._process_message(
            channel_id=channel_id,
            user_id=user_id,
            text=text,
            thread_ts=thread_ts,
            say=say,
            client=client,
            recent_context=recent_context,
        )

    async def _handle_slash_command(self, command: dict, respond: Callable):
        """Handle /hmm slash command."""
        text = command.get("text", "").strip()
        channel_id = command.get("channel_id")
        user_id = command.get("user_id")

        if not text:
            await respond(
                "Usage: `/hmm <your question>`\n"
                "I'll respond with the help of Claude and remember our conversation!"
            )
            return

        # Special commands
        if text.lower() == "stats":
            await self._send_stats(channel_id, respond)
            return
        elif text.lower() == "clear":
            await self._clear_conversation(channel_id, respond)
            return
        elif text.lower() == "help":
            await respond(
                "*HMM Slack Bot Commands:*\n"
                "• `/hmm <question>` - Ask a question\n"
                "• `/hmm stats` - Show conversation statistics\n"
                "• `/hmm clear` - Clear conversation memory\n"
                "• `/hmm help` - Show this help message\n\n"
                "You can also DM me or @mention me in channels!"
            )
            return

        # Process as regular message
        await respond("🤔 Thinking...")

        # Get conversation manager
        manager = await self._get_or_create_conversation(channel_id)

        try:
            # Stream response
            response_text = ""
            async for event in manager.chat_stream(text, include_tool_events=True):
                if isinstance(event, str):
                    response_text += event

            # Update with final response
            await respond(self._format_response(response_text))

        except Exception as e:
            logger.exception(f"Error processing slash command: {e}")
            await respond(f"❌ Error: {str(e)}")

    async def _process_message(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        thread_ts: str,
        say: Callable,
        client: AsyncWebClient,
        recent_context: Optional[List[Dict[str, Any]]] = None,
    ):
        """Process an incoming message and generate a response.

        Args:
            channel_id: Slack channel ID
            user_id: User who sent the message
            text: Message text
            thread_ts: Thread timestamp for replies
            say: Slack say function
            client: Slack web client
            recent_context: Recent messages from the channel (for context)
        """
        # Cancel any existing task for this channel/thread
        task_key = f"{channel_id}:{thread_ts}"
        if task_key in self._active_tasks:
            self._active_tasks[task_key].cancel()

        # Show thinking indicator
        thinking_msg = None
        if self.slack_config.show_thinking:
            result = await say(
                "🤔 Thinking...",
                thread_ts=thread_ts if self.slack_config.response_thread else None,
            )
            thinking_msg = result.get("ts")

        # Create task for response generation
        task = asyncio.create_task(
            self._generate_response(
                channel_id=channel_id,
                user_id=user_id,
                text=text,
                thread_ts=thread_ts,
                thinking_msg_ts=thinking_msg,
                client=client,
                recent_context=recent_context,
            )
        )
        self._active_tasks[task_key] = task

        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Task cancelled for {task_key}")
        finally:
            self._active_tasks.pop(task_key, None)

    async def _generate_response(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        thread_ts: str,
        thinking_msg_ts: Optional[str],
        client: AsyncWebClient,
        recent_context: Optional[List[Dict[str, Any]]] = None,
    ):
        """Generate and stream response to Slack.

        Args:
            channel_id: Slack channel ID
            user_id: User who sent the message
            text: Message text
            thread_ts: Thread timestamp
            thinking_msg_ts: Timestamp of "thinking" message to update
            client: Slack web client
            recent_context: Recent messages from the channel (for context)
        """
        # Get conversation manager
        manager = await self._get_or_create_conversation(channel_id, thread_ts)

        # Build message with context if available
        if recent_context:
            context_lines = ["[Recent channel messages for context:]"]
            for msg in recent_context:
                user = msg.get("user", "unknown")
                msg_text = msg.get("text", "")
                context_lines.append(f"  <@{user}>: {msg_text}")
            context_lines.append("")
            context_lines.append("[Your message:]")
            context_lines.append(text)
            full_text = "\n".join(context_lines)
        else:
            full_text = text

        # Collect response
        response_text = ""
        tool_calls: List[str] = []
        last_update = datetime.now()

        try:
            async for event in manager.chat_stream(full_text, include_tool_events=True):
                if isinstance(event, str):
                    response_text += event

                    # Update message periodically (to show streaming progress)
                    now = datetime.now()
                    if (now - last_update).total_seconds() >= self.slack_config.update_interval_seconds:
                        await self._update_response_message(
                            client=client,
                            channel_id=channel_id,
                            message_ts=thinking_msg_ts,
                            text=response_text,
                            tool_calls=tool_calls,
                            is_final=False,
                        )
                        last_update = now

                elif isinstance(event, ToolCallEvent):
                    if self.slack_config.show_tool_calls:
                        tool_calls.append(f"🔧 `{event.tool_name}`")
                        # Update to show tool usage
                        await self._update_response_message(
                            client=client,
                            channel_id=channel_id,
                            message_ts=thinking_msg_ts,
                            text=response_text,
                            tool_calls=tool_calls,
                            is_final=False,
                        )

                elif isinstance(event, UsageEvent):
                    # Log usage for tracking
                    logger.info(
                        f"Usage for {channel_id}: {event.input_tokens} in, "
                        f"{event.output_tokens} out, ${event.cost_usd:.4f}"
                    )

            # Send final response
            await self._update_response_message(
                client=client,
                channel_id=channel_id,
                message_ts=thinking_msg_ts,
                text=response_text,
                tool_calls=tool_calls,
                is_final=True,
            )

        except Exception as e:
            logger.exception(f"Error generating response: {e}")
            error_text = f"❌ Sorry, I encountered an error: {str(e)}"
            if thinking_msg_ts:
                await client.chat_update(
                    channel=channel_id,
                    ts=thinking_msg_ts,
                    text=error_text,
                )
            else:
                await client.chat_postMessage(
                    channel=channel_id,
                    text=error_text,
                    thread_ts=thread_ts if self.slack_config.response_thread else None,
                )

    async def _update_response_message(
        self,
        client: AsyncWebClient,
        channel_id: str,
        message_ts: Optional[str],
        text: str,
        tool_calls: List[str],
        is_final: bool,
    ):
        """Update the response message in Slack.

        Args:
            client: Slack web client
            channel_id: Channel ID
            message_ts: Message timestamp to update (None to post new)
            text: Response text
            tool_calls: List of tool calls made
            is_final: Whether this is the final update
        """
        # Format the message
        formatted_text = self._format_response(text, tool_calls, is_final)

        # Chunk if too long
        chunks = self._chunk_message(formatted_text)

        try:
            if message_ts and len(chunks) == 1:
                # Update existing message
                await client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=chunks[0],
                )
            elif message_ts:
                # First chunk updates, rest are new messages
                await client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=chunks[0],
                )
                for chunk in chunks[1:]:
                    await client.chat_postMessage(
                        channel=channel_id,
                        text=chunk,
                        thread_ts=message_ts,  # Reply in thread
                    )
            else:
                # Post new messages
                for chunk in chunks:
                    await client.chat_postMessage(
                        channel=channel_id,
                        text=chunk,
                    )
        except Exception as e:
            logger.exception(f"Error updating Slack message: {e}")

    def _format_response(
        self,
        text: str,
        tool_calls: Optional[List[str]] = None,
        is_final: bool = True,
    ) -> str:
        """Format response text for Slack.

        Args:
            text: Response text
            tool_calls: List of tool calls made
            is_final: Whether this is the final response

        Returns:
            Formatted Slack message
        """
        parts = []

        # Add tool calls summary if any
        if tool_calls and self.slack_config.show_tool_calls:
            parts.append(" ".join(tool_calls))
            parts.append("")

        # Add response text
        if text:
            parts.append(text)
        elif not is_final:
            parts.append("⏳ Working...")

        # Add indicator if still processing
        if not is_final and text:
            parts.append("\n_Still typing..._")

        return "\n".join(parts)

    def _chunk_message(self, text: str) -> List[str]:
        """Split message into chunks that fit Slack's limits.

        Args:
            text: Full message text

        Returns:
            List of message chunks
        """
        if len(text) <= self.slack_config.max_message_length:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= self.slack_config.max_message_length:
                chunks.append(remaining)
                break

            # Find a good break point (newline or space)
            chunk_size = self.slack_config.max_message_length

            # Try to break at newline
            last_newline = remaining[:chunk_size].rfind("\n")
            if last_newline > chunk_size // 2:
                chunk_size = last_newline + 1
            else:
                # Try to break at space
                last_space = remaining[:chunk_size].rfind(" ")
                if last_space > chunk_size // 2:
                    chunk_size = last_space + 1

            chunks.append(remaining[:chunk_size])
            remaining = remaining[chunk_size:]

        return chunks

    async def _send_stats(self, channel_id: str, respond: Callable):
        """Send conversation statistics."""
        manager = self._conversations.get(channel_id)
        if not manager:
            await respond("No conversation history for this channel yet!")
            return

        try:
            stats = await manager.get_conversation_stats()

            levels = stats.get("compression_levels", {})
            token_stats = stats.get("token_stats", {})

            message = (
                f"*📊 Conversation Statistics*\n\n"
                f"*Messages:* {stats.get('total_nodes', 0)} total "
                f"({stats.get('user_messages', 0)} user, {stats.get('ai_messages', 0)} AI)\n\n"
                f"*Memory Levels:*\n"
                f"• Full: {levels.get('full', 0)} messages\n"
                f"• Summary: {levels.get('summary', 0)} messages\n"
                f"• Meta: {levels.get('meta', 0)} groups\n"
                f"• Archive: {levels.get('archive', 0)} archives\n\n"
                f"*Token Stats:*\n"
                f"• Current context: ~{token_stats.get('total_current_tokens', 0):,} tokens\n"
                f"• Original content: ~{token_stats.get('total_original_tokens', 0):,} tokens\n"
                f"• Compression ratio: {token_stats.get('overall_compression_ratio', 0)}x\n"
                f"• Tokens saved: {token_stats.get('tokens_saved_percent', 0)}%"
            )

            await respond(message)

        except Exception as e:
            logger.exception(f"Error getting stats: {e}")
            await respond(f"❌ Error getting stats: {str(e)}")

    async def _clear_conversation(self, channel_id: str, respond: Callable):
        """Clear conversation memory for a channel."""
        if channel_id in self._conversations:
            manager = self._conversations.pop(channel_id)
            await manager.close()
            await respond("✅ Conversation memory cleared! Starting fresh.")
        else:
            await respond("No conversation history to clear.")

    async def start(self):
        """Start the Slack bot."""
        logger.info("Starting SlackHMMBot...")

        # Get bot user ID
        try:
            auth_result = await self.app.client.auth_test()
            self._bot_user_id = auth_result.get("user_id")
            bot_name = auth_result.get("user")
            logger.info(f"Bot authenticated as @{bot_name} ({self._bot_user_id})")
        except Exception as e:
            logger.error(f"Failed to authenticate: {e}")
            raise

        # Start Socket Mode handler
        handler = AsyncSocketModeHandler(self.app, self.slack_config.app_token)

        print(f"🤖 SlackHMMBot is running! (Bot: @{bot_name})")
        print("   - DM me directly")
        print("   - @mention me in channels")
        print("   - Use /hmm <question> command")
        print("\nPress Ctrl+C to stop.")

        await handler.start_async()

    async def close(self):
        """Clean up resources."""
        logger.info("Shutting down SlackHMMBot...")

        # Cancel active tasks
        for task in self._active_tasks.values():
            task.cancel()

        # Close conversation managers
        for manager in self._conversations.values():
            await manager.close()

        self._conversations.clear()
        self._active_tasks.clear()

        logger.info("SlackHMMBot shut down complete")


async def run_slack_bot(
    permission_mode: str = "default",
    db_path: Optional[str] = None,
):
    """Run the Slack bot.

    Args:
        permission_mode: Permission mode for Claude Code tools
        db_path: Optional custom database path
    """
    # Load config
    slack_config = SlackBotConfig.from_env()
    hmm_config = Config()

    if db_path:
        hmm_config.db_path = db_path

    # Create and start bot
    bot = SlackHMMBot(
        slack_config=slack_config,
        hmm_config=hmm_config,
        permission_mode=permission_mode,
    )

    # Handle shutdown
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        asyncio.create_task(bot.close())
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.close()


def main():
    """CLI entry point for the Slack bot."""
    import argparse

    parser = argparse.ArgumentParser(description="HMM Slack Bot")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Working directory for the bot (where to find dagster assets, etc.)",
    )
    parser.add_argument(
        "--permission-mode",
        choices=["default", "acceptEdits", "bypassPermissions"],
        default="default",
        help="Permission mode for Claude Code tools",
    )
    parser.add_argument(
        "--db-path",
        help="Custom database path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Skip all permission checks (alias for --permission-mode bypassPermissions)",
    )

    args = parser.parse_args()

    # Handle --dangerously-skip-permissions as alias
    if args.dangerously_skip_permissions:
        args.permission_mode = "bypassPermissions"

    # Change to the specified working directory
    work_path = os.path.abspath(args.path)
    if not os.path.isdir(work_path):
        print(f"❌ Error: Path '{args.path}' does not exist or is not a directory")
        sys.exit(1)
    os.chdir(work_path)
    print(f"📁 Working directory: {work_path}")

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check for slack-bolt
    if not SLACK_BOLT_AVAILABLE:
        print("❌ Error: slack-bolt is required for Slack integration.")
        print("   Install it with: pip install slack-bolt aiohttp")
        sys.exit(1)

    # Check for required environment variables
    if not os.environ.get("SLACK_BOT_TOKEN"):
        print("❌ Error: SLACK_BOT_TOKEN environment variable is required")
        print("   Get it from: https://api.slack.com/apps → OAuth & Permissions")
        sys.exit(1)

    if not os.environ.get("SLACK_APP_TOKEN"):
        print("❌ Error: SLACK_APP_TOKEN environment variable is required")
        print("   Get it from: https://api.slack.com/apps → Basic Information → App-Level Tokens")
        print("   (Create one with 'connections:write' scope)")
        sys.exit(1)

    # Run the bot
    asyncio.run(run_slack_bot(
        permission_mode=args.permission_mode,
        db_path=args.db_path,
    ))


if __name__ == "__main__":
    main()
