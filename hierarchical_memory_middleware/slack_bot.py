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
import hashlib
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
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
from .conversation_db_manager import ConversationDBManager


@dataclass
class SlackBotConfig:
    """Configuration for the Slack bot."""
    bot_token: str
    app_token: str
    # Optional customization
    response_thread: bool = True  # Reply in threads
    show_tool_calls: bool = True  # Show tool usage during streaming (hidden in final message)
    show_tool_details: bool = True  # Show tool arguments during execution
    show_thinking: bool = True  # Show "thinking" status
    verbose_tools: bool = False  # Show full tool details (args + results) in final message
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
            show_tool_details=os.environ.get("SLACK_SHOW_TOOL_DETAILS", "true").lower() == "true",
            show_thinking=os.environ.get("SLACK_SHOW_THINKING", "true").lower() == "true",
            verbose_tools=os.environ.get("SLACK_VERBOSE_TOOLS", "false").lower() == "true",
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
        working_dir: Optional[str] = None,
    ):
        """Initialize the Slack HMM bot.

        Args:
            slack_config: Slack bot configuration
            hmm_config: HMM configuration (uses defaults if not provided)
            permission_mode: Permission mode for Claude Code tools
                - "default": SDK handles permissions
                - "acceptEdits": Auto-approve file edits
                - "bypassPermissions": No prompts (for automation)
            working_dir: Working directory for file operations (defaults to cwd)
        """
        if not SLACK_BOLT_AVAILABLE:
            raise ImportError(
                "slack-bolt is required for Slack integration. "
                "Install it with: pip install slack-bolt aiohttp"
            )

        self.slack_config = slack_config
        self.hmm_config = hmm_config or Config()
        self.permission_mode = permission_mode
        self.working_dir = working_dir or os.getcwd()

        # Initialize Slack app
        self.app = AsyncApp(token=slack_config.bot_token)

        # Track active conversations (channel_id -> conversation_manager)
        self._conversations: Dict[str, ClaudeAgentSDKConversationManager] = {}

        # Track active response tasks (for cancellation)
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # Track streaming state for each active task (for stop/interrupt functionality)
        # task_key -> {"response_text": str, "message_ts": str, "chunk_message_ts": list, "client": AsyncWebClient, "channel_id": str}
        self._streaming_state: Dict[str, Dict[str, Any]] = {}

        # Message batching: accumulate rapid messages before processing
        self._pending_messages: Dict[str, List[Dict[str, Any]]] = {}  # task_key -> list of message data
        self._batch_timers: Dict[str, asyncio.Task] = {}  # task_key -> debounce timer task
        self._batch_delay: float = 1.5  # seconds to wait for additional messages

        # Track Slack client per conversation (for history tool)
        self._conversation_clients: Dict[str, AsyncWebClient] = {}
        self._conversation_channels: Dict[str, str] = {}

        # Bot user ID (filled on startup)
        self._bot_user_id: Optional[str] = None

        # Get model config for conversation managers
        self._model_config = ModelManager.get_model_config(self.hmm_config.work_model)

        # Per-conversation DB manager for concurrent access
        self._db_manager = ConversationDBManager(
            main_db_path=self.hmm_config.db_path,
            active_dir=None,  # Uses default: alongside main DB
            idle_timeout=timedelta(minutes=5),
            cleanup_interval=timedelta(minutes=1),
        )

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

        # Check if existing conversation's storage was synced back and closed (idle timeout)
        # If so, we need to recreate with fresh storage
        if conv_key in self._conversations:
            conv_id = hashlib.sha256(conv_key.encode()).hexdigest()[:36]
            if not self._db_manager.is_active(conv_id):
                # Storage was synced back due to idle timeout - remove stale manager
                logger.info(f"Conversation {conv_key} storage was synced back, recreating with fresh storage")
                del self._conversations[conv_key]

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
- `download_slack_file`: Download files shared in Slack to the slack_files directory
- `get_slack_file_info`: Get file metadata including download URL
- `upload_slack_file`: Upload and share a file to Slack (code, images, documents, etc.)

FILE HANDLING: When users share files in Slack:
1. Use `download_slack_file` with the file URL to save it locally
2. For images: After downloading, use the Read tool with the saved path to view the image
3. For text files: Use the Read tool to read the contents

SHARING FILES: To share files back to Slack:
1. Use `upload_slack_file` with the file path to upload and share it
2. Optionally add a title and initial_comment to describe the file
3. The file will be posted in the current channel (or specify a different channel_id)

Use these tools when you need more context about what was discussed in the channel."""

            # Create Slack MCP config to enable Slack tools
            slack_mcp_config = SlackMCPConfig(
                bot_token=self.slack_config.bot_token,
                channel_id=channel_id,
                working_dir=self.working_dir,
            )

            # Generate deterministic conversation ID based on channel/thread
            # This allows persistent memory across bot restarts
            conv_id = hashlib.sha256(conv_key.encode()).hexdigest()[:36]

            # Get isolated storage for this conversation (per-conversation DuckDB)
            storage = await self._db_manager.get_storage(conv_id)

            manager = ClaudeAgentSDKConversationManager(
                config=self.hmm_config,
                model_config=model_config,
                storage=storage,  # Use per-conversation isolated storage
                permission_mode=self.permission_mode,
                enable_memory_tools=True,
                agentic_mode=False,  # Slack is interactive, not agentic
                custom_instructions=slack_instructions,
                slack_mcp_config=slack_mcp_config,
            )

            # Start conversation with the same ID used for storage
            await manager.start_conversation(conversation_id=conv_id)

            # Mark storage as dirty since we just started/activated it
            self._db_manager.mark_dirty(conv_id)

            self._conversations[conv_key] = manager
            self._conversation_channels[conv_key] = channel_id
            logger.info(f"Created new conversation for {conv_key} (conv_id: {conv_id})")

        return self._conversations[conv_key]

    async def _handle_message_event(
        self, event: dict, say: Callable, client: AsyncWebClient
    ):
        """Handle incoming message events (DMs)."""
        channel_type = event.get("channel_type", "")
        channel_id = event.get("channel", "")
        logger.debug(f"Message event received: channel_type={channel_type}, channel={channel_id}, text={event.get('text', '')[:50]}")

        # Ignore bot messages (including our own)
        # But allow file_share subtype (messages with attachments)
        subtype = event.get("subtype")
        if event.get("bot_id") or (subtype and subtype != "file_share"):
            logger.debug(f"Ignoring bot/subtype message: bot_id={event.get('bot_id')}, subtype={subtype}")
            return

        # Ignore messages in channels (those need @mention)
        if channel_type not in ("im", "mpim"):  # Direct message or multi-person DM
            logger.debug(f"Ignoring channel message (need @mention): channel_type={channel_type}")
            return

        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        files = event.get("files", [])

        # Check for "stop" command to interrupt active response
        if text.strip().lower() == "stop":
            stopped = await self._handle_stop_command(channel_id, thread_ts, client)
            if stopped:
                logger.info(f"User stopped active response in {channel_id}")
                return
            # If no active task, fall through and treat as regular message

        # Build message text including file descriptions
        message_parts = []
        if text.strip():
            message_parts.append(text.strip())

        # Add file information to the message
        for file in files:
            file_info = self._format_file_info(file)
            if file_info:
                message_parts.append(file_info)

        # Skip if no text and no files
        if not message_parts:
            return

        text = "\n\n".join(message_parts)

        await self._process_message(
            channel_id=channel_id,
            user_id=user_id,
            text=text,
            thread_ts=thread_ts,
            say=say,
            client=client,
        )

    async def _handle_mention_event(
        self, event: dict, say: Callable, client: AsyncWebClient
    ):
        """Handle @mentions in channels."""
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        files = event.get("files", [])

        # Remove the @mention from the text
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        # Check for "stop" command to interrupt active response
        if text.lower() == "stop":
            stopped = await self._handle_stop_command(channel_id, thread_ts, client)
            if stopped:
                logger.info(f"User stopped active response in {channel_id}")
                return
            # If no active task, fall through and treat as regular message

        # Build message text including file descriptions
        message_parts = []
        if text:
            message_parts.append(text)

        # Add file information to the message
        for file in files:
            file_info = self._format_file_info(file)
            if file_info:
                message_parts.append(file_info)

        if not message_parts:
            await say(
                "Hi! How can I help you? Just @ mention me with your question.",
                thread_ts=thread_ts if self.slack_config.response_thread else None,
            )
            return

        text = "\n\n".join(message_parts)

        await self._process_message(
            channel_id=channel_id,
            user_id=user_id,
            text=text,
            thread_ts=thread_ts,
            say=say,
            client=client,
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

            # Update with final response (no tool info for slash commands - simple mode)
            await respond(self._format_response(response_text, is_final=True))

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
    ):
        """Process an incoming message and generate a response.

        Uses message batching: if multiple messages arrive quickly,
        they're combined into a single request to avoid losing messages.

        Args:
            channel_id: Slack channel ID
            user_id: User who sent the message
            text: Message text
            thread_ts: Thread timestamp for replies
            say: Slack say function
            client: Slack web client
        """
        task_key = f"{channel_id}:{thread_ts}"

        # If there's an active response being generated, queue this message
        if task_key in self._active_tasks:
            # Add to pending for next batch
            if task_key not in self._pending_messages:
                self._pending_messages[task_key] = []
            self._pending_messages[task_key].append({
                "user_id": user_id,
                "text": text,
                "say": say,
                "client": client,
            })
            logger.info(f"Queued message for {task_key} (response in progress)")
            return

        # Add message to pending batch
        if task_key not in self._pending_messages:
            self._pending_messages[task_key] = []

        self._pending_messages[task_key].append({
            "user_id": user_id,
            "text": text,
            "say": say,
            "client": client,
        })

        # Cancel existing batch timer if any
        if task_key in self._batch_timers:
            self._batch_timers[task_key].cancel()

        # Set a new batch timer
        async def process_batch():
            await asyncio.sleep(self._batch_delay)
            await self._process_batched_messages(channel_id, thread_ts, task_key)

        self._batch_timers[task_key] = asyncio.create_task(process_batch())

    async def _process_batched_messages(
        self,
        channel_id: str,
        thread_ts: str,
        task_key: str,
    ):
        """Process all batched messages as a single request."""
        # Get and clear pending messages
        messages = self._pending_messages.pop(task_key, [])
        self._batch_timers.pop(task_key, None)

        if not messages:
            return

        # Combine all message texts
        combined_texts = [msg["text"] for msg in messages]
        combined_text = "\n\n".join(combined_texts) if len(combined_texts) > 1 else combined_texts[0]

        # Use the last message's say/client
        last_msg = messages[-1]
        say = last_msg["say"]
        client = last_msg["client"]
        user_id = last_msg["user_id"]

        # Show thinking indicator
        thinking_msg = None
        if self.slack_config.show_thinking:
            indicator = "🤔 Thinking..." if len(messages) == 1 else f"🤔 Processing {len(messages)} messages..."
            result = await say(
                indicator,
                thread_ts=thread_ts if self.slack_config.response_thread else None,
            )
            thinking_msg = result.get("ts")

        # Create task for response generation
        task = asyncio.create_task(
            self._generate_response(
                channel_id=channel_id,
                user_id=user_id,
                text=combined_text,
                thread_ts=thread_ts,
                thinking_msg_ts=thinking_msg,
                client=client,
                task_key=task_key,
            )
        )
        self._active_tasks[task_key] = task

        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Task cancelled for {task_key}")
        finally:
            self._active_tasks.pop(task_key, None)
            self._streaming_state.pop(task_key, None)

            # Check if there are more pending messages that arrived while we were processing
            if task_key in self._pending_messages and self._pending_messages[task_key]:
                logger.info(f"Processing {len(self._pending_messages[task_key])} queued messages for {task_key}")
                await self._process_batched_messages(channel_id, thread_ts, task_key)

    async def _handle_stop_command(
        self,
        channel_id: str,
        thread_ts: str,
        client: AsyncWebClient,
    ) -> bool:
        """Handle "stop" command to interrupt an active response.

        Args:
            channel_id: Slack channel ID
            thread_ts: Thread timestamp
            client: Slack web client

        Returns:
            True if a task was stopped, False if no active task found
        """
        task_key = f"{channel_id}:{thread_ts}"

        if task_key not in self._active_tasks:
            return False

        task = self._active_tasks[task_key]
        state = self._streaming_state.get(task_key, {})

        logger.info(f"Stopping active task for {task_key}")

        # Cancel the task and wait for it to complete
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected when cancelling
        except Exception as e:
            logger.warning(f"Error while cancelling task: {e}")

        # Get the partial response and conversation manager
        # (streaming state should now be finalized after task cancellation)
        response_text = state.get("response_text", "")
        message_ts = state.get("message_ts")
        chunk_message_ts = state.get("chunk_message_ts", [])
        conv_key = f"{channel_id}:{thread_ts}" if thread_ts else channel_id
        manager = self._conversations.get(conv_key)

        # Save the partial response to memory
        # The manager's _current_tool_calls will have all tool calls that were
        # received before cancellation
        if manager:
            try:
                # Log tool call state before saving
                tool_calls_count = len(getattr(manager, '_current_tool_calls', []))
                tool_results_count = len(getattr(manager, '_current_tool_results', []))
                logger.info(
                    f"Saving partial response for {task_key}: "
                    f"{len(response_text)} chars, "
                    f"{tool_calls_count} tool calls, "
                    f"{tool_results_count} tool results"
                )
                await manager.save_partial_response(response_text)
                logger.info(f"Saved partial response for {task_key}")
            except Exception as e:
                logger.warning(f"Failed to save partial response: {e}")

        # Update the Slack message to indicate interruption
        if message_ts:
            # Get tool information from streaming state to preserve in final message
            active_tool_calls = state.get("active_tool_calls", {})
            completed_tools = state.get("completed_tools", [])
            completed_tool_names = state.get("completed_tool_names", [])

            # Format the response with tool information preserved
            if response_text or completed_tools or active_tool_calls:
                # Use _format_response to include tool information
                interrupted_text = self._format_response(
                    text=response_text,
                    active_tools=active_tool_calls,
                    completed_tool_names=completed_tool_names,
                    is_final=True,
                    completed_tools=completed_tools,
                )
                interrupted_text += "\n\n_⏹️ Response stopped by user_"
            else:
                interrupted_text = "_⏹️ Response stopped by user before any output_"

            try:
                await client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=interrupted_text,
                )
                # Delete any chunk messages
                for chunk_ts in chunk_message_ts:
                    try:
                        await client.chat_delete(channel=channel_id, ts=chunk_ts)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed to update message after stop: {e}")

        return True

    async def _generate_response(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        thread_ts: str,
        thinking_msg_ts: Optional[str],
        client: AsyncWebClient,
        task_key: Optional[str] = None,
    ):
        """Generate and stream response to Slack.

        Args:
            channel_id: Slack channel ID
            user_id: User who sent the message
            text: Message text
            thread_ts: Thread timestamp
            thinking_msg_ts: Timestamp of "thinking" message to update
            client: Slack web client
            task_key: Key for tracking this task's streaming state
        """
        # Get conversation manager
        manager = await self._get_or_create_conversation(channel_id, thread_ts)

        # Initialize streaming state for stop command support
        if task_key:
            self._streaming_state[task_key] = {
                "response_text": "",
                "message_ts": thinking_msg_ts,
                "chunk_message_ts": [],
                "client": client,
                "channel_id": channel_id,
            }

        # Collect response
        response_text = ""
        # Track tool calls: {tool_id: {"name": str, "args": dict, "result": str, "is_error": bool}}
        active_tool_calls: Dict[str, Dict[str, Any]] = {}
        completed_tool_names: List[str] = []  # Just tool names for final message
        completed_tools: List[Dict[str, Any]] = []  # Full tool info for verbose mode
        last_update = datetime.now()
        # Track additional message timestamps for chunked messages (for updates)
        chunk_message_ts: List[str] = []

        try:
            async for event in manager.chat_stream(text, include_tool_events=True):
                if isinstance(event, str):
                    response_text += event

                    # Update streaming state for stop command support
                    if task_key and task_key in self._streaming_state:
                        self._streaming_state[task_key]["response_text"] = response_text

                    # Update message periodically (to show streaming progress)
                    now = datetime.now()
                    if (now - last_update).total_seconds() >= self.slack_config.update_interval_seconds:
                        chunk_message_ts = await self._update_response_message(
                            client=client,
                            channel_id=channel_id,
                            message_ts=thinking_msg_ts,
                            text=response_text,
                            active_tools=active_tool_calls,
                            completed_tool_names=completed_tool_names,
                            completed_tools=completed_tools,
                            is_final=False,
                            chunk_message_ts=chunk_message_ts,
                        )
                        # Update streaming state with chunk timestamps
                        if task_key and task_key in self._streaming_state:
                            self._streaming_state[task_key]["chunk_message_ts"] = chunk_message_ts
                        last_update = now

                elif isinstance(event, ToolCallEvent):
                    if self.slack_config.show_tool_calls:
                        # Track the active tool call with details
                        active_tool_calls[event.tool_id] = {
                            "name": event.tool_name,
                            "args": event.tool_input,
                            "result": None,
                            "is_error": False,
                        }
                        # Update streaming state for stop command support
                        if task_key and task_key in self._streaming_state:
                            self._streaming_state[task_key]["active_tool_calls"] = active_tool_calls.copy()
                        # Update to show tool usage
                        chunk_message_ts = await self._update_response_message(
                            client=client,
                            channel_id=channel_id,
                            message_ts=thinking_msg_ts,
                            text=response_text,
                            active_tools=active_tool_calls,
                            completed_tool_names=completed_tool_names,
                            completed_tools=completed_tools,
                            is_final=False,
                            chunk_message_ts=chunk_message_ts,
                        )

                elif isinstance(event, ToolResultEvent):
                    if self.slack_config.show_tool_calls:
                        # Update tool call with result
                        if event.tool_id in active_tool_calls:
                            tool_info = active_tool_calls[event.tool_id]
                            tool_info["result"] = event.content
                            tool_info["is_error"] = event.is_error
                            # Move to completed (both simple and detailed)
                            completed_tool_names.append(tool_info["name"])
                            completed_tools.append(tool_info.copy())
                            del active_tool_calls[event.tool_id]
                        # Update streaming state for stop command support
                        if task_key and task_key in self._streaming_state:
                            self._streaming_state[task_key]["active_tool_calls"] = active_tool_calls.copy()
                            self._streaming_state[task_key]["completed_tools"] = completed_tools.copy()
                            self._streaming_state[task_key]["completed_tool_names"] = completed_tool_names.copy()
                        # Update to show tool result
                        chunk_message_ts = await self._update_response_message(
                            client=client,
                            channel_id=channel_id,
                            message_ts=thinking_msg_ts,
                            text=response_text,
                            active_tools=active_tool_calls,
                            completed_tool_names=completed_tool_names,
                            completed_tools=completed_tools,
                            is_final=False,
                            chunk_message_ts=chunk_message_ts,
                        )

                elif isinstance(event, UsageEvent):
                    # Log usage for tracking
                    logger.info(
                        f"Usage for {channel_id}: {event.input_tokens} in, "
                        f"{event.output_tokens} out, ${event.cost_usd:.4f}"
                    )

            # Send final response (with tool details shown in verbose mode)
            await self._update_response_message(
                client=client,
                channel_id=channel_id,
                message_ts=thinking_msg_ts,
                text=response_text,
                active_tools={},  # No active tools in final
                completed_tool_names=completed_tool_names,
                completed_tools=completed_tools,
                is_final=True,
                chunk_message_ts=chunk_message_ts,
            )

            # Mark storage as dirty (conversation nodes were saved)
            if manager.conversation_id:
                self._db_manager.mark_dirty(manager.conversation_id)

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
        active_tools: Dict[str, Dict[str, Any]],
        completed_tool_names: List[str],
        is_final: bool,
        completed_tools: Optional[List[Dict[str, Any]]] = None,
        chunk_message_ts: Optional[List[str]] = None,
    ) -> List[str]:
        """Update the response message in Slack.

        Args:
            client: Slack web client
            channel_id: Channel ID
            message_ts: Message timestamp to update (None to post new)
            text: Response text
            active_tools: Currently executing tools with details
            completed_tool_names: Names of completed tools (for final summary)
            is_final: Whether this is the final update
            completed_tools: Full tool info (args + results) for verbose mode
            chunk_message_ts: Timestamps of additional chunk messages (for updates)

        Returns:
            Updated list of chunk message timestamps
        """
        if chunk_message_ts is None:
            chunk_message_ts = []

        # Format the message
        formatted_text = self._format_response(text, active_tools, completed_tool_names, is_final, completed_tools)

        # Chunk if too long
        chunks = self._chunk_message(formatted_text)

        try:
            if message_ts and len(chunks) == 1:
                # Single chunk - update existing message, delete any extra chunk messages
                await client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=chunks[0],
                )
                # Delete any previous chunk messages that are no longer needed
                for old_ts in chunk_message_ts:
                    try:
                        await client.chat_delete(channel=channel_id, ts=old_ts)
                    except Exception:
                        pass  # Ignore errors deleting old messages
                return []  # No more chunk messages

            elif message_ts:
                # Multiple chunks - update first, then update/create additional chunks
                await client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=chunks[0],
                )

                new_chunk_ts = []
                for i, chunk in enumerate(chunks[1:]):
                    if i < len(chunk_message_ts):
                        # Update existing chunk message
                        await client.chat_update(
                            channel=channel_id,
                            ts=chunk_message_ts[i],
                            text=chunk,
                        )
                        new_chunk_ts.append(chunk_message_ts[i])
                    else:
                        # Post new chunk message
                        result = await client.chat_postMessage(
                            channel=channel_id,
                            text=chunk,
                            thread_ts=message_ts,  # Reply in thread
                        )
                        new_chunk_ts.append(result["ts"])

                # Delete any extra old chunk messages that are no longer needed
                for old_ts in chunk_message_ts[len(chunks) - 1:]:
                    try:
                        await client.chat_delete(channel=channel_id, ts=old_ts)
                    except Exception:
                        pass  # Ignore errors deleting old messages

                return new_chunk_ts

            else:
                # Post new messages (no existing message to update)
                for chunk in chunks:
                    await client.chat_postMessage(
                        channel=channel_id,
                        text=chunk,
                    )
                return []

        except Exception as e:
            logger.exception(f"Error updating Slack message: {e}")
            return chunk_message_ts  # Return unchanged on error

    def _format_response(
        self,
        text: str,
        active_tools: Optional[Dict[str, Dict[str, Any]]] = None,
        completed_tool_names: Optional[List[str]] = None,
        is_final: bool = True,
        completed_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format response text for Slack.

        During streaming: shows detailed tool information (name, args, status)
        After completion: hides tool details unless verbose_tools is enabled

        Args:
            text: Response text
            active_tools: Currently executing tools with details
            completed_tool_names: Names of completed tools
            is_final: Whether this is the final response
            completed_tools: Full tool info (args + results) for verbose mode

        Returns:
            Formatted Slack message
        """
        parts = []

        # Show tool information during streaming OR in final message (verbose mode)
        show_tools = self.slack_config.show_tool_calls and (
            not is_final or (is_final and self.slack_config.verbose_tools)
        )

        if show_tools:
            tool_lines = []

            if self.slack_config.verbose_tools:
                # VERBOSE MODE: Show full tool details (args + results)
                if completed_tools:
                    for tool_info in completed_tools:
                        name = tool_info["name"]
                        args = tool_info.get("args", {})
                        result = tool_info.get("result", "")
                        is_error = tool_info.get("is_error", False)

                        # Check if this is an Edit operation (show full content for edits)
                        is_edit_tool = name == "Edit"

                        # Format args (as JSON code block)
                        args_block = ""
                        if args:
                            import json
                            args_str = json.dumps(args, indent=2)
                            # Don't truncate Edit args so user sees exactly what was edited
                            if not is_edit_tool and len(args_str) > 500:
                                args_str = args_str[:497] + "..."
                            args_block = f"\n```{args_str}```"

                        # Format result
                        result_block = ""
                        if result:
                            result_preview = result[:800] if len(result) > 800 else result
                            # Truncate to first 10 lines
                            lines = result_preview.split("\n")
                            if len(lines) > 10:
                                result_preview = "\n".join(lines[:10]) + f"\n... ({len(lines)} lines total)"
                            status = ":x:" if is_error else ":white_check_mark:"
                            # Include tool name next to the checkmark
                            result_block = f"\n{status} `{name}` ```{result_preview}```"

                        tool_lines.append(f":wrench: `{name}`{args_block}{result_block}")

                # Show active tools (in progress)
                if active_tools:
                    for tool_id, tool_info in active_tools.items():
                        name = tool_info["name"]
                        args = tool_info.get("args", {})

                        # Format args (as JSON code block, truncated)
                        args_block = ""
                        if args:
                            import json
                            args_str = json.dumps(args, indent=2)
                            if len(args_str) > 500:
                                args_str = args_str[:497] + "..."
                            args_block = f"\n```{args_str}```"

                        tool_lines.append(f":wrench: `{name}`{args_block}\n:hourglass_flowing_sand: _running..._")
            else:
                # COMPACT MODE: Show simple tool status
                # Show completed tools (compact)
                if completed_tool_names:
                    for name in completed_tool_names:
                        tool_lines.append(f":white_check_mark: `{name}`")

                # Show active tools with details
                if active_tools and self.slack_config.show_tool_details:
                    for tool_id, tool_info in active_tools.items():
                        name = tool_info["name"]
                        args = tool_info.get("args", {})

                        # Format args preview (truncate long values)
                        args_preview = self._format_tool_args(args)
                        if args_preview:
                            tool_lines.append(f":wrench: `{name}` {args_preview}")
                        else:
                            tool_lines.append(f":wrench: `{name}` :hourglass_flowing_sand:")
                elif active_tools:
                    # Show just tool names without details
                    for tool_id, tool_info in active_tools.items():
                        tool_lines.append(f":wrench: `{tool_info['name']}` :hourglass_flowing_sand:")

            if tool_lines:
                if is_final and self.slack_config.verbose_tools:
                    parts.append("*Tool Calls:*")
                parts.append("\n\n".join(tool_lines))
                if is_final and self.slack_config.verbose_tools:
                    parts.append("---")  # Divider before response text
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

    def _format_tool_args(self, args: Dict[str, Any], max_length: int = 60) -> str:
        """Format tool arguments for display.

        Args:
            args: Tool arguments dictionary
            max_length: Maximum length for the preview

        Returns:
            Formatted args preview string
        """
        if not args:
            return ""

        # Special formatting for common tool types
        if "pattern" in args:
            # Grep/Glob tool
            preview = f"pattern=`{args['pattern']}`"
            if "path" in args:
                preview += f" in `{args['path']}`"
            return preview
        elif "file_path" in args:
            # Read/Write/Edit tool
            path = args["file_path"]
            if len(path) > 40:
                path = "..." + path[-37:]
            return f"`{path}`"
        elif "command" in args:
            # Bash tool
            cmd = args["command"]
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            return f"`{cmd}`"
        elif "query" in args:
            # Search tool
            query = args["query"]
            if len(query) > 50:
                query = query[:47] + "..."
            return f"`{query}`"
        elif "url" in args:
            # WebFetch tool
            return f"`{args['url'][:50]}...`" if len(args.get("url", "")) > 50 else f"`{args.get('url', '')}`"

        # Generic: show first key-value pair
        for key, value in args.items():
            val_str = str(value)
            if len(val_str) > 40:
                val_str = val_str[:37] + "..."
            return f"{key}=`{val_str}`"

        return ""

    def _format_file_info(self, file: dict) -> Optional[str]:
        """Format a Slack file attachment into a text description for the AI.

        Args:
            file: Slack file object from the event

        Returns:
            Formatted string describing the file, or None if not processable
        """
        file_id = file.get("id", "")
        file_type = file.get("filetype", "unknown")
        file_name = file.get("name", "unnamed")
        file_title = file.get("title", file_name)
        mimetype = file.get("mimetype", "")
        url_private = file.get("url_private", "")

        # Check if it's an image
        is_image = mimetype.startswith("image/") or file_type in (
            "png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"
        )

        if is_image:
            # For images, include the URL so Claude can download and analyze it
            parts = [f"[Image attached: {file_title}]"]
            if url_private:
                parts.append(f"URL: {url_private}")
                parts.append("(Use download_slack_file tool with this URL to view the image)")
            return "\n".join(parts)

        # For other files, provide metadata and download instructions
        file_size = file.get("size", 0)
        size_str = self._format_file_size(file_size) if file_size else "unknown size"

        parts = [f"[File attached: {file_title} ({file_type}, {size_str})]"]

        # Include file ID for API access
        if file_id:
            parts.append(f"File ID: {file_id}")

        if url_private:
            parts.append(f"URL: {url_private}")
            parts.append("(Use download_slack_file tool with this URL to access the file contents)")

        # Include preview content for text files if available
        preview = file.get("preview", "")
        if preview:
            parts.append(f"Preview:\n{preview}")

        return "\n".join(parts)

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

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

        # Run startup healing for orphaned conversation files
        merged_count = await self._db_manager.startup()
        if merged_count > 0:
            logger.info(f"Healed {merged_count} orphaned conversation files from previous shutdown")

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
        print(f"   - Database: {self.hmm_config.db_path}")
        print(f"   - Active conversations dir: {self._db_manager.active_dir}")
        print("\nPress Ctrl+C to stop.")

        await handler.start_async()

    async def close(self):
        """Clean up resources."""
        logger.info("Shutting down SlackHMMBot...")

        # Cancel batch timers
        for timer in self._batch_timers.values():
            timer.cancel()

        # Cancel active tasks
        for task in self._active_tasks.values():
            task.cancel()

        # Close conversation managers (but don't close their storage - DB manager handles that)
        for manager in self._conversations.values():
            # Don't close storage via manager - the DB manager will handle sync-back
            manager.storage = None  # Prevent manager from closing shared storage
            await manager.close()

        self._conversations.clear()
        self._active_tasks.clear()
        self._batch_timers.clear()
        self._pending_messages.clear()

        # Shutdown DB manager - this syncs all active conversations back to main DB
        logger.info("Syncing active conversations to main database...")
        await self._db_manager.shutdown()

        logger.info("SlackHMMBot shut down complete")


async def run_slack_bot(
    permission_mode: str = "default",
    db_path: Optional[str] = None,
    working_dir: Optional[str] = None,
    verbose_tools: bool = False,
):
    """Run the Slack bot.

    Args:
        permission_mode: Permission mode for Claude Code tools
        db_path: Optional custom database path
        working_dir: Working directory for file operations (defaults to cwd)
        verbose_tools: Show full tool details in final messages
    """
    # Load config
    slack_config = SlackBotConfig.from_env()
    hmm_config = Config()

    if db_path:
        hmm_config.db_path = db_path

    # Override verbose_tools from CLI flag
    if verbose_tools:
        slack_config.verbose_tools = True

    # Create and start bot
    bot = SlackHMMBot(
        slack_config=slack_config,
        hmm_config=hmm_config,
        permission_mode=permission_mode,
        working_dir=working_dir or os.getcwd(),
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
    parser.add_argument(
        "--verbose-tools",
        action="store_true",
        help="Show full tool call details (arguments and results) in final messages. Useful for debugging/coding sessions.",
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
        working_dir=work_path,
        verbose_tools=args.verbose_tools,
    ))


if __name__ == "__main__":
    main()
