# Slack Bot Setup Guide

This guide walks you through setting up the HMM Slack bot to enable Claude-powered conversations with persistent memory in your Slack workspace.

## Prerequisites

- A Slack workspace where you have permission to install apps
- Python 3.11+ with HMM installed
- Claude CLI authenticated (for Claude Pro/Max subscription) OR Anthropic API key

## Quick Start

### 1. Install Slack Dependencies

```bash
pip install hierarchical-memory-middleware[slack]
# or
pip install slack-bolt aiohttp
```

### 2. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** → **From scratch**
3. Enter an app name (e.g., "HMM Bot") and select your workspace
4. Click **Create App**

### 3. Configure Bot Permissions

Navigate to **OAuth & Permissions** in the sidebar and add these **Bot Token Scopes**:

| Scope | Purpose |
|-------|---------|
| `app_mentions:read` | Receive @mentions in channels |
| `chat:write` | Send messages |
| `im:history` | Read DM history |
| `im:read` | Access DM info |
| `im:write` | Send DMs |
| `channels:history` | Read channel history (for thread context) |
| `commands` | Handle slash commands (optional) |

### 4. Enable Socket Mode

Socket Mode lets the bot receive events without exposing a public URL.

1. Go to **Socket Mode** in the sidebar
2. Toggle **Enable Socket Mode** to ON
3. You'll be prompted to create an app-level token:
   - Name: "socket-mode"
   - Scope: `connections:write`
   - Click **Generate**
4. **Copy the `xapp-...` token** - this is your `SLACK_APP_TOKEN`

### 5. Subscribe to Events

Go to **Event Subscriptions** in the sidebar:

1. Toggle **Enable Events** to ON
2. Under **Subscribe to bot events**, add:
   - `app_mention` - When the bot is @mentioned
   - `message.im` - Direct messages to the bot
   - `message.mpim` - Multi-person DM messages (optional)

### 6. Create Slash Command (Optional)

Go to **Slash Commands** in the sidebar:

1. Click **Create New Command**
2. Configure:
   - Command: `/hmm`
   - Description: "Ask the HMM AI assistant"
   - Usage hint: `<your question>`
3. Click **Save**

### 7. Install the App

1. Go to **Install App** in the sidebar
2. Click **Install to Workspace**
3. Review permissions and click **Allow**
4. **Copy the `xoxb-...` token** - this is your `SLACK_BOT_TOKEN`

### 8. Set Environment Variables

```bash
export SLACK_BOT_TOKEN="xoxb-your-bot-token"
export SLACK_APP_TOKEN="xapp-your-app-level-token"
```

### 9. Run the Bot

```bash
hmm-slack
# or
python -m hierarchical_memory_middleware.slack_bot
```

You should see:
```
🤖 SlackHMMBot is running! (Bot: @hmm-bot)
   - DM me directly
   - @mention me in channels
   - Use /hmm <question> command

Press Ctrl+C to stop.
```

## Usage

### Direct Messages
Just DM the bot with your question - it will respond with Claude's answer.

### Channel Mentions
In any channel where the bot is added, @mention it:
```
@HMM Bot What's the best way to structure a Python project?
```

### Slash Command
```
/hmm How do I set up a virtual environment?
```

### Special Commands (via /hmm)
- `/hmm stats` - Show conversation statistics and memory usage
- `/hmm clear` - Clear conversation memory
- `/hmm help` - Show help message

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SLACK_BOT_TOKEN` | Yes | Bot OAuth token (`xoxb-...`) |
| `SLACK_APP_TOKEN` | Yes | App-level token for Socket Mode (`xapp-...`) |
| `SLACK_RESPONSE_THREAD` | No | Reply in threads (default: `true`) |
| `SLACK_SHOW_TOOL_CALLS` | No | Show tool usage indicators (default: `true`) |
| `SLACK_SHOW_THINKING` | No | Show "Thinking..." status (default: `true`) |

### Command Line Options

```bash
hmm-slack --help

Options:
  --permission-mode [default|acceptEdits|bypassPermissions]
                        Permission mode for Claude Code tools
  --db-path PATH        Custom database path for conversation storage
  --debug               Enable debug logging
```

### Permission Modes

- **default**: Claude Code asks for confirmation on sensitive operations
- **acceptEdits**: Auto-approve file edits (good for code assistants)
- **bypassPermissions**: No prompts (use with caution in automation)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Slack Workspace                                            │
│    ├── DMs          ──┐                                     │
│    ├── @mentions    ──┼──→ Socket Mode ──→ SlackHMMBot     │
│    └── /hmm command ──┘                        │            │
│                                                │            │
│                                                ▼            │
│                              ClaudeAgentSDKConversationManager
│                                                │            │
│                                                ▼            │
│                              ┌─────────────────────────────┐│
│                              │ Claude Code SDK             ││
│                              │  - File operations          ││
│                              │  - Bash commands            ││
│                              │  - Git operations           ││
│                              │  - Web search               ││
│                              └─────────────────────────────┘│
│                                                │            │
│                              ┌─────────────────┴───────────┐│
│                              │ HMM Memory (DuckDB)         ││
│                              │  - Hierarchical compression ││
│                              │  - Per-channel memory       ││
│                              │  - Persistent across restarts│
│                              └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Memory Management

Each Slack channel/DM gets its own conversation with persistent memory:

- **Channel conversations**: Shared memory for all users in the channel
- **DM conversations**: Private memory per user
- **Thread conversations**: Separate memory per thread (optional)

Memory persists across bot restarts using DuckDB storage.

## Troubleshooting

### Bot doesn't respond

1. Check that the bot is running (`hmm-slack`)
2. Verify the bot is added to the channel (for @mentions)
3. Check environment variables are set correctly
4. Enable debug logging: `hmm-slack --debug`

### "Invalid token" errors

- Bot token should start with `xoxb-`
- App token should start with `xapp-`
- Regenerate tokens if they've been rotated

### Socket Mode connection issues

- Ensure Socket Mode is enabled in app settings
- Verify the app-level token has `connections:write` scope
- Check firewall allows outbound WebSocket connections

### Missing permissions

If the bot can't read/send messages:
1. Go to **OAuth & Permissions**
2. Add missing scopes
3. **Reinstall the app** (required after adding scopes)

## Production Deployment

### Running as a Service (systemd)

Create `/etc/systemd/system/hmm-slack.service`:

```ini
[Unit]
Description=HMM Slack Bot
After=network.target

[Service]
Type=simple
User=hmm
WorkingDirectory=/opt/hmm
Environment=SLACK_BOT_TOKEN=xoxb-your-token
Environment=SLACK_APP_TOKEN=xapp-your-token
ExecStart=/opt/hmm/venv/bin/hmm-slack --permission-mode=acceptEdits
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable hmm-slack
sudo systemctl start hmm-slack
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install hierarchical-memory-middleware[slack]

ENV SLACK_BOT_TOKEN=""
ENV SLACK_APP_TOKEN=""

CMD ["hmm-slack"]
```

Run:
```bash
docker run -d \
  -e SLACK_BOT_TOKEN=xoxb-your-token \
  -e SLACK_APP_TOKEN=xapp-your-token \
  -v hmm-data:/app/data \
  hmm-slack
```

## Security Considerations

1. **Token Storage**: Never commit tokens to git. Use environment variables or secrets management.
2. **Permission Mode**: Use `default` mode unless you trust all users in your workspace.
3. **Channel Access**: The bot can see messages in channels it's added to.
4. **File Access**: In `acceptEdits` or `bypassPermissions` mode, Claude can modify files on the host system.

## Further Reading

- [Slack Bolt for Python](https://github.com/slackapi/bolt-python)
- [Slack API Documentation](https://api.slack.com/docs)
- [HMM Documentation](../README.md)
