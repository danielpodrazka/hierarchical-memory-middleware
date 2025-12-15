#!/usr/bin/env python3
"""Minimal stdio MCP server for Claude Agent SDK integration.

This server is designed to be spawned as a subprocess by the Claude Agent SDK
and provides memory tools via stdio transport.

Usage:
    python -m hierarchical_memory_middleware.mcp_server.stdio_memory_server \
        --conversation-id <id> --db-path <path>
"""

import argparse
import http.server
import json
import logging
import os
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any

from fastmcp import FastMCP

# Suppress logging to stderr since stdio is used for communication
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/hmm_stdio_server.log")],
)
logger = logging.getLogger(__name__)


def create_memory_server(conversation_id: str, db_path: str) -> FastMCP:
    """Create a FastMCP server with memory tools.

    Args:
        conversation_id: The conversation ID to use for all operations
        db_path: Path to the DuckDB database

    Returns:
        Configured FastMCP server instance
    """
    # Import storage lazily to avoid startup overhead
    from ..storage import DuckDBStorage
    from ..models import CompressionLevel, NodeType

    # Initialize storage
    storage = DuckDBStorage(db_path)

    # Create FastMCP server
    mcp = FastMCP("memory-tools")

    @mcp.tool()
    async def expand_node(node_id: int) -> Dict[str, Any]:
        """Retrieve full content of a conversation node by its ID.

        Use this tool to see the complete original content of a compressed
        or summarized message from the conversation history.

        Args:
            node_id: The numeric node identifier

        Returns:
            Full node content including text, metadata, and timestamps
        """
        try:
            node = await storage.get_node(node_id, conversation_id)
            if not node:
                return {"error": f"Node {node_id} not found in this conversation"}

            return {
                "success": True,
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "content": node.content,
                "summary": node.summary,
                "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                "compression_level": node.level.name,
                "topics": node.topics or [],
                "line_count": node.line_count,
            }
        except Exception as e:
            logger.error(f"Error expanding node {node_id}: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def search_memory(
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """Search conversation history for specific content.

        Use this tool to find past messages or topics discussed earlier
        in the conversation.

        Args:
            query: Text to search for in the conversation history
            limit: Maximum number of results to return (default: 10)
            mode: Search mode - "keyword" (exact match), "semantic" (meaning-based),
                  or "hybrid" (combines both, default)

        Returns:
            List of matching messages with relevance scores
        """
        try:
            # Choose search method based on mode
            if mode == "semantic":
                results = await storage.search_nodes_semantic(
                    conversation_id=conversation_id,
                    query=query,
                    limit=limit,
                )
            elif mode == "hybrid":
                results = await storage.search_nodes_hybrid(
                    conversation_id=conversation_id,
                    query=query,
                    limit=limit,
                )
            else:  # keyword (default fallback)
                results = await storage.search_nodes(
                    conversation_id=conversation_id,
                    query=query,
                    limit=limit,
                )

            return {
                "success": True,
                "query": query,
                "mode": mode,
                "count": len(results),
                "results": [
                    {
                        "node_id": r.node.node_id,
                        "content_preview": r.node.content[:300]
                        if r.node.content
                        else "",
                        "summary": r.node.summary,
                        "relevance": r.relevance_score,
                        "match_type": r.match_type,
                        "compression_level": r.node.level.name,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {"error": str(e)}

    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text (roughly 4 chars per token for Claude models)."""
        if not text:
            return 0
        return len(text) // 4

    @mcp.tool()
    async def get_memory_stats() -> Dict[str, Any]:
        """Get statistics about the conversation memory.

        Use this tool to understand how much conversation history is stored
        and how it's organized across compression levels.

        Returns:
            Statistics including message counts, compression distribution, and token counts
        """
        try:
            nodes = await storage.get_conversation_nodes(conversation_id)

            # Group nodes by compression level
            nodes_by_level = {level: [] for level in CompressionLevel}
            for n in nodes:
                nodes_by_level[n.level].append(n)

            # Calculate token stats for each level
            def get_context_text(node):
                """Get the text that would actually be used in context."""
                if node.level == CompressionLevel.FULL:
                    return node.content
                elif node.summary:
                    return node.summary
                else:
                    return node.content

            level_counts = {}
            token_stats_by_level = {}
            total_current_tokens = 0
            total_original_tokens = 0

            for level in CompressionLevel:
                level_nodes = nodes_by_level[level]
                count = len(level_nodes)
                if count > 0:
                    level_counts[level.name] = count

                    current_tokens = sum(_estimate_tokens(get_context_text(n)) for n in level_nodes)
                    original_tokens = sum(_estimate_tokens(n.content) for n in level_nodes)

                    token_stats_by_level[level.name] = {
                        "count": count,
                        "current_tokens": current_tokens,
                        "original_tokens": original_tokens,
                        "compression_ratio": round(original_tokens / current_tokens, 2) if current_tokens > 0 else 0,
                    }
                    total_current_tokens += current_tokens
                    total_original_tokens += original_tokens

            # Count by type
            user_count = len([n for n in nodes if n.node_type == NodeType.USER])
            ai_count = len([n for n in nodes if n.node_type == NodeType.AI])

            return {
                "success": True,
                "conversation_id": conversation_id,
                "total_messages": len(nodes),
                "user_messages": user_count,
                "ai_messages": ai_count,
                "compression_levels": level_counts,
                "token_stats": {
                    "by_level": token_stats_by_level,
                    "total_current_tokens": total_current_tokens,
                    "total_original_tokens": total_original_tokens,
                    "overall_compression_ratio": round(total_original_tokens / total_current_tokens, 2) if total_current_tokens > 0 else 0,
                    "tokens_saved": total_original_tokens - total_current_tokens,
                    "tokens_saved_percent": round((1 - total_current_tokens / total_original_tokens) * 100, 1) if total_original_tokens > 0 else 0,
                },
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_recent_nodes(count: int = 10) -> Dict[str, Any]:
        """Get the most recent conversation messages.

        Use this tool to see recent messages in full detail.

        Args:
            count: Number of recent messages to retrieve (default: 10)

        Returns:
            List of recent messages with full content
        """
        try:
            nodes = await storage.get_conversation_nodes(conversation_id)

            # Filter to FULL level nodes and get most recent
            full_nodes = [n for n in nodes if n.level == CompressionLevel.FULL]
            recent = full_nodes[-count:] if len(full_nodes) > count else full_nodes

            return {
                "success": True,
                "count": len(recent),
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "node_type": n.node_type.value,
                        "content": n.content,
                        "timestamp": n.timestamp.isoformat() if n.timestamp else None,
                    }
                    for n in recent
                ],
            }
        except Exception as e:
            logger.error(f"Error getting recent nodes: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_system_prompt() -> Dict[str, Any]:
        """Get the current system prompt / scratchpad for this conversation.

        Use this tool to read your persistent notes and behavioral preferences
        that you've saved for this conversation session.

        Returns:
            The current system prompt content, or empty if not set
        """
        try:
            content = await storage.get_system_prompt(conversation_id)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "content": content or "",
                "has_content": content is not None and len(content) > 0,
            }
        except Exception as e:
            logger.error(f"Error getting system prompt: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def set_system_prompt(content: str) -> Dict[str, Any]:
        """Set or replace the system prompt / scratchpad for this conversation.

        Use this tool to save persistent notes, behavioral preferences,
        or context that should persist across the conversation. This acts
        as a scratchpad where you can write notes to yourself.

        Examples of what to store:
        - User preferences: "User prefers concise responses"
        - Project context: "Working on TypeScript strict mode project"
        - Behavioral notes: "Remember: don't explain basics, user is senior dev"
        - Task tracking: "Currently debugging auth flow in middleware"

        Args:
            content: The new system prompt content (replaces existing)

        Returns:
            Success status and the new content
        """
        try:
            await storage.set_system_prompt(conversation_id, content)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "content": content,
                "action": "replaced",
            }
        except Exception as e:
            logger.error(f"Error setting system prompt: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def append_to_system_prompt(content: str) -> Dict[str, Any]:
        """Append content to the system prompt / scratchpad.

        Use this tool to add notes without replacing existing content.
        Useful for incrementally building up context or adding new
        observations throughout the conversation.

        Args:
            content: The content to append (will be added on a new line)

        Returns:
            Success status and the updated full content
        """
        try:
            updated = await storage.append_system_prompt(conversation_id, content)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "content": updated,
                "action": "appended",
            }
        except Exception as e:
            logger.error(f"Error appending to system prompt: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def backfill_embeddings() -> Dict[str, Any]:
        """Generate embeddings for all messages that don't have them yet.

        This enables semantic search by creating vector embeddings for
        conversation history. Run this once to enable semantic search
        on existing conversations.

        Note: Requires the embeddings optional dependency to be installed:
        pip install 'hierarchical-memory-middleware[embeddings]'

        Returns:
            Number of messages updated with embeddings
        """
        try:
            updated_count = await storage.backfill_embeddings(conversation_id)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "embeddings_generated": updated_count,
                "message": f"Generated embeddings for {updated_count} messages",
            }
        except Exception as e:
            logger.error(f"Error backfilling embeddings: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def yield_to_human(reason: str = "Task complete") -> Dict[str, Any]:
        """Signal that you need human input or have completed the current task.

        Use this tool in agentic mode when:
        - You've finished a multi-step task and want human review
        - You need clarification or a decision from the user
        - You're blocked and need additional information
        - You've reached a natural stopping point

        The chat interface will detect this tool call and pause for human input
        instead of auto-continuing.

        Args:
            reason: Brief explanation of why you're yielding (e.g., "Task complete",
                   "Need clarification on X", "Blocked on Y")

        Returns:
            Confirmation that the yield signal was sent
        """
        # This tool is a signal - the actual pausing logic is in the chat loop
        return {
            "success": True,
            "action": "yield_to_human",
            "reason": reason,
            "message": f"Yielding to human: {reason}",
        }

    # Track active ngrok deployments for cleanup (stores PIDs for background processes)
    _active_deployments: Dict[str, Dict[str, Any]] = {}

    # Deployment state file for persistence across restarts
    _deployment_state_file = Path("/tmp/hmm_deployments.json")

    @mcp.tool()
    async def deploy_html(file_path: str) -> Dict[str, Any]:
        """Deploy an HTML file to a public URL using ngrok.

        Use this tool to make a local HTML file accessible via the internet.
        This is useful for:
        - Sharing reports with others who don't have local access
        - Testing web content from external devices
        - Creating shareable links for research deliverables

        The deployment runs as a background process and will remain active
        until stop_deployment is called or the system is restarted.

        If the same file is already deployed, returns the existing URL.

        Args:
            file_path: Absolute path to the HTML file to deploy

        Returns:
            Public URL where the file is accessible
        """
        try:
            # Validate file exists
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            if not path.is_file():
                return {"error": f"Path is not a file: {file_path}"}

            # Check if this file is already deployed
            # Load persisted deployments
            persisted = {}
            if _deployment_state_file.exists():
                try:
                    persisted = json.loads(_deployment_state_file.read_text())
                except:
                    pass

            # Check all deployments for this file
            all_deployments = {**persisted, **_active_deployments}
            for dep_id, dep in all_deployments.items():
                if dep.get("file_path") == str(path):
                    # Check if the processes are still running
                    server_alive = False
                    ngrok_alive = False

                    if "server_pid" in dep:
                        try:
                            os.kill(dep["server_pid"], 0)  # Signal 0 just checks if process exists
                            server_alive = True
                        except ProcessLookupError:
                            pass

                    if "ngrok_pid" in dep:
                        try:
                            os.kill(dep["ngrok_pid"], 0)
                            ngrok_alive = True
                        except ProcessLookupError:
                            pass

                    if server_alive and ngrok_alive:
                        # Deployment still active, return existing URL
                        return {
                            "success": True,
                            "public_url": dep["public_url"],
                            "file_path": str(path),
                            "deployment_id": dep_id,
                            "local_port": dep.get("port"),
                            "message": f"Already deployed! Existing URL: {dep['public_url']}",
                            "reused": True,
                        }
                    else:
                        # Clean up dead deployment entry
                        _active_deployments.pop(dep_id, None)
                        persisted.pop(dep_id, None)
                        # Update persisted state
                        try:
                            if persisted:
                                _deployment_state_file.write_text(json.dumps(persisted, indent=2))
                            elif _deployment_state_file.exists():
                                _deployment_state_file.unlink()
                        except:
                            pass

            # Find ngrok
            ngrok_path = shutil.which("ngrok") or os.path.expanduser("~/.local/bin/ngrok")
            if not os.path.exists(ngrok_path):
                return {"error": "ngrok not found. Install it with: curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar -xz -C ~/.local/bin/"}

            # Kill any existing ngrok processes (free tier only allows one)
            # This prevents stale tunnels from interfering with new deployments
            try:
                result = subprocess.run(
                    ["pkill", "-f", "ngrok http"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    time.sleep(1)  # Give ngrok time to clean up
                    logger.debug("Killed existing ngrok process")
            except Exception as e:
                logger.debug(f"No existing ngrok to kill: {e}")

            # Find an available port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]

            serve_dir = str(path.parent)
            serve_file = path.name

            # Create a standalone server script that runs independently
            server_script = f'''#!/usr/bin/env python3
import http.server
import socketserver
import os
import signal
import sys

os.chdir({repr(serve_dir)})

class SingleFileHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory={repr(serve_dir)}, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_GET(self):
        if self.path == "/" or self.path == "/{serve_file}":
            self.path = "/{serve_file}"
        super().do_GET()

# Handle termination gracefully
def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

with socketserver.TCPServer(("", {port}), SingleFileHandler) as httpd:
    httpd.serve_forever()
'''

            # Write the server script to a temp file
            script_path = Path(f"/tmp/hmm_server_{port}.py")
            script_path.write_text(server_script)
            script_path.chmod(0o755)

            # Start HTTP server as a completely independent background process
            # Using start_new_session=True to detach from parent process group
            server_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Give the server a moment to start
            time.sleep(0.5)

            # Start ngrok tunnel as independent background process
            # We use DEVNULL for stdout because reading from pipe can block/hang
            ngrok_process = subprocess.Popen(
                [ngrok_path, "http", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for ngrok to establish tunnel and get URL via API
            # This is more reliable than parsing log output
            public_url = None
            start_time = time.time()
            timeout = 15  # seconds

            while time.time() - start_time < timeout:
                time.sleep(1)  # Give ngrok time to start
                try:
                    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5) as resp:
                        data = json.loads(resp.read().decode())
                        for tunnel in data.get("tunnels", []):
                            if tunnel.get("public_url", "").startswith("https://"):
                                public_url = tunnel["public_url"]
                                break
                    if public_url:
                        break
                except Exception as e:
                    # ngrok API not ready yet, keep trying
                    logger.debug(f"Waiting for ngrok API: {e}")
                    continue

            if not public_url:
                # Clean up on failure
                try:
                    os.kill(ngrok_process.pid, signal.SIGTERM)
                    ngrok_process.wait(timeout=5)  # Reap zombie
                except:
                    pass
                try:
                    os.kill(server_process.pid, signal.SIGTERM)
                    server_process.wait(timeout=5)  # Reap zombie
                except:
                    pass
                return {"error": "Failed to establish ngrok tunnel. Make sure ngrok is authenticated: ngrok config add-authtoken YOUR_TOKEN"}

            # Store deployment info (PIDs for cleanup)
            deployment_id = f"deploy_{port}"
            deployment_info = {
                "port": port,
                "file_path": str(path),
                "public_url": public_url,
                "server_pid": server_process.pid,
                "ngrok_pid": ngrok_process.pid,
                "script_path": str(script_path),
            }
            _active_deployments[deployment_id] = deployment_info

            # Persist to file for recovery across restarts
            try:
                existing = {}
                if _deployment_state_file.exists():
                    existing = json.loads(_deployment_state_file.read_text())
                existing[deployment_id] = deployment_info
                _deployment_state_file.write_text(json.dumps(existing, indent=2))
            except Exception as e:
                logger.warning(f"Could not persist deployment state: {e}")

            return {
                "success": True,
                "public_url": public_url,
                "file_path": str(path),
                "deployment_id": deployment_id,
                "local_port": port,
                "message": f"Deployed! Public URL: {public_url}",
            }
        except Exception as e:
            logger.error(f"Error deploying HTML: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def stop_deployment(deployment_id: str = None) -> Dict[str, Any]:
        """Stop an active HTML deployment.

        Use this to clean up ngrok tunnels when you're done sharing.
        If no deployment_id is provided, stops all active deployments.

        Args:
            deployment_id: The deployment ID returned by deploy_html (optional)

        Returns:
            Status of the stopped deployment(s)
        """
        try:
            stopped = []

            # Load persisted deployments
            persisted = {}
            if _deployment_state_file.exists():
                try:
                    persisted = json.loads(_deployment_state_file.read_text())
                except:
                    pass

            # Merge with in-memory deployments
            all_deployments = {**persisted, **_active_deployments}

            def stop_one(dep_id: str, dep: Dict) -> bool:
                """Stop a single deployment. Returns True if stopped."""
                try:
                    # Kill ngrok process
                    if "ngrok_pid" in dep:
                        try:
                            os.kill(dep["ngrok_pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass  # Already dead
                    elif "ngrok_process" in dep:
                        dep["ngrok_process"].terminate()

                    # Kill server process
                    if "server_pid" in dep:
                        try:
                            os.kill(dep["server_pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass  # Already dead
                    elif "httpd" in dep:
                        dep["httpd"].shutdown()

                    # Clean up script file
                    if "script_path" in dep:
                        try:
                            Path(dep["script_path"]).unlink(missing_ok=True)
                        except:
                            pass

                    return True
                except Exception as e:
                    logger.warning(f"Error stopping deployment {dep_id}: {e}")
                    return False

            if deployment_id:
                # Stop specific deployment
                if deployment_id in all_deployments:
                    if stop_one(deployment_id, all_deployments[deployment_id]):
                        stopped.append(deployment_id)
                    _active_deployments.pop(deployment_id, None)
                    persisted.pop(deployment_id, None)
                else:
                    return {"error": f"Deployment not found: {deployment_id}"}
            else:
                # Stop all deployments
                for dep_id, dep in list(all_deployments.items()):
                    if stop_one(dep_id, dep):
                        stopped.append(dep_id)
                _active_deployments.clear()
                persisted.clear()

            # Update persisted state
            try:
                if persisted:
                    _deployment_state_file.write_text(json.dumps(persisted, indent=2))
                elif _deployment_state_file.exists():
                    _deployment_state_file.unlink()
            except:
                pass

            return {
                "success": True,
                "stopped": stopped,
                "message": f"Stopped {len(stopped)} deployment(s)",
            }
        except Exception as e:
            logger.error(f"Error stopping deployment: {e}")
            return {"error": str(e)}

    return mcp


def main():
    """Main entry point for the stdio memory server."""
    parser = argparse.ArgumentParser(description="Stdio MCP memory server")
    parser.add_argument("--conversation-id", required=True, help="Conversation ID")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    logger.info(
        f"Starting stdio memory server for conversation: {args.conversation_id}"
    )

    try:
        mcp = create_memory_server(args.conversation_id, args.db_path)
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
