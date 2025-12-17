"""HTML Context View - generates a browser-viewable HTML of what the AI sees.

This module creates an HTML file that shows the exact context being sent to the AI,
including the hierarchical memory compression levels, scratchpad, and system prompt.
"""

import html
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..models import CompressionLevel, NodeType, ConversationNode


def generate_context_html(
    conversation_id: str,
    system_prompt: str,
    nodes: List[ConversationNode],
    scratchpad: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    agentic_mode: bool = False,
    current_user_message: Optional[str] = None,
    token_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an HTML view of the AI's context.

    Args:
        conversation_id: The conversation ID
        system_prompt: The full system prompt being sent
        nodes: All conversation nodes
        scratchpad: The user's scratchpad/notes
        custom_instructions: Any custom instructions
        agentic_mode: Whether agentic mode is enabled
        current_user_message: The current user message being processed
        token_stats: Optional token usage statistics

    Returns:
        HTML string
    """
    # Group nodes by compression level
    full_nodes = [n for n in nodes if n.level == CompressionLevel.FULL]
    summary_nodes = [n for n in nodes if n.level == CompressionLevel.SUMMARY]
    meta_nodes = [n for n in nodes if n.level == CompressionLevel.META]
    archive_nodes = [n for n in nodes if n.level == CompressionLevel.ARCHIVE]

    # Calculate stats
    total_nodes = len(nodes)

    # Token estimation helper (roughly 4 chars per token for Claude models)
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return len(text) // 4

    def get_context_text(node: ConversationNode) -> str:
        """Get the text that would actually be used in context for this node."""
        if node.level == CompressionLevel.FULL:
            return node.content or ""
        elif node.summary:
            return node.summary
        else:
            return node.content or ""

    def estimate_archive_tokens(archive_nodes_list: List[ConversationNode]) -> int:
        """Estimate tokens for archive nodes as they're actually rendered in context."""
        if not archive_nodes_list:
            return 0
        batch_size = 20
        total_tokens = 0
        for i in range(0, len(archive_nodes_list), batch_size):
            batch = archive_nodes_list[i:i + batch_size]
            if not batch:
                continue
            all_topics = []
            total_lines = 0
            for node in batch:
                if node.topics:
                    all_topics.extend(node.topics)
                total_lines += node.line_count or 1
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            top_topics = sorted(topic_counts.keys(), key=lambda t: topic_counts[t], reverse=True)[:5]
            topics_str = ", ".join(top_topics) if top_topics else "general"
            batch_text = f"Nodes {batch[0].node_id}-{batch[-1].node_id}:({len(batch)} nodes, {total_lines} lines) [Topics: {topics_str}]"
            total_tokens += estimate_tokens(batch_text)
        return total_tokens

    # Calculate tokens per section
    archive_tokens = estimate_archive_tokens(archive_nodes)
    meta_tokens = sum(estimate_tokens(get_context_text(n)) for n in meta_nodes)
    summary_tokens = sum(estimate_tokens(get_context_text(n)) for n in summary_nodes)
    full_tokens = sum(estimate_tokens(get_context_text(n)) for n in full_nodes)
    scratchpad_tokens = estimate_tokens(scratchpad) if scratchpad else 0
    system_prompt_tokens = estimate_tokens(system_prompt)

    # Format token count for display
    def format_tokens(tokens: int) -> str:
        if tokens >= 1000:
            return f"~{tokens / 1000:.1f}k tokens"
        return f"~{tokens:,} tokens"

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Context View - {conversation_id[:8]}</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-tertiary: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent-archive: #e94560;
            --accent-meta: #f39c12;
            --accent-summary: #3498db;
            --accent-full: #2ecc71;
            --accent-user: #9b59b6;
            --accent-assistant: #1abc9c;
            --border-color: #2d3748;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color);
        }}

        .header h1 {{
            font-size: 1.5rem;
            margin-bottom: 10px;
            color: var(--accent-full);
        }}

        .header .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .header .meta span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .stats-bar {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}

        .stat-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .stat-badge.archive {{ background: var(--accent-archive); }}
        .stat-badge.meta {{ background: var(--accent-meta); color: #000; }}
        .stat-badge.summary {{ background: var(--accent-summary); }}
        .stat-badge.full {{ background: var(--accent-full); color: #000; }}

        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}

        .section-header {{
            padding: 15px 20px;
            font-weight: bold;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}

        .section-header:hover {{
            filter: brightness(1.1);
        }}

        .section-header .toggle {{
            font-size: 0.8rem;
            opacity: 0.7;
        }}

        .section.archive .section-header {{ background: var(--accent-archive); }}
        .section.meta .section-header {{ background: var(--accent-meta); color: #000; }}
        .section.summary .section-header {{ background: var(--accent-summary); }}
        .section.full .section-header {{ background: var(--accent-full); color: #000; }}
        .section.scratchpad .section-header {{ background: #8e44ad; }}
        .section.system .section-header {{ background: var(--bg-tertiary); }}
        .section.current .section-header {{ background: #e74c3c; }}

        .section-content {{
            padding: 15px 20px;
            max-height: 600px;
            overflow-y: auto;
        }}

        .section-content.collapsed {{
            display: none;
        }}

        .node {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 10px;
            border-left: 4px solid var(--border-color);
        }}

        .node.user {{ border-left-color: var(--accent-user); }}
        .node.assistant {{ border-left-color: var(--accent-assistant); }}

        .node-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .node-role {{
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }}

        .node-role.user {{ background: var(--accent-user); color: white; }}
        .node-role.assistant {{ background: var(--accent-assistant); color: white; }}

        .node-content {{
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.9rem;
        }}

        .node-tools {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px dashed var(--border-color);
            font-size: 0.8rem;
            color: var(--accent-meta);
        }}

        .hint {{
            color: var(--accent-archive);
            font-size: 0.75rem;
            margin-left: 10px;
        }}

        .archive-batch {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }}

        .archive-batch .topics {{
            color: var(--accent-meta);
        }}

        .meta-item {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }}

        .meta-item .topics {{
            font-weight: bold;
            color: var(--accent-meta);
        }}

        pre.system-prompt {{
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.85rem;
            line-height: 1.5;
            background: var(--bg-tertiary);
            padding: 15px;
            border-radius: 8px;
        }}

        .token-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            padding: 15px;
        }}

        .token-stat {{
            background: var(--bg-tertiary);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}

        .token-stat .value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent-full);
        }}

        .token-stat .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }}

        .refresh-toggle {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--bg-tertiary);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            cursor: pointer;
            user-select: none;
            border: 1px solid var(--border-color);
        }}

        .refresh-toggle:hover {{
            border-color: var(--accent-full);
        }}

        .refresh-toggle .toggle-switch {{
            width: 40px;
            height: 20px;
            background: var(--bg-primary);
            border-radius: 10px;
            position: relative;
            transition: background 0.2s;
        }}

        .refresh-toggle .toggle-switch.active {{
            background: var(--accent-full);
        }}

        .refresh-toggle .toggle-switch::after {{
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.2s;
        }}

        .refresh-toggle .toggle-switch.active::after {{
            transform: translateX(20px);
        }}

        .current-message {{
            background: rgba(231, 76, 60, 0.2);
            border: 2px solid #e74c3c;
            border-radius: 8px;
            padding: 15px;
            font-size: 0.95rem;
        }}

        @media (max-width: 768px) {{
            body {{ padding: 10px; }}
            .stats-bar {{ flex-direction: column; }}
            .header .meta {{ flex-direction: column; gap: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
            <div>
                <h1>🤖 AI Context View</h1>
                <div class="meta">
                    <span>📋 Conversation: <strong>{conversation_id[:12]}...</strong></span>
                    <span>🕐 Updated: <strong>{datetime.now().strftime("%H:%M:%S")}</strong></span>
                    <span>📊 Total Nodes: <strong>{total_nodes}</strong></span>
                    {"<span>🤖 <strong style='color: var(--accent-archive)'>AGENTIC MODE</strong></span>" if agentic_mode else ""}
                </div>
            </div>
            <div class="refresh-toggle" onclick="toggleAutoRefresh()">
                <span>🔄 Auto-refresh</span>
                <div class="toggle-switch" id="refreshToggle"></div>
            </div>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat-badge archive">📦 Archive: {len(archive_nodes)}</div>
        <div class="stat-badge meta">📑 Meta: {len(meta_nodes)}</div>
        <div class="stat-badge summary">📝 Summary: {len(summary_nodes)}</div>
        <div class="stat-badge full">💬 Full: {len(full_nodes)}</div>
    </div>
'''

    # Token stats section if available
    if token_stats:
        html_content += '''
    <div class="section">
        <div class="section-header" onclick="toggleSection(this)" style="background: var(--bg-tertiary)">
            <span>📊 Token Statistics</span>
            <span class="toggle">▼</span>
        </div>
        <div class="section-content">
            <div class="token-stats">
'''
        for key, value in token_stats.items():
            label = key.replace('_', ' ').title()
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"${value:.2f}" if 'cost' in key.lower() or 'usd' in key.lower() else f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            html_content += f'''
                <div class="token-stat">
                    <div class="value">{formatted_value}</div>
                    <div class="label">{label}</div>
                </div>
'''
        html_content += '''
            </div>
        </div>
    </div>
'''

    # Current user message (if being processed)
    if current_user_message:
        html_content += f'''
    <div class="section current">
        <div class="section-header" onclick="toggleSection(this)">
            <span>🔴 Current User Message (being processed)</span>
            <span class="toggle">▼</span>
        </div>
        <div class="section-content">
            <div class="current-message">{html.escape(current_user_message)}</div>
        </div>
    </div>
'''

    # Scratchpad section
    if scratchpad:
        html_content += f'''
    <div class="section scratchpad">
        <div class="section-header" onclick="toggleSection(this)">
            <span>📝 Scratchpad / Notes ({format_tokens(scratchpad_tokens)})</span>
            <span class="toggle">▼</span>
        </div>
        <div class="section-content">
            <div style="background: rgba(142, 68, 173, 0.2); border: 1px solid #8e44ad; border-radius: 6px; padding: 10px; margin-bottom: 15px; font-size: 0.85rem; color: var(--text-secondary);">
                ℹ️ This scratchpad is appended to the system prompt and persists across the conversation. The AI can read and update it using memory tools.
            </div>
            <pre class="system-prompt">{html.escape(scratchpad)}</pre>
        </div>
    </div>
'''

    # Archive section
    if archive_nodes:
        html_content += f'''
    <div class="section archive">
        <div class="section-header" onclick="toggleSection(this)">
            <span>📦 ARCHIVED CONTEXT ({format_tokens(archive_tokens)})</span>
            <span class="toggle">▼</span>
        </div>
        <div class="section-content collapsed">
'''
        # Group archives into batches like the actual context builder does
        batch_size = 20
        for i in range(0, len(archive_nodes), batch_size):
            batch = archive_nodes[i:i + batch_size]
            if not batch:
                continue

            first_node = batch[0]
            last_node = batch[-1]

            # Collect all topics
            all_topics = []
            total_lines = 0
            for node in batch:
                if node.topics:
                    all_topics.extend(node.topics)
                total_lines += node.line_count or 1

            # Get top 5 topics
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            top_topics = sorted(topic_counts.keys(), key=lambda t: topic_counts[t], reverse=True)[:5]
            topics_str = ", ".join(top_topics) if top_topics else "general"

            html_content += f'''
            <div class="archive-batch">
                Nodes {first_node.node_id}-{last_node.node_id}: ({len(batch)} nodes, {total_lines} lines)
                [Topics: <span class="topics">{html.escape(topics_str)}</span>]
            </div>
'''
        html_content += '''
        </div>
    </div>
'''

    # Meta summaries section
    if meta_nodes:
        html_content += f'''
    <div class="section meta">
        <div class="section-header" onclick="toggleSection(this)">
            <span>📑 META SUMMARIES ({format_tokens(meta_tokens)})</span>
            <span class="toggle">▼</span>
        </div>
        <div class="section-content collapsed">
'''
        for node in meta_nodes:
            topics = ", ".join(node.topics) if node.topics else "general"
            content = node.summary or (node.content[:300] if node.content else "")
            html_content += f'''
            <div class="meta-item">
                [<span class="topics">{html.escape(topics)}</span>] {html.escape(content)}
            </div>
'''
        html_content += '''
        </div>
    </div>
'''

    # Summary nodes section
    if summary_nodes:
        html_content += f'''
    <div class="section summary">
        <div class="section-header" onclick="toggleSection(this)">
            <span>📝 CONVERSATION SUMMARIES ({format_tokens(summary_tokens)})</span>
            <span class="toggle">▼</span>
        </div>
        <div class="section-content">
'''
        displayed_summaries = summary_nodes[-10:]  # Last 10 like the real context
        for node in displayed_summaries:
            role = "User" if node.node_type == NodeType.USER else "Assistant"
            role_class = "user" if node.node_type == NodeType.USER else "assistant"
            content = node.summary or (node.content[:200] if node.content else "")
            html_content += f'''
            <div class="node {role_class}">
                <div class="node-header">
                    <span class="node-role {role_class}">{role}</span>
                    <span>ID: {node.node_id}</span>
                </div>
                <div class="node-content">{html.escape(content)}</div>
            </div>
'''
        html_content += '''
        </div>
    </div>
'''

    # Full (recent) nodes section
    if full_nodes:
        html_content += f'''
    <div class="section full">
        <div class="section-header" onclick="toggleSection(this)">
            <span>💬 RECENT CONVERSATION ({format_tokens(full_tokens)})</span>
            <span class="toggle">▼</span>
        </div>
        <div class="section-content">
'''
        # Show last N full nodes (matching what AI sees)
        displayed_full = full_nodes[-10:]  # Approximate recent limit
        for node in displayed_full:
            role = "User" if node.node_type == NodeType.USER else "Assistant"
            role_class = "user" if node.node_type == NodeType.USER else "assistant"
            content = node.content or ""

            # Tool actions for AI nodes
            tool_html = ""
            if node.node_type == NodeType.AI and node.ai_components:
                tool_calls = node.ai_components.get("tool_calls", [])
                if tool_calls:
                    tool_names = [tc.get("tool_name", "unknown") for tc in tool_calls]
                    tool_html = f'<div class="node-tools">🔧 Tools: {html.escape(", ".join(tool_names))}</div>'

            html_content += f'''
            <div class="node {role_class}">
                <div class="node-header">
                    <span class="node-role {role_class}">{role}</span>
                    <span>ID: {node.node_id}</span>
                </div>
                <div class="node-content">{html.escape(content)}</div>
                {tool_html}
            </div>
'''
        html_content += '''
        </div>
    </div>
'''

    # Full prompt section (collapsed by default)
    html_content += f'''
    <div class="section system">
        <div class="section-header" onclick="toggleSection(this)">
            <span>📄 Full Prompt ({format_tokens(system_prompt_tokens)})</span>
            <span class="toggle">▶</span>
        </div>
        <div class="section-content collapsed">
            <p style="color: #888; font-size: 0.85em; margin: 0 0 10px 0;">
                ℹ️ This is sent as the <strong>system prompt</strong> to the Claude Agent SDK
            </p>
            <pre class="system-prompt">{html.escape(system_prompt)}</pre>
        </div>
    </div>

    <script>
        const STORAGE_KEY = 'hmm_context_view_state';

        // Load saved state
        function loadState() {{
            try {{
                return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
            }} catch (e) {{
                return {{}};
            }}
        }}

        // Save state
        function saveState(state) {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        }}

        // Toggle section and save state
        function toggleSection(header) {{
            const content = header.nextElementSibling;
            const toggle = header.querySelector('.toggle');
            const sectionClass = header.parentElement.className.split(' ')[1]; // Get section type (archive, meta, etc.)

            if (content.classList.contains('collapsed')) {{
                content.classList.remove('collapsed');
                toggle.textContent = '▼';
            }} else {{
                content.classList.add('collapsed');
                toggle.textContent = '▶';
            }}

            // Save collapsed state
            const state = loadState();
            state.sections = state.sections || {{}};
            state.sections[sectionClass] = content.classList.contains('collapsed');
            saveState(state);
        }}

        // Toggle auto-refresh
        function toggleAutoRefresh() {{
            const state = loadState();
            state.autoRefresh = !state.autoRefresh;
            saveState(state);
            updateRefreshToggle();

            if (state.autoRefresh) {{
                scheduleRefresh();
            }}
        }}

        // Update toggle UI
        function updateRefreshToggle() {{
            const state = loadState();
            const toggle = document.getElementById('refreshToggle');
            if (state.autoRefresh) {{
                toggle.classList.add('active');
            }} else {{
                toggle.classList.remove('active');
            }}
        }}

        // Schedule refresh
        let refreshTimeout = null;
        function scheduleRefresh() {{
            if (refreshTimeout) {{
                clearTimeout(refreshTimeout);
            }}
            const state = loadState();
            if (state.autoRefresh) {{
                refreshTimeout = setTimeout(() => {{
                    // Save scroll positions before refresh
                    saveScrollPositions();
                    location.reload();
                }}, 5000);
            }}
        }}

        // Save scroll positions
        function saveScrollPositions() {{
            const state = loadState();
            state.scrollPositions = {{}};

            // Save main scroll
            state.scrollPositions.main = window.scrollY;

            // Save each section's scroll position
            document.querySelectorAll('.section-content').forEach((content, idx) => {{
                if (content.scrollTop > 0) {{
                    state.scrollPositions['section_' + idx] = content.scrollTop;
                }}
            }});

            saveState(state);
        }}

        // Restore scroll positions
        function restoreScrollPositions() {{
            const state = loadState();
            if (!state.scrollPositions) return;

            // Restore main scroll
            if (state.scrollPositions.main) {{
                window.scrollTo(0, state.scrollPositions.main);
            }}

            // Restore section scrolls
            document.querySelectorAll('.section-content').forEach((content, idx) => {{
                const savedScroll = state.scrollPositions['section_' + idx];
                if (savedScroll) {{
                    content.scrollTop = savedScroll;
                }}
            }});
        }}

        // Restore section collapsed states
        function restoreSectionStates() {{
            const state = loadState();
            if (!state.sections) return;

            document.querySelectorAll('.section').forEach(section => {{
                const sectionClass = section.className.split(' ')[1];
                const isCollapsed = state.sections[sectionClass];
                const content = section.querySelector('.section-content');
                const toggle = section.querySelector('.toggle');

                if (isCollapsed !== undefined) {{
                    if (isCollapsed) {{
                        content.classList.add('collapsed');
                        toggle.textContent = '▶';
                    }} else {{
                        content.classList.remove('collapsed');
                        toggle.textContent = '▼';
                    }}
                }}
            }});
        }}

        // Initialize on load
        window.onload = function() {{
            // Set default auto-refresh to ON if not set
            const state = loadState();
            if (state.autoRefresh === undefined) {{
                state.autoRefresh = true;
                saveState(state);
            }}

            updateRefreshToggle();
            restoreSectionStates();
            restoreScrollPositions();

            // Start refresh timer if enabled
            scheduleRefresh();
        }};

        // Save scroll position periodically
        setInterval(saveScrollPositions, 1000);
    </script>
</body>
</html>
'''

    return html_content


def save_context_html(
    output_dir: str,
    conversation_id: str,
    system_prompt: str,
    nodes: List[ConversationNode],
    scratchpad: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    agentic_mode: bool = False,
    current_user_message: Optional[str] = None,
    token_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Save the HTML context view to a file.

    Args:
        output_dir: Directory to save the HTML file
        conversation_id: The conversation ID
        system_prompt: The full system prompt
        nodes: All conversation nodes
        scratchpad: User's scratchpad
        custom_instructions: Custom instructions
        agentic_mode: Whether agentic mode is enabled
        current_user_message: Current message being processed
        token_stats: Token usage statistics

    Returns:
        Path to the saved HTML file
    """
    os.makedirs(output_dir, exist_ok=True)

    html_content = generate_context_html(
        conversation_id=conversation_id,
        system_prompt=system_prompt,
        nodes=nodes,
        scratchpad=scratchpad,
        custom_instructions=custom_instructions,
        agentic_mode=agentic_mode,
        current_user_message=current_user_message,
        token_stats=token_stats,
    )

    output_path = os.path.join(output_dir, f"{conversation_id}_context.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
