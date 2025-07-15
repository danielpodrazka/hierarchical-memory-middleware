"""Data models for hierarchical memory system."""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import json
from pydantic import BaseModel, Field, field_validator


class CompressionLevel(Enum):
    """Compression levels for conversation nodes."""

    FULL = 0  # Recent nodes - complete content
    SUMMARY = 1  # Older nodes - 1-sentence summaries
    META = 2  # Groups of summaries (20-40 nodes each group)
    ARCHIVE = 3  # Very old - high-level context


class NodeType(Enum):
    """Types of conversation nodes."""

    USER = "user"
    AI = "ai"


class ConversationNode(BaseModel):
    """A single conversation node storing user message and AI response."""

    node_id: int
    conversation_id: str
    node_type: NodeType
    content: str  # For USER: the message. For AI: all content combined
    timestamp: datetime
    sequence_number: int  # Order within conversation
    line_count: int  # Number of lines in the full content

    # Compression fields
    level: CompressionLevel = CompressionLevel.FULL
    summary: Optional[str] = None
    summary_metadata: Optional[Dict[str, Any]] = None
    parent_summary_node_id: Optional[int] = None

    # Node-specific fields
    tokens_used: Optional[int] = None
    expandable: bool = True

    # For AI nodes: structured breakdown of what it contains
    ai_components: Optional[Dict[str, Any]] = (
        None  # {"assistant_text": str, "tool_calls": [...], "tool_results": [...], "errors": [...]}
    )

    # Semantic fields for better retrieval
    topics: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None

    # Relationship fields
    relates_to_node_id: Optional[int] = None  # For follow-ups, corrections, etc.

    @field_validator("summary_metadata", "ai_components", mode="before")
    @classmethod
    def parse_json_fields(cls, v):
        """Parse JSON strings to dictionaries for metadata fields."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v

    @field_validator("topics", mode="before")
    @classmethod
    def parse_topics(cls, v):
        """Ensure topics is always a list, parsing from JSON if needed."""
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v


class ConversationState(BaseModel):
    """Overall state of a conversation with statistics."""

    conversation_id: str
    total_nodes: int
    compression_stats: Dict[CompressionLevel, int]
    current_goal: Optional[str] = None
    key_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: Optional[datetime] = None


class CompressionResult(BaseModel):
    """Result of compressing a conversation node."""

    original_node_id: int
    compressed_content: str
    compression_ratio: float
    topics_extracted: List[str]
    metadata: Dict[str, Any]


class SearchResult(BaseModel):
    """Result from searching conversation memory."""

    node: ConversationNode
    relevance_score: float
    match_type: str  # 'content', 'summary', 'topic'
    matched_text: str


class HierarchyThresholds(BaseModel):
    """Thresholds for different compression levels in the advanced hierarchy system."""
    
    # Number of nodes that trigger compression to next level
    summary_threshold: int = 10  # Recent nodes before compression starts
    meta_threshold: int = 50     # Summary nodes before META grouping
    archive_threshold: int = 200  # META groups before ARCHIVE
    
    # Group sizes for META level compression
    meta_group_size: int = 20    # Minimum nodes per META group
    meta_group_max: int = 40     # Maximum nodes per META group


class MetaGroup(BaseModel):
    """Represents a group of nodes compressed into META level."""
    
    start_node_id: int
    end_node_id: int
    start_sequence: int
    end_sequence: int
    node_count: int
    total_lines: int
    main_topics: List[str] = Field(default_factory=list)
    summary: str
    timestamp_range: Tuple[datetime, datetime]
    compression_level: CompressionLevel = CompressionLevel.META

    @field_validator("timestamp_range", mode="before")
    @classmethod
    def parse_timestamp_range(cls, v):
        """Ensure timestamp_range is properly parsed."""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (v[0], v[1])
        return v
