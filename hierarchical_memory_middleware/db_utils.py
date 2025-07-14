import logging
from contextlib import contextmanager

import duckdb
import time
import random


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection(
    db_path: str,
    max_attempts: int = 300,
    base_delay: float = 0.1,
    max_delay: int = 10,
    init_schema: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Create and yield a DuckDB connection, retrying if the database is locked.

    Args:
        db_path: Path to the DuckDB file (required)
        max_attempts: Maximum number of connection attempts before giving up
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        init_schema: Whether to initialize the database schema after connection
    """
    attempt = 0

    while True:
        attempt += 1
        try:
            logger.debug(
                f"Opening DuckDB connection (attempt {attempt}/{max_attempts})..."
            )
            connection = duckdb.connect(db_path)
            logger.debug("Successfully connected to DuckDB")
            break
        except duckdb.IOException as e:
            if attempt >= max_attempts:
                logger.error(
                    f"Failed to connect to DuckDB after {max_attempts} attempts: {str(e)}"
                )
                raise

            delay = min(
                base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1),
                max_delay,
            )
            logger.warning(
                f"Database locked, retrying in {delay:.2f} seconds (attempt {attempt}/{max_attempts})"
            )
            time.sleep(delay)

        except Exception as e:
            logger.exception(f"Failed to connect to DuckDB: {str(e)}")
            raise
    try:
        # Initialize schema if requested
        if init_schema:
            _init_schema(connection)
        yield connection
    finally:
        if connection:
            logger.debug("Closing DuckDB connection...")
            connection.close()
            logger.debug("DuckDB connection closed")


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Initialize database schema."""
    # Create sequence for node IDs
    conn.execute("CREATE SEQUENCE IF NOT EXISTS nodes_seq")
    
    # Create nodes table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY DEFAULT nextval('nodes_seq'),
            conversation_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sequence_number INTEGER NOT NULL,
            line_count INTEGER DEFAULT 1,

            -- Compression fields
            level INTEGER DEFAULT 0,
            summary TEXT,
            summary_metadata JSON,
            parent_summary_id INTEGER,

            -- Tracking fields
            tokens_used INTEGER,
            expandable BOOLEAN DEFAULT TRUE,

            -- AI components (for AI nodes)
            ai_components JSON,

            -- Semantic fields
            topics JSON,
            embedding FLOAT[],

            -- Relationship fields
            relates_to_node_id INTEGER,

        )
    """)

    # Create indexes for performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON nodes(conversation_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON nodes(level)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON nodes(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sequence ON nodes(conversation_id, sequence_number)")

    # Create conversations table for metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            total_nodes INTEGER DEFAULT 0,
            compression_stats JSON,
            current_goal TEXT,
            key_decisions JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    logger.debug("Database schema initialized")
