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


def _run_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    """Run database migrations to update existing databases."""
    logger.debug("Running database migrations...")

    # Migration 1: Add name column to conversations table
    try:
        # Check if conversations table exists first
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        if 'conversations' not in table_names:
            logger.debug("Conversations table doesn't exist yet, skipping migration")
            return
        
        # Try to select the name column to see if it exists
        try:
            conn.execute("SELECT name FROM conversations LIMIT 1").fetchall()
            logger.debug("Name column already exists")
        except Exception:
            # Column doesn't exist, add it
            logger.debug("Adding name column to conversations table...")
            conn.execute("ALTER TABLE conversations ADD COLUMN name TEXT")
            # Add unique constraint separately (some databases don't support it in ADD COLUMN)
            try:
                conn.execute("ALTER TABLE conversations ADD CONSTRAINT unique_name UNIQUE (name)")
                logger.debug("Name column added with unique constraint")
            except Exception:
                # If unique constraint fails, just log it but continue
                logger.debug("Name column added (unique constraint failed, continuing...)")
        
    except Exception as e:
        logger.debug(f"Migration failed: {e}")
        # This is expected for new databases where the table doesn't exist yet
        pass

    # Migration 2: Ensure unique constraint exists on name column
    try:
        # Check if conversations table exists and has the name column
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]

        if 'conversations' in table_names:
            try:
                # Try to add unique constraint if it doesn't exist
                conn.execute("ALTER TABLE conversations ADD CONSTRAINT unique_name UNIQUE (name)")
                logger.debug("Unique constraint added to name column")
            except Exception:
                # Constraint might already exist, that's fine
                logger.debug("Unique constraint already exists or couldn't be added")
    except Exception as e:
        logger.debug(f"Unique constraint migration failed: {e}")
        pass

    # Migration 3: Add system_prompt column to conversations table
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]

        if 'conversations' in table_names:
            try:
                conn.execute("SELECT system_prompt FROM conversations LIMIT 1").fetchall()
                logger.debug("system_prompt column already exists")
            except Exception:
                # Column doesn't exist, add it
                logger.debug("Adding system_prompt column to conversations table...")
                conn.execute("ALTER TABLE conversations ADD COLUMN system_prompt TEXT")
                logger.debug("system_prompt column added")
    except Exception as e:
        logger.debug(f"system_prompt migration failed: {e}")
        pass

    logger.debug("Database migrations completed")


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Initialize database schema."""
    # Create nodes table with composite primary key
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id INTEGER NOT NULL,
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
            parent_summary_node_id INTEGER,

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

            -- Composite primary key
            PRIMARY KEY (node_id, conversation_id)
        )
    """)

    # Create indexes for performance
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation ON nodes(conversation_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON nodes(level)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON nodes(timestamp)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sequence ON nodes(conversation_id, sequence_number)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_id ON nodes(conversation_id, node_id)"
    )

    # Create conversations table for metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            name TEXT,
            total_nodes INTEGER DEFAULT 0,
            compression_stats JSON,
            current_goal TEXT,
            key_decisions JSON,
            system_prompt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Run migrations
    _run_migrations(conn)
    
    logger.debug("Database schema initialized")
