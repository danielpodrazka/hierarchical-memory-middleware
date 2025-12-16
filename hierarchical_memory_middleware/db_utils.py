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

    # Migration 4: Fix token_usage table to have auto-incrementing id
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]

        if 'token_usage' in table_names:
            # Check if table has DEFAULT nextval by looking at the DDL
            ddl = conn.execute(
                "SELECT sql FROM duckdb_tables() WHERE table_name = 'token_usage'"
            ).fetchone()
            has_default = ddl and 'nextval' in ddl[0].lower()

            if not has_default:
                # Table exists but doesn't have DEFAULT clause - recreate it
                logger.debug("Recreating token_usage table with auto-increment...")
                # Get existing data
                existing_data = conn.execute("SELECT * FROM token_usage").fetchall()
                # Drop old table and sequence
                conn.execute("DROP TABLE token_usage")
                conn.execute("DROP SEQUENCE IF EXISTS token_usage_seq")
                # Create sequence and new table
                conn.execute("CREATE SEQUENCE token_usage_seq")
                conn.execute("""
                    CREATE TABLE token_usage (
                        id INTEGER PRIMARY KEY DEFAULT nextval('token_usage_seq'),
                        conversation_id TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        cache_read_tokens INTEGER,
                        cache_creation_tokens INTEGER,
                        total_tokens INTEGER,
                        cost_usd FLOAT,
                        duration_ms INTEGER,
                        model TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    )
                """)
                # Restore data if any existed
                if existing_data:
                    for row in existing_data:
                        conn.execute("""
                            INSERT INTO token_usage (id, conversation_id, timestamp,
                                input_tokens, output_tokens, cache_read_tokens,
                                cache_creation_tokens, total_tokens, cost_usd, duration_ms, model)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, row)
                    # Update sequence to be after max id
                    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM token_usage").fetchone()[0]
                    for _ in range(max_id):
                        conn.execute("SELECT nextval('token_usage_seq')")
                logger.debug("token_usage table recreated with auto-increment")
            else:
                logger.debug("token_usage table already has auto-increment")
    except Exception as e:
        logger.debug(f"token_usage migration failed: {e}")
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

    # Create token usage table for tracking API costs
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS token_usage_seq;
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY DEFAULT nextval('token_usage_seq'),
            conversation_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_creation_tokens INTEGER,
            total_tokens INTEGER,
            cost_usd FLOAT,
            duration_ms INTEGER,
            model TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_token_usage_conversation ON token_usage(conversation_id)"
    )

    # Run migrations
    _run_migrations(conn)

    logger.debug("Database schema initialized")
