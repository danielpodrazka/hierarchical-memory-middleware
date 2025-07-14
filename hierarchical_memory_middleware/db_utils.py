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
) -> duckdb.DuckDBPyConnection:
    """Create and yield a DuckDB connection, retrying if the database is locked.

    Args:
        db_path: Path to the DuckDB file (required)
        max_attempts: Maximum number of connection attempts before giving up
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
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
        yield connection
    finally:
        if connection:
            logger.debug("Closing DuckDB connection...")
            connection.close()
            logger.debug("DuckDB connection closed")