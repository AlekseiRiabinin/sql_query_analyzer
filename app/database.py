import os
import psycopg
from dotenv import load_dotenv


load_dotenv()


def get_db() -> psycopg.Connection:
    """Create a connection to the PostgreSQL database."""

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL environment variable is not set"
        )

    conn = psycopg.connect(database_url)
    return conn
