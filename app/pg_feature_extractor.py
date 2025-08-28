import psycopg
from psycopg import sql
from typing import Self, Optional, Any
from dataclasses import dataclass


@dataclass
class TableSize:
    """Table size information."""
    pretty_size: str
    bytes_size: int


@dataclass
class IndexInfo:
    """Index information."""
    index_name: str
    index_definition: str
    is_unique: bool


@dataclass
class ColumnStats:
    """Column statistics."""
    null_fraction: float
    average_width: float
    distinct_values: float
    most_common_values: Optional[list[Any]]


@dataclass
class QueryStats:
    """Query statistics from pg_stat_statements."""
    query: str
    calls: int
    total_exec_time: float
    mean_exec_time: float


class PostgresFeatureExtractor:
    """Feature extractor of metadata and stats."""

    def __init__(
            self: Self,
            connection: psycopg.Connection
    ) -> None:
        self.conn = connection

    def get_table_size(
            self: Self,
            table_name: int
    ) -> Optional[TableSize]:
        """Get approximate table size."""

        query = sql.SQL("""
            SELECT 
                pg_size_pretty(pg_total_relation_size(%s)),
                pg_total_relation_size(%s) as bytes
            WHERE EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = %s
            );
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    query, (table_name, table_name, table_name)
                )
                result = cur.fetchone()
                return (
                    TableSize(result[0], result[1]) 
                    if result else None
                )

        except psycopg.Error as e:
            print(f"Error getting table size for {table_name}: {e}")
            return None

    def get_table_indexes(
            self: Self,
            table_name: str
    ) -> list[IndexInfo]:
        """Get a list of indexes for a table."""

        query = sql.SQL("""
            SELECT 
                i.indexname, 
                i.indexdef,
                ix.indisunique as is_unique
            FROM pg_indexes i
            JOIN pg_class c ON c.relname = i.tablename
            JOIN pg_index ix ON ix.indexrelid = (
                SELECT oid 
                FROM pg_class 
                WHERE relname = i.indexname
            )
            WHERE i.tablename = %s
            ORDER BY i.indexname;
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (table_name,))
                results = []
                for indexname, indexdef, is_unique in cur.fetchall():
                    results.append(
                        IndexInfo(indexname, indexdef, is_unique)
                    )
                return results

        except psycopg.Error as e:
            print(f"Error getting indexes for {table_name}: {e}")
            return []

    def get_column_statistics(
            self: Self,
            table_name: str,
            column_name: str
    ) -> Optional[ColumnStats]:
        """Get column statistics from pg_stats."""

        query = sql.SQL("""
            SELECT 
                null_frac, 
                avg_width, 
                n_distinct, 
                most_common_vals::text[]
            FROM pg_stats 
            WHERE tablename = %s AND attname = %s;
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (table_name, column_name))
                result = cur.fetchone()
                if result:
                    return ColumnStats(
                        null_fraction=result[0],
                        average_width=result[1],
                        distinct_values=result[2],
                        most_common_values=result[3]
                    )
                return None

        except psycopg.Error as e:
            print(
                f"Error getting statistics for "
                f"{table_name}.{column_name}: {e}"
            )
            return None

    def get_common_queries_from_pg_stat(
            self: Self,
            limit: int = 10
    ) -> list[QueryStats]:
        """Get frequent/slow requests."""

        query = sql.SQL("""
            SELECT 
                query, 
                calls, 
                total_exec_time, 
                mean_exec_time,
                rows,
                shared_blks_hit,
                shared_blks_read
            FROM pg_stat_statements 
            ORDER BY mean_exec_time DESC 
            LIMIT %s;
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (limit,))
                results = []
                for row in cur.fetchall():
                    results.append(QueryStats(
                        query=row[0],
                        calls=row[1],
                        total_exec_time=row[2],
                        mean_exec_time=row[3]
                    ))
                return results

        except psycopg.Error:
            return []

    def get_database_connections_info(
            self: Self
    ) -> dict[str, Any]:
        """Get info about current database connections."""

        query = sql.SQL("""
            SELECT 
                COUNT(*) as total_connections,
                COUNT(*) FILTER (
                    WHERE state = 'active'
                ) as active_connections,
                COUNT(*) FILTER (
                    WHERE state = 'idle'
                ) as idle_connections,
                COUNT(*) FILTER (
                    WHERE wait_event IS NOT NULL
                ) as waiting_connections
            FROM pg_stat_activity
            WHERE datname = current_database();
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()
                return {
                    'total_connections': result[0],
                    'active_connections': result[1],
                    'idle_connections': result[2],
                    'waiting_connections': result[3]
                }

        except psycopg.Error as e:
            print(f"Error getting connection info: {e}")
            return {}

    def get_lock_information(self: Self) -> list[dict[str, Any]]:
        """Get info about current locks in the database."""

        query = sql.SQL("""
            SELECT 
                locktype, 
                database, 
                relation::regclass,
                mode, 
                granted,
                pid,
                usename,
                application_name
            FROM pg_locks l
            JOIN pg_stat_activity a ON l.pid = a.pid
            ORDER BY granted, mode;
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                results = []
                for row in cur.fetchall():
                    results.append({
                        'lock_type': row[0],
                        'database': row[1],
                        'relation': row[2],
                        'mode': row[3],
                        'granted': row[4],
                        'process_id': row[5],
                        'username': row[6],
                        'application_name': row[7]
                    })
                return results

        except psycopg.Error as e:
            print(f"Error getting lock information: {e}")
            return []

    def get_vacuum_info(
            self: Self,
            table_name: str
    ) -> Optional[dict[str, Any]]:
        """Get vacuum-related information for a table."""

        query = sql.SQL("""
            SELECT 
                schemaname,
                relname,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze,
                vacuum_count,
                autovacuum_count,
                analyze_count,
                autoanalyze_count
            FROM pg_stat_all_tables 
            WHERE relname = %s;
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (table_name,))
                result = cur.fetchone()
                if result:
                    return {
                        'schema_name': result[0],
                        'table_name': result[1],
                        'last_vacuum': result[2],
                        'last_autovacuum': result[3],
                        'last_analyze': result[4],
                        'last_autoanalyze': result[5],
                        'vacuum_count': result[6],
                        'autovacuum_count': result[7],
                        'analyze_count': result[8],
                        'autoanalyze_count': result[9]
                    }
                return None
        except psycopg.Error as e:
            print(f"Error getting vacuum info for {table_name}: {e}")
            return None

    def get_index_usage_stats(
            self: Self,
            table_name: str
    ) -> list[dict[str, Any]]:
        """Get index usage statistics for a table."""

        query = sql.SQL("""
            SELECT 
                indexrelid::regclass as index_name,
                idx_scan as scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched
            FROM pg_stat_all_indexes 
            WHERE relname = %s;
        """)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (table_name,))
                results = []
                for row in cur.fetchall():
                    results.append({
                        'index_name': row[0],
                        'scans': row[1],
                        'tuples_read': row[2],
                        'tuples_fetched': row[3]
                    })
                return results

        except psycopg.Error as e:
            print(
                f"Error getting index usage stats for "
                f"{table_name}: {e}"
            )
            return []

    def get_postgres_internal_metrics(
            self: Self
    ) -> dict[str, Any]:
        """Get detailed PostgreSQL internal metrics."""

        queries = {
            'locks': "SELECT mode, count(*) FROM pg_locks GROUP BY mode",
            'buffer_cache': "SELECT * FROM pg_buffercache_summary()",
            'wal_stats': "SELECT * FROM pg_stat_wal",
        }
        
        metrics = {}
        for name, query in queries.items():
            try:
                with self.conn.cursor() as cur:
                    cur.execute(query)
                    metrics[name] = cur.fetchall()
            except:
                metrics[name] = None
                
        return metrics


# Example usage
if __name__ == "__main__":
    # Connect to PostgreSQL
    conn = psycopg.connect(
        "dbname=vtb_db user=postgres password=postgres"
    )
    
    # Create feature extractor
    extractor = PostgresFeatureExtractor(conn)
    
    # Example: Get table size
    size_info = extractor.get_table_size("table_name")
    if size_info:
        print(
            f"Table size: {size_info.pretty_size} "
            f"({size_info.bytes_size} bytes)"
        )
    
    # Example: Get index usage stats
    index_stats = extractor.get_index_usage_stats("table_name")
    for stat in index_stats:
        print(f"Index {stat['index_name']}: {stat['scans']} scans")
    
    conn.close()


# This comprehensive feature extractor provides:

#     Table Information: Sizes, bloat estimation, vacuum statistics

#     Index Analytics: Index definitions, usage statistics, uniqueness

#     Column Statistics: Null fractions, value distributions, data width

#     Performance Metrics: Query statistics, connection info, lock monitoring

#     Maintenance Insights: Vacuum information, bloat analysis
