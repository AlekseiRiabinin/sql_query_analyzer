import psycopg
import time
from datetime import datetime
from typing import Self, Any, Optional
from dataclasses import dataclass, field
from pg_feature_extractor import PostgresFeatureExtractor
from resource_monitor import ContainersResourceMonitor


@dataclass
class QueryAnalysisResult:
    """
    Complete analysis result with containers monitoring metrics.
    Contains query execution plan analysis and system resources.
    """
    # Core query analysis metrics
    query: str
    total_cost: float
    plan_rows: float
    node_type: str
    relation_name: Optional[str]
    shared_buffers_hit: int
    shared_buffers_read: int
    
    # Recommendations and metadata
    recommendations: list[dict[str, Any]] = (
        field(default_factory=list)
    )
    analysis_timestamp: datetime = (
        field(default_factory=datetime.now)
    )
    
    # Historical performance context
    historical_stats: Optional[dict[str, Any]] = None
    
    # Container resource metrics (cross-container monitoring)
    container_resources: Optional[dict[str, Any]] = None
    
    # PostgreSQL internal metrics
    postgres_metrics: Optional[dict[str, Any]] = None
    
    # Execution plan details
    execution_plan: Optional[dict[str, Any]] = None
    
    # Additional plan metadata
    plan_width: Optional[int] = None
    actual_rows: Optional[int] = None
    actual_loops: Optional[int] = None
    planning_time: Optional[float] = None
    execution_time: Optional[float] = None
    
    # I/O timing metrics (if available)
    io_read_time: Optional[float] = None
    io_write_time: Optional[float] = None
    
    # Resource usage predictions
    predicted_memory_bytes: Optional[int] = None
    predicted_cpu_seconds: Optional[float] = None
    
    # Query characteristics
    query_type: Optional[str] = None
    contains_join: bool = False
    contains_sort: bool = False
    contains_aggregate: bool = False
    
    # Security and safety flags
    is_read_only: bool = True
    requires_superuser: bool = False

    def to_dict(self: Self) -> dict[str, Any]:
        """Convert the analysis result to a dict."""
        return {
            'query': self.query,
            'performance_metrics': {
                'total_cost': self.total_cost,
                'plan_rows': self.plan_rows,
                'node_type': self.node_type,
                'relation_name': self.relation_name,
                'shared_buffers_hit': self.shared_buffers_hit,
                'shared_buffers_read': self.shared_buffers_read,
                'plan_width': self.plan_width,
                'planning_time': self.planning_time,
                'execution_time': self.execution_time
            },
            'resource_metrics': {
                'container_resources': self.container_resources,
                'postgres_metrics': self.postgres_metrics,
                'predicted_memory_bytes': self.predicted_memory_bytes,
                'predicted_cpu_seconds': self.predicted_cpu_seconds
            },
            'query_characteristics': {
                'query_type': self.query_type,
                'contains_join': self.contains_join,
                'contains_sort': self.contains_sort,
                'contains_aggregate': self.contains_aggregate,
                'is_read_only': self.is_read_only,
                'requires_superuser': self.requires_superuser
            },
            'recommendations': self.recommendations,
            'historical_context': self.historical_stats,
            'execution_plan': self.execution_plan,
            'timestamp': self.analysis_timestamp.isoformat()
        }


class QueryAnalyzer:
    """Analyzes SQL queries with EXPLAIN."""

    def __init__(
        self: Self,
        connection: psycopg.Connection
    ) -> None:
        """Initialization with database connection."""

        self.conn = connection
        self.feature_extractor = PostgresFeatureExtractor(connection)
        self.resource_monitor = ContainersResourceMonitor()

        # Initialize cache for historical data to avoid repeated queries
        self._historical_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self.cache_ttl = 300  # 5 minutes TTL for cache

    def analyze_query(
        self: Self,
        sql_query: str,
        include_resources: bool = True,
        include_historical: bool = True
    ) -> QueryAnalysisResult:
        """Analyze SQL query and provide recommendations."""
        
        # Input validation
        if not sql_query or not sql_query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(sql_query.strip()) > 10000:
            raise ValueError("Query is too long for analysis")
        
        try:
            # Get resource metrics
            all_resources = None
            if include_resources:
                try:
                    all_resources = (
                        self.resource_monitor.get_all_container_resources()
                    )
                except Exception as resource_error:
                    print(
                        f"Warning: Could not get "
                        f"container resources: {resource_error}"
                    )
            
            # Get historical performance data (with caching)
            historical_stats = None
            if include_historical:
                historical_stats = (
                    self._get_historical_query_stats_with_cache(
                        sql_query
                    )
                )
            
            # Get execution plan with timeout protection
            execution_plan = (
                self._get_execution_plan_with_timeout(
                    sql_query, timeout_seconds=10
                )
            )
            
            if not execution_plan:
                raise psycopg.Error(
                    "Failed to get execution plan for query"
                )
  
            plan_data = execution_plan[0].get("Plan", {})
            
            # Get PostgreSQL internal metrics if successful connection
            postgres_metrics = None
            try:
                postgres_metrics = (
                    self.feature_extractor.get_postgres_internal_metrics()
                )

            except Exception as metrics_error:
                print(
                    f"Warning: Could not get PostgreSQL "
                    f"internal metrics: {metrics_error}"
                )
            
            # Create analysis result with all collected data
            analysis_result = QueryAnalysisResult(
                query=sql_query,
                total_cost=plan_data.get("Total Cost", 0),
                plan_rows=plan_data.get("Plan Rows", 0),
                node_type=plan_data.get("Node Type", "Unknown"),
                relation_name=plan_data.get("Relation Name"),
                shared_buffers_hit=plan_data.get("Shared Hit Blocks", 0),
                shared_buffers_read=plan_data.get("Shared Read Blocks", 0),
                recommendations=[],
                historical_stats=historical_stats,
                execution_plan=execution_plan,
                container_resources=all_resources,
                postgres_metrics=postgres_metrics,
                plan_width=plan_data.get("Plan Width"),
                actual_rows=plan_data.get("Actual Rows"),
                actual_loops=plan_data.get("Actual Loops")
            )
            
            # Generate recommendations
            self._generate_recommendations(analysis_result, plan_data)
            
            # Add query type classification
            self._classify_query_type(analysis_result, sql_query)
            
            return analysis_result
            
        except psycopg.Error as e:
            error_msg = f"Failed to analyze query: {e}"
            if "canceling statement due to statement timeout" in str(e):
                error_msg = (
                    f"Query analysis timed out. "
                    f"The query may be too complex for analysis."
                )
            raise psycopg.Error(error_msg) from e
            
        except Exception as e:
            raise psycopg.Error(
                f"Unexpected error during query analysis: {e}"
            ) from e

    def _get_historical_query_stats(
            self: Self,
            sql_query: str
    ) -> Optional[dict[str, Any]]:
        """Get historical performance data for similar queries."""
        try:
            normalized_pattern = (  # simplified version
                self._normalize_query_pattern(sql_query)
            )
            
            with self.conn.cursor() as cur:
                query = """
                    SELECT 
                        query, calls, total_exec_time, mean_exec_time,
                        rows, shared_blks_hit, shared_blks_read,
                        (shared_blks_hit * 100.0 / NULLIF(
                            shared_blks_hit + shared_blks_read, 0)
                        ) as cache_hit_ratio
                    FROM pg_stat_statements 
                    WHERE query LIKE %s
                    ORDER BY mean_exec_time DESC 
                    LIMIT 1
                """
                cur.execute(query, (f"%{normalized_pattern}%",))
                result = cur.fetchone()
                
                if result:
                    return {
                        "query": result[0],
                        "calls": result[1],
                        "total_exec_time": result[2],
                        "mean_exec_time": result[3],
                        "rows": result[4],
                        "shared_blks_hit": result[5],
                        "shared_blks_read": result[6],
                        "cache_hit_ratio": result[7]
                    }

        except psycopg.Error:
            pass
        return None

    def _normalize_query_pattern(
            self: Self,
            sql_query: str
    ) -> str:
        """Create a simplified pattern for query matching."""

        normalized = ' '.join(sql_query.split()).lower()
        normalized = ' '.join(
            line for line 
            in normalized.split('--') 
            if line.strip()
        )
        return normalized[:100]

    def _generate_recommendations(
            self: Self,
            analysis_result: QueryAnalysisResult,
            plan_data: dict[str, Any]
    ) -> None:
        """Main recommendation orchestrator."""
        recs = analysis_result.recommendations
        
        # Plan-based analysis
        if plan_data.get("Node Type") == "Seq Scan":
            self._analyze_sequential_scan(
                analysis_result, plan_data, recs
            )
        
        self._analyze_cache_efficiency(analysis_result, recs)
        
        if plan_data.get("Node Type") == "Sort":
            self._analyze_sort_operation(
                analysis_result, plan_data, recs
            )
        
        # Cross-container resource analysis
        self._generate_cross_container_recommendations(
            analysis_result, recs
        )
        
        # Historical context
        self._add_historical_context(analysis_result, recs)      
      
        self._generate_resource_aware_recommendations(
            analysis_result, recs
        )
        
        self._add_historical_context(analysis_result, recs)

    def _analyze_sequential_scan(
            self: Self,
            plan_data: dict[str, Any],
            recs: list[dict[str, Any]]
    ) -> None:
        """Analyze sequential scan operations."""

        table_name = plan_data.get("Relation Name")
        if table_name:
            table_size = (
                self.feature_extractor.get_table_size(table_name)
            )
            if (
                table_size and 
                table_size.bytes > 1024 * 1024 * 100  # 100MB threshold
            ):
                filter_condition = plan_data.get(
                    "Filter", "unknown condition"
                )
                recs.append({
                    "type": "index",
                    "priority": "HIGH",
                    "message": (
                        f"Sequential scan on large table "
                        f"'{table_name}' ({table_size.pretty_size}). "
                        f"Consider adding an index for condition: "
                        f"{filter_condition}"
                    ),
                    "table_name": table_name,
                    "table_size_bytes": table_size.bytes
                })

    def _analyze_cache_efficiency(
            self: Self,
            analysis_result: QueryAnalysisResult, 
            recs: list[dict[str, Any]]
    ) -> None:
        """Analyze buffer cache efficiency."""
    
        total_buffers = (
            analysis_result.shared_buffers_hit + 
            analysis_result.shared_buffers_read
        )
        if total_buffers > 0:
            cache_hit_ratio = (
                analysis_result.shared_buffers_hit / total_buffers
            ) * 100
            if cache_hit_ratio < 90:
                recs.append({
                    "type": "configuration",
                    "priority": "MEDIUM",
                    "message": (
                        f"Low buffer cache hit ratio "
                        f"({cache_hit_ratio:.2f}%). "
                        f"Query is reading heavily from disk. "
                        f"Consider increasing shared_buffers "
                        f"or optimizing working set."
                    ),
                    "cache_hit_ratio": cache_hit_ratio
                })

    def _analyze_sort_operation(
            self: Self,
            analysis_result: QueryAnalysisResult, 
            recs: list[dict[str, Any]]
    ) -> None:
        """Analyze sort operations."""

        if analysis_result.plan_rows > 10000:
            recs.append({
                "type": "query",
                "priority": "MEDIUM",
                "message": (
                    f"Expensive Sort operation planned on "
                    f"~{analysis_result.plan_rows:,.0f} rows. "
                    f"Check if an index on ORDER BY columns "
                    f"could eliminate the sort."
                ),
                "estimated_rows": analysis_result.plan_rows
            })

    def _generate_cross_container_recommendations(
            self: Self,
            analysis_result: QueryAnalysisResult,
            recs: list[dict[str, Any]]
    ) -> None:
        """Generate recommendations from containers resources."""
        if not analysis_result.container_resources:
            return
            
        app_metrics = (
            analysis_result.container_resources.get(
                'application_container', {}
            )
        )
        pg_metrics = (
            analysis_result.container_resources.get(
                'postgres_container', {}
            )
        )
        pg_internal = analysis_result.postgres_metrics or {}
        
        # PostgreSQL Container Memory Pressure
        pg_memory_percent = (
            pg_metrics.get('memory', {}).get('percent_used', 0)
        )
        if pg_memory_percent > 85:
            recs.append({
                "type": "resource",
                "priority": "HIGH",
                "message": (
                    f"PostgreSQL container memory usage "
                    f"is critically high ({pg_memory_percent}%). "
                    f"High-cost queries may cause "
                    f"out-of-memory errors."
                ),
                "metric": "postgres_memory_usage",
                "value": pg_memory_percent,
                "threshold": 85
            })
        
        # PostgreSQL Container CPU Usage
        pg_cpu_percent = (
            pg_metrics.get('cpu', {}).get('percent_used', 0)
        )
        cpu_intensive_ops = [
            "Hash Join", "Sort", "Aggregate", "Hash", "WindowAgg"
        ]
        
        if (
            pg_cpu_percent > 90 and 
            analysis_result.node_type in cpu_intensive_ops
        ):
            recs.append({
                "type": "performance",
                "priority": "MEDIUM",
                "message": (
                    f"PostgreSQL CPU usage is very high "
                    f"({pg_cpu_percent}%). "
                    f"CPU-intensive {analysis_result.node_type} "
                    f"operations may be slower than expected."
                ),
                "metric": "postgres_cpu_usage",
                "value": pg_cpu_percent,
                "threshold": 90
            })

        # Disk I/O Pressure Detection
        pg_disk_write = (
            pg_metrics.get('disk_io', {}).get('write_bytes', 0)
        )
        if pg_disk_write > 50 * 1024 * 1024:  # 50MB/s write threshold
            recs.append({
                "type": "io",
                "priority": "MEDIUM",
                "message": (
                    f"PostgreSQL is getting high disk write I/O. "
                    f"Write operations may be slower and "
                    f"impact overall performance."
                ),
                "metric": "disk_write_throughput",
                "value": pg_disk_write,
                "unit": "bytes_per_second"
            })
        
        # Application Container Memory Pressure
        app_memory_percent = (
            app_metrics.get('memory', {}).get('percent_used', 0)
        )
        if (
            app_memory_percent > 90 and 
            analysis_result.plan_rows > 10000
        ):
            recs.append({
                "type": "resource",
                "priority": "MEDIUM",
                "message": (
                    f"Application container memory usage "
                    f"is high ({app_memory_percent}%). "
                    f"Large result sets may increase memory pressure."
                ),
                "metric": "app_memory_usage",
                "value": app_memory_percent,
                "threshold": 90
            })
        
        # Database Connection Pool Pressure
        active_connections = pg_internal.get(
            'active_connections', 0
        )
        max_connections = pg_internal.get(
            'max_connections', 100
        )
        
        connection_ratio = (
            (active_connections / max_connections * 100) 
            if max_connections > 0 else 0
        )
        if connection_ratio > 80:
            recs.append({
                "type": "connection",
                "priority": "MEDIUM", 
                "message": (
                    f"High database connection usage "
                    f"({active_connections}/{max_connections} "
                    f"connections, {connection_ratio:.1f}%). "
                    f"Consider optimizing connection pooling "
                    f"or increasing max_connections."
                ),
                "metric": "connection_pool_usage",
                "value": connection_ratio,
                "threshold": 80
            })

    def _generate_resource_aware_recommendations(
            self: Self,
            analysis_result: QueryAnalysisResult,
            recs: list[dict[str, Any]]
    ) -> None:
        """Focuses on application container resources."""
        if not analysis_result.container_resources:
            return
            
        app_metrics = (
            analysis_result.container_resources.get(
                'application_container', {}
            )
        )
        app_memory_pressure = (
            app_metrics.get('memory', {}).get('percent_used', 0) > 80
        )
        app_high_cpu = (
            app_metrics.get('cpu', {}).get('percent_used', 0) > 70
        )
        
        if (
            app_memory_pressure and 
            analysis_result.plan_rows > 10000
        ):
            recs.append({
                "type": "scheduling",
                "priority": "MEDIUM",
                "message": (
                    f"Application is under memory pressure "
                    f"and query expects large result set. "
                    f"Consider streaming results or "
                    f"processing in smaller batches."
                )
            })

        if (
            app_high_cpu and 
            analysis_result.total_cost > 5000
        ):
            recs.append({
                "type": "performance", 
                "priority": "LOW",
                "message": (
                    f"System is busy. "
                    f"This analysis may be slower than usual. "
                    f"Consider using cached results if available."
                )
            })

    def _add_historical_context(
        self: Self,
        analysis_result: QueryAnalysisResult,
        recs: list[dict[str, Any]]
    ) -> None:
        """Add historical performance context."""

        if not analysis_result.historical_stats:
            return

        historical = analysis_result.historical_stats

        for rec in recs:
            if (
                rec["type"] == "index" and 
                historical["mean_exec_time"] > 100
            ):
                rec["message"] += (
                    f"Historical data shows this "
                    f"query pattern averages "
                    f"{historical['mean_exec_time']:.2f}ms "
                    f"over {historical['calls']} executions."
                )
                rec["historical_mean_time"] = historical["mean_exec_time"]
                rec["historical_calls"] = historical["calls"]

    def _get_historical_query_stats_with_cache(
            self: Self,
            sql_query: str
    ) -> Optional[dict[str, Any]]:
        """Get historical stats to avoid repeated database queries."""

        cache_key = self._normalize_query_pattern(sql_query)
        current_time = time.time()
        
        time_diff = (
            current_time - 
            self._cache_timestamps.get(cache_key, 0)
        )
        # Check cache first
        if (
            cache_key in self._historical_cache and 
            time_diff < self.cache_ttl
        ):
            return self._historical_cache[cache_key]
        
        # Not in cache or expired, query the database
        historical_stats = self._get_historical_query_stats(sql_query)
        
        if historical_stats:
            self._historical_cache[cache_key] = historical_stats
            self._cache_timestamps[cache_key] = current_time
        
        return historical_stats

    def _get_execution_plan_with_timeout(
            self: Self,
            sql_query: str,
            timeout_seconds: int = 10
    ) -> Optional[dict[str, Any]]:
        """Get execution plan with timeout protection."""

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SET statement_timeout = {timeout_seconds * 1000};"
                )
                cur.execute(
                    f"EXPLAIN (FORMAT JSON, COSTS, BUFFERS) {sql_query}"
                )
                result = cur.fetchone()
                # Reset timeout
                cur.execute("RESET statement_timeout;")
                return result[0] if result else None

        except psycopg.Error as e:
            if "statement timeout" in str(e):
                print(
                    f"Query explanation timed out "
                    f"after {timeout_seconds} seconds"
                )
                return None
            raise

    def _classify_query_type(
            self: Self,
            analysis_result: QueryAnalysisResult,
            sql_query: str
    ) -> None:
        """Classify query type and set relevant flags."""

        query_upper = sql_query.upper().strip()  

        if query_upper.startswith('SELECT'):
            analysis_result.query_type = "SELECT"
            analysis_result.is_read_only = True

        elif query_upper.startswith('INSERT'):
            analysis_result.query_type = "INSERT"
            analysis_result.is_read_only = False

        elif query_upper.startswith('UPDATE'):
            analysis_result.query_type = "UPDATE" 
            analysis_result.is_read_only = False

        elif query_upper.startswith('DELETE'):
            analysis_result.query_type = "DELETE"
            analysis_result.is_read_only = False

        else:
            analysis_result.query_type = "OTHER"
        
        # Detect complex operations from the plan
        if analysis_result.execution_plan:
            plan_str = str(analysis_result.execution_plan).upper()
            analysis_result.contains_join = any(
                op in plan_str 
                for op in [
                    'JOIN', 'NESTED LOOP',
                    'HASH JOIN', 'MERGE JOIN'
                ]
            )
            analysis_result.contains_sort = 'SORT' in plan_str
            analysis_result.contains_aggregate = any(
                op in plan_str 
                for op in ['AGGREGATE', 'GROUP', 'HASHAGG']
            )

    def should_reject_query(
        self: Self,
        analysis_result: QueryAnalysisResult, 
        max_cost: float = 10000
    ) -> bool:
        """Determine if a query should be rejected."""

        if analysis_result.total_cost > max_cost:
            return True
            
        if analysis_result.container_resources:
            resources = analysis_result.container_resources

            if (
                analysis_result.total_cost > max_cost * 0.7 and 
                resources['memory']['percent_used'] > 90
            ):
                return True
                
        return False
