"""Analyzer of SQL queries."""


import time
import psycopg
from datetime import datetime
from typing import Self, Any, Optional
from dataclasses import dataclass, field
from pg_feature_extractor import PostgresFeatureExtractor
from resource_monitor import ContainersResourceMonitor
from pg_feature_extractor import TableSize
from advanced_analyzer import (
    AdvancedQueryAnalyzer,
    AdvancedPlanMetrics
)


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

    # Advanced metrics
    advanced_metrics: Optional[dict[str, Any]] = None

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
                'is_read_only': self.is_read_only
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
        self.resource_monitor = None
        self.feature_extractor = PostgresFeatureExtractor(connection)

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
                    monitor = self._get_resource_monitor()
                    if monitor:
                        all_resources = (
                            monitor.get_all_container_resources()
                        )
                    else:
                        all_resources = None

                except Exception as resource_error:
                    print(
                        f"Warning: Could not get container "
                        f"resources: {resource_error}"
                    )
                    all_resources = None
            
            # Get historical performance data (with caching)
            historical_stats = None
            if include_historical:
                historical_stats = (
                    self._get_historical_query_stats_with_cache(
                        sql_query, limit=1
                    )
                )
                historical_stats = (
                    historical_stats[0] 
                    if historical_stats else None
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
  
            if (
                not execution_plan or 
                not isinstance(execution_plan, list)
            ):
                raise psycopg.Error("Invalid execution plan format")

            plan_data = (
                execution_plan[0].get("Plan", {}) 
                if execution_plan else {}
            )

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


            # Get system metrics for CPU prediction
            system_metrics = None
            if all_resources:
                system_metrics = all_resources.get('system', {})

            # Predict memory and CPU usage
            predicted_memory = self._predict_memory_bytes(
                plan_data, postgres_metrics
            )
            predicted_cpu = self._predict_cpu_seconds(
                plan_data, system_metrics
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
                actual_loops=plan_data.get("Actual Loops"),
                planning_time=plan_data.get("Planning Time"),
                execution_time=plan_data.get("Execution Time"),
                predicted_memory_bytes=predicted_memory,
                predicted_cpu_seconds=predicted_cpu,
                query_type=None,
                contains_join=False,
                contains_sort=False,
                contains_aggregate=False,
                is_read_only=True
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

    def _predict_memory_bytes(
        self: Self,
        plan_data: dict[str, Any],
        postgres_metrics: Optional[dict[str, Any]]
    ) -> Optional[int]:
        """Estimate memory usage based on execution plan."""

        try:
            base_memory = 1024 * 1024  # 1MB base
            
            plan_rows = plan_data.get("Plan Rows", 0)
            plan_width = plan_data.get("Plan Width", 0)

            # bytes for row processing
            row_memory = plan_rows * plan_width
            
            # Memory for operations
            operation_memory = 0
            node_type = plan_data.get("Node Type", "")
            
            if node_type == "Sort":  # 2x for sort overhead
                operation_memory = plan_rows * plan_width * 2
                
            elif node_type == "Hash":  # 3x for hash tables
                operation_memory = plan_rows * plan_width * 3
                
            elif node_type == "Aggregate":  # 1.5x for aggregation
                operation_memory = plan_rows * plan_width * 1.5
                
            total_memory = base_memory + row_memory + operation_memory
            
            if (
                postgres_metrics and 
                'available_memory' in postgres_metrics
            ):
                total_memory = min(
                    total_memory,
                    postgres_metrics['available_memory'] * 0.8
                )

            return int(total_memory)
            
        except (TypeError, ValueError):
            return None

    def _predict_cpu_seconds(
        self: Self,
        plan_data: dict[str, Any],
        system_metrics: Optional[dict[str, Any]]
    ) -> Optional[float]:
        """Estimate CPU time based on execution plan and system."""
        try:
            base_cpu = 0.001  # 1ms base overhead
            
            total_cost = plan_data.get("Total Cost", 0)
            
            # Convert PostgreSQL cost units to seconds
            cost_to_cpu_factor = 0.0001
            
            # Operation complexity
            complexity_factor = 1.0
            node_type = plan_data.get("Node Type", "")

            if node_type == "Seq Scan":
                complexity_factor = 1.0

            elif node_type == "Index Scan":
                complexity_factor = 0.7

            elif node_type == "Nested Loop":
                complexity_factor = 1.5

            elif node_type == "Hash Join":
                complexity_factor = 2.0

            elif node_type == "Sort":
                complexity_factor = 2.5

            elif node_type == "Aggregate":
                complexity_factor = 1.8

            # System load if available
            load_factor = 1.0
            if system_metrics:
                load_1min = None
                
                # Case 1: Direct load_average dictionary
                if (
                    'load_average' in system_metrics and 
                    isinstance(system_metrics['load_average'], dict)
                ):
                    load_1min = (
                        system_metrics['load_average'].get('1min')
                    )

                # Case 2: Direct load_1min value
                elif 'load_1min' in system_metrics:
                    load_1min = system_metrics['load_1min']

                # Case 3: System metrics might have cpu load info
                elif (
                    'cpu' in system_metrics and 
                    isinstance(system_metrics['cpu'], dict)
                ):
                    cpu_metrics = system_metrics['cpu']

                    if 'load_1min' in cpu_metrics:
                        load_1min = cpu_metrics['load_1min']

                    elif (
                        'load_average' in cpu_metrics 
                        and isinstance(
                            cpu_metrics['load_average'], dict
                        )
                    ):
                        load_1min = (
                            cpu_metrics['load_average'].get('1min')
                        )

                if (
                    load_1min is not None and 
                    isinstance(load_1min, (int, float)) and 
                    load_1min > 0
                ):
                    load_factor = max(1.0, load_1min * 0.5)
                    
            estimated_cpu = base_cpu + (
                total_cost * 
                cost_to_cpu_factor * 
                complexity_factor * 
                load_factor
            )
            
            return round(estimated_cpu, 4)
            
        except (TypeError, ValueError):
            return None

    def _add_resource_prediction_recommendations(
        self: Self,
        analysis_result: QueryAnalysisResult,
        recs: list[dict[str, Any]]
    ) -> None:
        """Recommendations based on predicted memory and CPU."""
        
        if analysis_result.predicted_memory_bytes:
            memory_mb = (
                analysis_result.predicted_memory_bytes
                / (1024 * 1024)
            )
            
            if memory_mb > 10:  # More than 10MB
                recs.append({
                    "type": "memory",
                    "priority": "MEDIUM",
                    "message": (
                        f"Query predicted to use "
                        f"{memory_mb:.1f}MB memory. "
                        f"Consider optimizing with smaller "
                        f"batches or indexes."
                    ),
                    "predicted_memory_mb": memory_mb
                })

            if memory_mb > 50:  # More than 50MB
                recs.append({
                    "type": "memory",
                    "priority": "HIGH",
                    "message": (
                        f"High memory usage predicted "
                        f"({memory_mb:.1f}MB). "
                        f"This may impact other queries. "
                        f"Consider increasing "
                        f"work_mem or optimizing query structure."
                    ),
                    "predicted_memory_mb": memory_mb
                })
        
        if analysis_result.predicted_cpu_seconds:
            cpu_ms = (
                analysis_result.predicted_cpu_seconds
                * 1000
            )

            if cpu_ms > 100:  # More than 100ms
                recs.append({
                    "type": "cpu",
                    "priority": "MEDIUM",
                    "message": (
                        f"Query predicted to use "
                        f"{cpu_ms:.0f}ms CPU time. "
                        f"Consider optimizing with "
                        f"better indexes or "
                        f"simplifying complex operations."
                    ),
                    "predicted_cpu_ms": cpu_ms
                })

            if cpu_ms > 500:  # More than 500ms
                recs.append({
                    "type": "cpu",
                    "priority": "HIGH",
                    "message": (
                        f"High CPU usage predicted "
                        f"({cpu_ms:.0f}ms). "
                        f"This may impact system performance. "
                        f"Consider running during off-peak hours "
                        f"or optimizing further."
                    ),
                    "predicted_cpu_ms": cpu_ms
                })

    def _get_resource_monitor(self: Self) -> ContainersResourceMonitor:
        """Lazy initialization of ContainersResourceMonitor."""
        
        if self.resource_monitor is None:
            try:
                self.resource_monitor = ContainersResourceMonitor()
            except Exception:
                self.resource_monitor = None
        return self.resource_monitor

    def _get_historical_query_stats(
            self: Self,
            sql_query: str,
            limit: int = 1
    ) -> Optional[list[dict[str, Any]]]:
        """Get historical performance data for similar queries."""

        try:
            normalized_pattern = self._normalize_query_pattern(
                sql_query
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
                    WHERE query ILIKE %s
                    ORDER BY total_exec_time DESC 
                    LIMIT %s
                """
                search_pattern = f"%{normalized_pattern}%"
                cur.execute(query, (search_pattern, limit))
                results = cur.fetchall()
                
                if results:
                    stats_list = []
                    for row in results:
                        stats_list.append({
                            "query": row[0],
                            "calls": row[1],
                            "total_exec_time": row[2],
                            "mean_exec_time": row[3],
                            "rows": row[4],
                            "shared_blks_hit": row[5],
                            "shared_blks_read": row[6],
                            "cache_hit_ratio": row[7]
                        })
                    return stats_list
                else:
                    return None

        except psycopg.Error:
            return None

    def _normalize_query_pattern(
        self: Self,
        sql_query: str
    ) -> str:
        """Create a pattern for matching queries."""

        cleaned_query = ' '.join(sql_query.split()).lower()  
        words = cleaned_query.split()

        if len(words) > 3:
            pattern = ' '.join(words[:4]) + '%'
        else:
            pattern = cleaned_query + '%'
        
        return pattern

    def _generate_recommendations(
            self: Self,
            analysis_result: QueryAnalysisResult,
            plan_data: dict[str, Any]
    ) -> None:
        """Main recommendation orchestrator."""

        recs = analysis_result.recommendations
        
        if plan_data.get("Node Type") == "Seq Scan":
            self._analyze_sequential_scan(
                plan_data, recs
            )
        
        self._analyze_cache_efficiency(
            analysis_result, recs
        )
        
        if plan_data.get("Node Type") == "Sort":
            self._analyze_sort_operation(
                analysis_result, recs
            )
        
        self._generate_cross_container_recommendations(
            analysis_result, recs
        )
        
        self._add_historical_context(
            analysis_result, recs
        )      
    
        self._generate_resource_aware_recommendations(
            analysis_result, recs
        )

        self._add_resource_prediction_recommendations(
            analysis_result, recs
        )

    def _analyze_sequential_scan(
            self: Self,
            plan_data: dict[str, Any],
            recs: list[dict[str, Any]]
    ) -> None:
        """Analyze sequential scan operations."""

        table_name = plan_data.get("Relation Name")
        if table_name:
            table_size: Optional[TableSize] = (
                self.feature_extractor.get_table_size(table_name)
            )

        # Check for filter condition
        filter_condition = plan_data.get("Filter")
        has_filter = (
            filter_condition and filter_condition != "true"
        )

        if has_filter and table_name:
            recs.append({
                "type": "index",
                "priority": (
                    "LOW" if 
                        not table_size or 
                        table_size.bytes_size < 10 * 1024 * 1024 
                    else "MEDIUM"
                ),
                "message": (
                    f"Sequential scan with filter "
                    f"on table '{table_name}'. "
                    f"Consider adding an index for "
                    f"condition: {filter_condition}"
                ),
                "table_name": table_name,
                "filter_condition": filter_condition
            })

        if (
            table_size and 
            hasattr(table_size, 'bytes_size') and 
            table_size.bytes_size > 1024 * 1024 * 100
        ):
                filter_condition = plan_data.get(
                    "Filter", "unknown condition"
                )

                table_size_bytes = getattr(
                    table_size, 'bytes_size', 0
                )
                pretty_size = getattr(
                    table_size, 'pretty_size', 'unknown size'
                )

                recs.append({
                    "type": "index",
                    "priority": "HIGH",
                    "message": (
                        f"Sequential scan on large table "
                        f"'{table_name}' ({pretty_size}). "
                        f"Consider adding an index for condition: "
                        f"{filter_condition}"
                    ),
                    "table_name": table_name,
                    "table_size_bytes": table_size_bytes
                })

    def _analyze_cache_efficiency(
            self: Self,
            analysis_result: QueryAnalysisResult, 
            recs: list[dict[str, Any]]
    ) -> None:
        """Analyze buffer cache efficiency."""
    
        hit = analysis_result.shared_buffers_hit
        read = analysis_result.shared_buffers_read

        total_buffers = hit + read

        if total_buffers > 0:
            cache_hit_ratio = (hit / total_buffers) * 100
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
        else:
            cache_hit_ratio = 100 

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

        monitor = self._get_resource_monitor()
        if not monitor:
            return

        container_resources = (
            monitor.get_all_container_resources()
        )
        if not container_resources:
            return

        app_metrics = container_resources.get(
            'application_container', {}
        )
        if not isinstance(app_metrics, dict):
            app_metrics = {}

        pg_metrics = container_resources.get(
            'postgres_container', {}
        )
        if not isinstance(pg_metrics, dict):
            pg_metrics = {}

        pg_internal = analysis_result.postgres_metrics or {}
        
        # PostgreSQL Container Memory Pressure
        pg_memory = pg_metrics.get('memory', {})

        if not isinstance(pg_memory, dict):
            pg_memory = {}

        pg_memory_percent = (
            pg_memory.get('percent_used', 0) 
            if isinstance(pg_memory, dict) else 0
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
        pg_cpu = pg_metrics.get('cpu', {})

        if not isinstance(pg_cpu, dict):
            pg_cpu = {}

        pg_cpu_percent = (
            pg_cpu.get('percent_used', 0) 
            if isinstance(pg_cpu, dict) else 0
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
        pg_disk_io = pg_metrics.get('disk_io', {})

        if not isinstance(pg_disk_io, dict):
            pg_disk_io = {}

        pg_disk_write = (
            pg_disk_io.get('write_bytes', 0) 
            if isinstance(pg_disk_io, dict) else 0
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
        app_memory = app_metrics.get('memory', {})

        if not isinstance(app_memory, dict):
            app_memory = {}

        app_memory_percent = (
            app_memory.get('percent_used', 0) 
            if isinstance(app_memory, dict) else 0
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

        # disk IOPS
        pg_disk_iops = (
            pg_disk_io.get('iops', 0) 
            if isinstance(pg_disk_io, dict) else 0
        )

        if pg_disk_iops > 1000:
            recs.append({
                "type": "io",
                "priority": "MEDIUM",
                "message": "High disk IOPS detected"
            })

    def _generate_resource_aware_recommendations(
            self: Self,
            analysis_result: QueryAnalysisResult,
            recs: list[dict[str, Any]]
    ) -> None:
        """Focuses on application container resources."""

        monitor = self._get_resource_monitor()
        if not monitor:
            return
        
        container_resources = (
            monitor.get_all_container_resources()
        )
        if not container_resources:
            return
            
        app_metrics = analysis_result.container_resources.get(
            'application_container', {}
        )
        if not isinstance(app_metrics, dict):
            app_metrics = {}

        # Memory metrics
        app_memory = app_metrics.get('memory', {})

        if not isinstance(app_memory, dict):
            app_memory = {}

        app_memory_percent = (
            app_memory.get('percent_used', 0) 
            if isinstance(app_memory, dict) else 0
        )

        # CPU metrics
        app_cpu = app_metrics.get('cpu', {})

        if not isinstance(app_cpu, dict):
            app_cpu = {}

        app_cpu_percent = (
            app_cpu.get('percent_used', 0) 
            if isinstance(app_cpu, dict) else 0
        )

        app_memory_pressure = app_memory_percent > 80
        app_high_cpu = app_cpu_percent > 70

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
            sql_query: str,
            limit: int = 1
    ) -> Optional[list[dict[str, Any]]]:
        """Get historical stats to avoid repeated database queries."""

        cache_key = (
            f"{self._normalize_query_pattern(sql_query)}_limit_{limit}"
        )
        current_time = time.time()

        time_diff = (
            current_time - 
            self._cache_timestamps.get(cache_key, 0)
        )

        if (
            cache_key in self._historical_cache and 
            time_diff < self.cache_ttl
        ):
            cached_data = self._historical_cache[cache_key]

            if isinstance(cached_data, list):
                return cached_data
            else:
                return [cached_data] if cached_data else None

        historical_stats_list = self._get_historical_query_stats(
            sql_query, limit=limit
        )

        if historical_stats_list:
            self._historical_cache[cache_key] = historical_stats_list
            self._cache_timestamps[cache_key] = current_time
        
        return historical_stats_list

    def _get_execution_plan_with_timeout(
            self: Self,
            sql_query: str,
            timeout_seconds: int = 10
    ) -> Optional[dict[str, Any]]:
        """Get execution plan with timeout protection."""
        
        if not self._is_valid_sql_query(sql_query):
            raise ValueError("Invalid SQL query")
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SET statement_timeout = {timeout_seconds * 1000};"
                )
                try:
                    cur.execute(
                        f"EXPLAIN (FORMAT JSON, COSTS, BUFFERS) {sql_query}"
                    )
                    result = cur.fetchone()
                    return result[0] if result else None
                finally:
                    cur.execute("RESET statement_timeout;")

        except psycopg.Error as e:
            if "statement timeout" in str(e):
                print(
                    f"Query explanation timed out "
                    f"after {timeout_seconds} seconds"
                )
                return None
            raise

    def _is_valid_sql_query(self: Self, sql_query: str) -> bool:
        """Basic validation to prevent SQL injection."""
        
        if not sql_query or len(sql_query.strip()) == 0:
            return False
        
        if len(sql_query) > 10000:
            return False
        
        suspicious_patterns = [
            ';', '--', '/*', '*/', 'xp_', 'exec(', 'union select',
            'insert into', 'update ', 'delete from', 'drop table',
            'create table', 'alter table'
        ]
        
        query_lower = sql_query.lower()
        
        # Allow only SELECT-like queries for analysis
        if not query_lower.strip().startswith((
            'select', 'with', 'explain'
        )):
            return False
        
        for pattern in suspicious_patterns:
            if pattern in query_lower:
                return False
        
        allowed_keywords = [
            'select', 'from', 'where', 'join', 'group by', 
            'order by', 'limit', 'offset', 'with'
        ]
        
        words = query_lower.split()
        unusual_words = [
            word for word in words if word 
            not in allowed_keywords and
            not word.replace('_', '').isalnum()
        ]
        
        if len(unusual_words) > 10:
            return False
        
        return True

    def _classify_query_type(
            self: Self,
            analysis_result: QueryAnalysisResult,
            sql_query: str
    ) -> None:
        """Classify query type and set relevant flags."""
        
        query_upper = sql_query.upper().strip()
        
        query_clean = ' '.join(
            line for line in query_upper.split('\n') 
            if not line.strip().startswith('--')
        ).strip()
        
        if query_clean.startswith(('WITH', 'SELECT')):
            analysis_result.query_type = "SELECT"
            analysis_result.is_read_only = True
            
        elif query_clean.startswith('INSERT'):
            analysis_result.query_type = "INSERT"
            analysis_result.is_read_only = False
            
        elif query_clean.startswith('UPDATE'):
            analysis_result.query_type = "UPDATE"
            analysis_result.is_read_only = False
            
        elif query_clean.startswith('DELETE'):
            analysis_result.query_type = "DELETE" 
            analysis_result.is_read_only = False
            
        elif query_clean.startswith(('CREATE', 'DROP', 'ALTER')):
            analysis_result.query_type = "DDL"
            analysis_result.is_read_only = False
            
        elif query_clean.startswith(('GRANT', 'REVOKE')):
            analysis_result.query_type = "DCL"
            analysis_result.is_read_only = False
            
        else:
            analysis_result.query_type = "OTHER"
            analysis_result.is_read_only = not any(
                keyword in query_clean 
                for keyword in [
                    'INSERT', 'UPDATE', 'DELETE',
                    'CREATE', 'DROP', 'ALTER'
                ]
            )
        
        if analysis_result.execution_plan:
            plan_str = str(analysis_result.execution_plan).upper()
            
            analysis_result.contains_join = any(
                join_pattern in plan_str 
                for join_pattern in [
                    'JOIN', 'NESTED LOOP', 'HASH JOIN',
                    'MERGE JOIN', 'NESTED LOOP INNER',
                    'NESTED LOOP LEFT', 'HASH INNER JOIN'
                ]
            )

            analysis_result.contains_sort = any(
                sort_pattern in plan_str 
                for sort_pattern in [
                    'SORT', 'ORDER BY', 'SORT KEY'
                ]
            )
            
            analysis_result.contains_aggregate = any(
                agg_pattern in plan_str 
                for agg_pattern in [
                    'AGGREGATE', 'GROUP', 'HASHAGG',
                    'GROUPAGG', 'GROUP BY', 
                    'COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('
                ]
            )
            
            if 'INDEX ONLY SCAN' in plan_str:
                analysis_result.recommendations.append({
                    "type": "performance",
                    "priority": "LOW", 
                    "message": "Query using index-only scan",
                    "optimization": "index_only_scan"
                })

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


class EnhancedQueryAnalyzer(QueryAnalyzer):
    """Enhanced query analyzer with advanced features."""
    
    def __init__(
        self: Self,
        connection: psycopg.Connection
    ) -> None:
        """Initialization with database connection."""

        super().__init__(connection)
        self.advanced_analyzer = AdvancedQueryAnalyzer(
            self.feature_extractor
        )
        self.advanced_metrics: Optional[AdvancedPlanMetrics] = None
    
    def analyze_query(
        self: Self,
        sql_query: str,
        include_resources: bool = True, 
        include_historical: bool = True
    ) -> QueryAnalysisResult:
        """Enhanced analyze_query with advanced metrics."""

        # Use the parent class's analyze_query method
        result = super().analyze_query(
            sql_query, include_resources, include_historical
        )
        
        # Perform advanced analysis on top of the basic analysis
        if result.execution_plan:
            self.advanced_metrics = (
                self.advanced_analyzer.analyze_advanced_metrics(
                    result.execution_plan
                )
            )

            # Add advanced recommendations to result
            if self.advanced_metrics:
                self._add_advanced_recommendations(
                    result, self.advanced_metrics
                )
                result.advanced_metrics = (
                    self.advanced_metrics.__dict__
                )

        return result
    
    def _add_advanced_recommendations(
        self: Self,
        result: QueryAnalysisResult, 
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Add advanced recommendations to analysis result."""

        advanced_recs = []

        advanced_recs.extend(metrics.join_recommendations)     
        advanced_recs.extend(metrics.aggregation_recommendations)
        advanced_recs.extend(metrics.partitioning_recommendations)
        advanced_recs.extend(metrics.index_recommendations)
        
        # Add to existing recommendations
        result.recommendations.extend(advanced_recs)
    
    def get_advanced_metrics(
        self: Self
    ) -> Optional[AdvancedPlanMetrics]:
        """Get advanced analysis metrics."""

        return self.advanced_metrics
