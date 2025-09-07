"""Analyzer of SQL queries."""


import re
import time
import json
import psycopg
from datetime import datetime
from typing import Self, Any, Optional
from dataclasses import dataclass, field, asdict
from pg_feature_extractor import PostgresFeatureExtractor
from resource_monitor import ContainersResourceMonitor
from constants import Defaults
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
        result_dict = {
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

        if self.advanced_metrics:
            if hasattr(self.advanced_metrics, '__dict__'):
                result_dict['advanced_metrics'] = (
                    self.advanced_metrics.__dict__
                )
            else:
                result_dict['advanced_metrics'] = (
                    self.advanced_metrics
                )

        return result_dict

class QueryAnalyzer:
    """Analyzes SQL queries with EXPLAIN."""

    def __init__(
        self: Self,
        connection: psycopg.Connection,
        query_length_limit: Optional[int] = None,
        base_memory_bytes: Optional[int] = None,
        base_cpu_seconds: Optional[float] = None,
        cost_to_cpu_factor: Optional[float] = None,
        memory_threshold_medium: Optional[int] = None,
        memory_threshold_high: Optional[int] = None,
        cpu_threshold_medium: Optional[int] = None,
        cpu_threshold_high: Optional[int] = None,
        large_table_threshold: Optional[int] = None,
        sort_threshold: Optional[int] = None,
        cache_hit_threshold: Optional[int] = None,
        connection_threshold: Optional[int] = None,
        disk_write_threshold: Optional[int] = None,
        disk_iops_threshold: Optional[int] = None,
        max_query_cost: Optional[float] = None,
        memory_critical_threshold: Optional[int] = None,
        cpu_critical_threshold: Optional[int] = None,
        app_memory_high_threshold: Optional[int] = None,
        app_memory_pressure_threshold: Optional[int] = None,
        app_cpu_high_threshold: Optional[int] = None
    ) -> None:
        """Initialization with database connection."""

        self.conn = connection
        self.resource_monitor = None
        self.feature_extractor = PostgresFeatureExtractor(connection)

        # Init cache for historical data to avoid repeated queries
        self._historical_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self.cache_ttl = Defaults.CACHE_TTL

        # Set thresholds with defaults
        self.QUERY_LENGTH_LIMIT = (
            query_length_limit or 
            Defaults.QUERY_LENGTH_LIMIT
        )
        self.BASE_MEMORY_BYTES = (
            base_memory_bytes or 
            Defaults.BASE_MEMORY_BYTES
        )
        self.BASE_CPU_SECONDS = (
            base_cpu_seconds or 
            Defaults.BASE_CPU_SECONDS
        )
        self.COST_TO_CPU_FACTOR = (
            cost_to_cpu_factor or 
            Defaults.COST_TO_CPU_FACTOR
        )
        self.MEMORY_THRESHOLD_MEDIUM = (
            memory_threshold_medium or 
            Defaults.MEMORY_THRESHOLD_MEDIUM
        )
        self.MEMORY_THRESHOLD_HIGH = (
            memory_threshold_high or 
            Defaults.MEMORY_THRESHOLD_HIGH
        )
        self.CPU_THRESHOLD_MEDIUM = (
            cpu_threshold_medium or 
            Defaults.CPU_THRESHOLD_MEDIUM
        )
        self.CPU_THRESHOLD_HIGH = (
            cpu_threshold_high or 
            Defaults.CPU_THRESHOLD_HIGH
        )
        self.LARGE_TABLE_THRESHOLD = (
            large_table_threshold or 
            Defaults.LARGE_TABLE_THRESHOLD
        )
        self.SORT_THRESHOLD = (
            sort_threshold or 
            Defaults.SORT_THRESHOLD
        )
        self.CACHE_HIT_THRESHOLD = (
            cache_hit_threshold or 
            Defaults.CACHE_HIT_THRESHOLD
        )
        self.CONNECTION_THRESHOLD = (
            connection_threshold or 
            Defaults.CONNECTION_THRESHOLD
        )
        self.DISK_WRITE_THRESHOLD = (
            disk_write_threshold or 
            Defaults.DISK_WRITE_THRESHOLD
        )
        self.DISK_IOPS_THRESHOLD = (
            disk_iops_threshold or 
            Defaults.DISK_IOPS_THRESHOLD
        )
        self.MAX_QUERY_COST = (
            max_query_cost or 
            Defaults.MAX_QUERY_COST
        )
        self.MEMORY_CRITICAL_THRESHOLD = (
            memory_critical_threshold or 
            Defaults.MEMORY_CRITICAL_THRESHOLD
        )
        self.CPU_CRITICAL_THRESHOLD = (
            cpu_critical_threshold or 
            Defaults.CPU_CRITICAL_THRESHOLD
        )
        self.APP_MEMORY_HIGH_THRESHOLD = (
            app_memory_high_threshold or 
            Defaults.APP_MEMORY_HIGH_THRESHOLD
        )
        self.APP_MEMORY_PRESSURE_THRESHOLD = (
            app_memory_pressure_threshold or 
            Defaults.APP_MEMORY_PRESSURE_THRESHOLD
        )
        self.APP_CPU_HIGH_THRESHOLD = (
            app_cpu_high_threshold or 
            Defaults.APP_CPU_HIGH_THRESHOLD
        )

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

        cleaned_query = ' '.join(sql_query.splitlines()).strip()

        if len(cleaned_query.strip()) > self.QUERY_LENGTH_LIMIT:
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

            if (
                isinstance(execution_plan, dict) and 
                "Plan" in execution_plan
            ):
                plan_data = execution_plan["Plan"]
            
            elif (
                isinstance(execution_plan, list) and 
                len(execution_plan) > 0
            ):
                first_element = execution_plan[0]
                if (
                    isinstance(first_element, dict) and 
                    "Plan" in first_element
                ):
                    plan_data = first_element["Plan"]
                else:
                    plan_data = first_element
            else:
                plan_data = execution_plan

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
                query=cleaned_query,
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
            base_memory = self.BASE_MEMORY_BYTES
            
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
            base_cpu = self.BASE_CPU_SECONDS
            
            total_cost = plan_data.get("Total Cost", 0)
            
            # Convert PostgreSQL cost units to seconds
            cost_to_cpu_factor = self.COST_TO_CPU_FACTOR
            
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
            
            if memory_mb > self.MEMORY_THRESHOLD_MEDIUM:
                # Recommended work_mem (1.5x predicted memory)
                recommended_work_mem = max(
                    4, int(memory_mb * 1.5)
                )
                recs.append({
                    "type": "memory",
                    "priority": "MEDIUM",
                    "message": (
                        f"Query predicted to use "
                        f"{memory_mb:.1f}MB memory. "
                        f"Consider increasing work_mem to "
                        f"{recommended_work_mem}MB "
                        f"for this session: SET work_mem = "
                        f"'{recommended_work_mem}MB';"
                    ),
                    "predicted_memory_mb": memory_mb,
                    "suggestion": (
                        f"SET work_mem = "
                        f"'{recommended_work_mem}MB';"
                    )
                })

            if memory_mb > self.MEMORY_THRESHOLD_HIGH:
                recs.append({
                    "type": "memory",
                    "priority": "HIGH",
                    "message": (
                        f"High memory usage predicted "
                        f"({memory_mb:.1f}MB). "
                        f"This may impact other queries. Consider: "
                        f"Increasing work_mem temporarily; "
                        f"Using smaller batches; "
                        f"Adding appropriate indexes; "
                        f"Reviewing query structure for "
                        f"memory-intensive operations"
                    ),
                    "predicted_memory_mb": memory_mb,
                    "suggestions": [
                        (
                            f"SET work_mem = '64MB'; "
                            f"-- Adjust based on system capacity"
                        ),
                        (
                            f"Break query into smaller "
                            f"batches if possible"
                        ),
                        (
                            f"Ensure proper indexes exist "
                            f"on filtered/sorted columns"
                        )
                    ]
                })
        
        if analysis_result.predicted_cpu_seconds:
            cpu_ms = (
                analysis_result.predicted_cpu_seconds
                * 1000
            )

            if cpu_ms > self.CPU_THRESHOLD_MEDIUM:
                recs.append({
                    "type": "cpu",
                    "priority": "MEDIUM",
                    "message": (
                        f"Query predicted to use "
                        f"{cpu_ms:.0f}ms CPU time. "
                        f"Optimization suggestions: "
                        f"Add indexes on "
                        f"filtered/sorted columns; "
                        f"Use WHERE clauses to "
                        f"reduce processed rows; "
                        f"Consider materialized views "
                        f"for complex aggregations"
                    ),
                    "predicted_cpu_ms": cpu_ms,
                    "suggestions": [
                        (
                            f"CREATE INDEX on "
                            f"frequently filtered columns"
                        ),
                        (
                            f"Use EXPLAIN ANALYZE to "
                            f"identify bottlenecks"
                        ),
                        (
                            f"Consider partitioning "
                            f"large tables"
                        )
                    ]
                })

            if cpu_ms > self.CPU_THRESHOLD_HIGH:
                recs.append({
                    "type": "cpu",
                    "priority": "HIGH",
                    "message": (
                        f"High CPU usage predicted "
                        f"({cpu_ms:.0f}ms). "
                        f"Critical optimization needed: "
                        f"Run during off-peak hours; "
                        f"Add missing indexes immediately; "
                        f"Consider query rewriting; "
                        f"Evaluate table partitioning"
                    ),
                    "predicted_cpu_ms": cpu_ms,
                    "suggestions": [
                        (
                            f"Schedule during "
                            f"maintenance windows"
                        ),
                        (
                            f"CREATE INDEX on join "
                            f"and filter columns"
                        ),
                        (
                            f"Review query for "
                            f"unnecessary complexity"
                        )
                    ]
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

                        cleaned_query = (
                            re.sub(r'\r?\n', ' ', row[0]).strip()
                        )

                        stats_list.append({
                            "query": cleaned_query,
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
                        table_size.bytes_size < self.LARGE_TABLE_THRESHOLD 
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
            table_size.bytes_size > self.LARGE_TABLE_THRESHOLD
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
            if cache_hit_ratio < self.CACHE_HIT_THRESHOLD:
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

        if analysis_result.plan_rows > self.SORT_THRESHOLD:
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

        if pg_memory_percent > self.MEMORY_CRITICAL_THRESHOLD:
            recs.append({
                "type": "resource",
                "priority": "HIGH",
                "message": (
                    f"PostgreSQL container memory "
                    f"usage critically high "
                    f"({pg_memory_percent}%). "
                    f"Immediate actions: "
                    f"Increase container memory limits; "
                    f"Optimize shared_buffers and work_mem; "
                    f"Monitor for memory-intensive queries; "
                    f"Consider connection pooling to "
                    f"reduce memory per connection"
                ),
                "metric": "postgres_memory_usage",
                "value": pg_memory_percent,
                "threshold": self.MEMORY_CRITICAL_THRESHOLD,
                "suggestions": [
                    (
                        f"ALTER SYSTEM SET shared_buffers = "
                        f"'25% of total RAM';"
                    ),
                    (
                        f"ALTER SYSTEM SET work_mem = '4MB'; "
                        f"-- Adjust based on usage"
                    ),
                    (
                        f"Consider using pgbouncer "
                        f"for connection pooling"
                    )
                ]
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
            pg_cpu_percent > self.CPU_CRITICAL_THRESHOLD and 
            analysis_result.node_type in cpu_intensive_ops
        ):
            recs.append({
                "type": "performance",
                "priority": "MEDIUM",
                "message": (
                    f"PostgreSQL CPU usage very high "
                    f"({pg_cpu_percent}%). "
                    f"CPU-intensive {analysis_result.node_type} "
                    f"operations detected. "
                    f"Suggestions: "
                    f"Add appropriate indexes; "
                    f"Increase effective_cache_size; "
                    f"Consider parallel query settings"
                ),
                "metric": "postgres_cpu_usage",
                "value": pg_cpu_percent,
                "threshold": self.CPU_CRITICAL_THRESHOLD,
                "suggestions": [
                    (
                        f"CREATE INDEX on frequently "
                        f"accessed columns"
                    ),
                    (
                        f"ALTER SYSTEM SET effective_cache_size "
                        f"= '75% of total RAM';"
                    ),
                    (
                        f"SET max_parallel_workers_per_gather = 4; "
                        f"-- If appropriate"
                    )
                ]
            })

        # Disk I/O Pressure Detection
        pg_disk_io = pg_metrics.get('disk_io', {})

        if not isinstance(pg_disk_io, dict):
            pg_disk_io = {}

        pg_disk_write = (
            pg_disk_io.get('write_bytes', 0) 
            if isinstance(pg_disk_io, dict) else 0
        )

        if pg_disk_write > self.DISK_WRITE_THRESHOLD:
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
            app_memory_percent > self.APP_MEMORY_HIGH_THRESHOLD and 
            analysis_result.plan_rows > self.SORT_THRESHOLD
        ):
            recs.append({
                "type": "resource",
                "priority": "MEDIUM",
                "message": (
                    f"Application container memory usage "
                    f"is high ({app_memory_percent}%). "
                    f"Large result sets may "
                    f"increase memory pressure."
                ),
                "metric": "app_memory_usage",
                "value": app_memory_percent,
                "threshold": self.APP_MEMORY_HIGH_THRESHOLD
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
        if connection_ratio > self.CONNECTION_THRESHOLD:
            recs.append({
                "type": "connection",
                "priority": "MEDIUM", 
                "message": (
                    f"High database connection usage "
                    f"({active_connections}/{max_connections}, "
                    f"{connection_ratio:.1f}%). "
                    f"Solutions: "
                    f"Implement connection pooling (pgbouncer); "
                    f"Increase max_connections if needed; "
                    f"Review application connection management"
                ),
                "metric": "connection_pool_usage",
                "value": connection_ratio,
                "threshold": self.CONNECTION_THRESHOLD,
                "suggestions": [
                    (
                        f"ALTER SYSTEM SET max_connections = 200; "
                        f"-- If system can handle"
                    ),
                    (
                        f"Implement pgbouncer with "
                        f"transaction pooling"
                    ),
                    (
                        f"Review application connection "
                        f"timeout settings"
                    )
                ]
            })

        # disk IOPS
        pg_disk_iops = (
            pg_disk_io.get('iops', 0) 
            if isinstance(pg_disk_io, dict) else 0
        )

        if pg_disk_iops > self.DISK_IOPS_THRESHOLD:
            recs.append({
                "type": "io",
                "priority": "MEDIUM",
                "message": "High disk IOPS detected"
            })

        self._add_autovacuum_recommendations(
            analysis_result, recs
        )

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

        app_memory_pressure = (
            app_memory_percent > self.APP_MEMORY_PRESSURE_THRESHOLD
        )
        app_high_cpu = (
            app_cpu_percent > self.APP_CPU_HIGH_THRESHOLD
        )

        if (
            app_memory_pressure and 
            analysis_result.plan_rows > self.SORT_THRESHOLD
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
            analysis_result.total_cost > self.MAX_QUERY_COST * 0.5
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
        
        query_lines = []
        for line in sql_query.splitlines():
            clean_line = line.split('--')[0].strip()
            if clean_line:
                query_lines.append(clean_line)
        
        query_clean = ' '.join(query_lines).upper()
        
        # Query type classification
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
            
        elif query_clean.startswith(
            ('CREATE', 'DROP', 'ALTER')
        ):
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
            if isinstance(
                analysis_result.execution_plan,
                (dict, list)
            ):
                plan_str = json.dumps(
                    analysis_result.execution_plan
                ).upper()
            else:
                plan_str = str(
                    analysis_result.execution_plan
                ).upper()
            
            join_patterns = [
                'JOIN', 'NESTED LOOP', 'HASH JOIN',
                'MERGE JOIN', 'NESTED LOOP INNER',
                'NESTED LOOP LEFT', 'HASH INNER JOIN'
            ]
            analysis_result.contains_join = any(
                pattern in plan_str 
                for pattern in join_patterns
            )

            sort_patterns = ['SORT', 'ORDER BY', 'SORT KEY']
            analysis_result.contains_sort = any(
                pattern in plan_str 
                for pattern in sort_patterns
            )
            
            agg_patterns = [
                'AGGREGATE', 'GROUP', 'HASHAGG',
                'GROUPAGG', 'GROUP BY', 
                'COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('
            ]
            analysis_result.contains_aggregate = any(
                pattern in plan_str 
                for pattern in agg_patterns
            )
            
            if 'INDEX ONLY SCAN' in plan_str:
                analysis_result.recommendations.append({
                    "type": "performance",
                    "priority": "LOW", 
                    "message": "Query using efficient index-only scan",
                    "optimization": "index_only_scan"
                })

    def _add_autovacuum_recommendations(
        self: Self,
        analysis_result: QueryAnalysisResult,
        recs: list[dict[str, Any]]
    ) -> None:
        """Add autovacuum tuning recommendations."""

        try:
            if not hasattr(
                self.feature_extractor, 'get_table_statistics'
            ):
                return
                
            table_stats = (
                self.feature_extractor.get_table_statistics()
            )
            
            if not table_stats:
                return
            
            main_table = analysis_result.relation_name
            
            for table_stat in table_stats:
                n_dead_tup = table_stat.get('n_dead_tup', 0)
                n_live_tup = table_stat.get('n_live_tup', 0)
                table_name = (
                    f"{table_stat.get('schemaname', 'public')}."
                    f"{table_stat.get('table_name', 'unknown')}"
                )
                
                if n_live_tup > 0 and n_dead_tup > 1000:
                    dead_tuple_ratio = (
                        n_dead_tup / n_live_tup
                    ) * 100
                    
                    if dead_tuple_ratio > 10:
                        is_main_table = (
                            main_table and 
                            (main_table == table_stat.get('table_name') or 
                            main_table in table_name)
                        )
                        
                        priority = "HIGH" if is_main_table else "MEDIUM"
                        
                        message = (
                            f"Table '{table_name}' has "
                            f"{dead_tuple_ratio:.1f}% "
                            f"dead tuples ({n_dead_tup:,} "
                            f"dead of {n_live_tup:,} total)."
                        )
                        
                        if is_main_table:
                            message += (
                                f" This is the main table in your query "
                                f"and may significantly impact performance."
                            )
                        
                        recs.append({
                            "type": "maintenance",
                            "priority": priority,
                            "message": message,
                            "table": table_name,
                            "dead_tuples": n_dead_tup,
                            "live_tuples": n_live_tup,
                            "dead_tuple_ratio": dead_tuple_ratio,
                            "is_main_query_table": is_main_table,
                            "suggestions": [
                                (
                                    f"ALTER TABLE {table_name} "
                                    f"SET (autovacuum_vacuum_scale_factor = 0.05, "
                                    f"autovacuum_vacuum_threshold = 1000);"
                                ),
                                (
                                    f"VACUUM ANALYZE {table_name}; "
                                    f"-- For immediate cleanup"
                                ),
                                (
                                    f"Check autovacuum settings: "
                                    f"SHOW autovacuum_vacuum_scale_factor; "
                                    f"SHOW autovacuum_vacuum_threshold;"
                                )
                            ]
                        })
                        
        except Exception as e:
            print(f"Debug: Autovacuum recommendation error: {e}")
            pass

    def should_reject_query(
        self: Self,
        analysis_result: QueryAnalysisResult, 
        max_cost: float = None
    ) -> bool:
        """Determine if a query should be rejected."""

        cost_threshold = max_cost or self.MAX_QUERY_COST

        # Primary rejection:
        # query is too expensive regardless of system state
        if analysis_result.total_cost > cost_threshold:
            return True
 
        # Secondary rejection: 
        # moderately expensive query + high memory pressure
        if analysis_result.container_resources:
            resources = analysis_result.container_resources
            
            memory_percent = 0
            if (
                'memory' in resources and 
                isinstance(resources['memory'], dict) and
                'percent_used' in resources['memory']
            ):
                memory_percent = resources['memory']['percent_used']
            
            if (
                analysis_result.total_cost > cost_threshold * 0.7 and 
                memory_percent > self.APP_MEMORY_PRESSURE_THRESHOLD
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
        
        if result.execution_plan:
            try:
                print(
                    f"Advanced analysis - Execution plan type: "
                    f"{type(result.execution_plan)}"
                )
                
                if (
                    isinstance(result.execution_plan, list) and 
                    result.execution_plan
                ):
                    first_item = result.execution_plan[0]
                    print(
                        f"Advanced analysis - First item type: "
                        f"{type(first_item)}"
                    )
                    
                    if isinstance(first_item, dict):
                        print(
                            f"Advanced analysis - First item keys: "
                            f"{list(first_item.keys())}"
                        )
                    else:
                        print(
                            f"Advanced analysis - "
                            f"First item is not a dict: {first_item}"
                        )
                
                # Perform advanced analysis
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
                    
                    # Convert to dict for serialization
                    try:
                        result.advanced_metrics = asdict(
                            self.advanced_metrics
                        )
                        print(
                            f"Advanced metrics added "
                            f"successfully using asdict()"
                        )
                    except TypeError as e:
                        print(
                            f"asdict() failed: {e}, "
                            f"trying __dict__"
                        )

                        if hasattr(
                            self.advanced_metrics, '__dict__'
                        ):
                            result.advanced_metrics = (
                                self.advanced_metrics.__dict__

                            )
                        else:
                            result.advanced_metrics = {}
                else:
                    result.advanced_metrics = {}
                    print("Advanced analysis returned no metrics")
                    
            except Exception as e:
                print(f"Advanced analysis failed: {str(e)}")
                import traceback
                traceback.print_exc()
                result.advanced_metrics = {}
        
        return result
    
    def _add_advanced_recommendations(
        self: Self,
        result: QueryAnalysisResult, 
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Add advanced recommendations to analysis result."""
        
        advanced_recs = []
        
        recommendation_types = [
            'join_recommendations',
            'aggregation_recommendations', 
            'partitioning_recommendations',
            'index_recommendations'
        ]
        
        for rec_type in recommendation_types:
            if hasattr(metrics, rec_type):
                recommendations = getattr(metrics, rec_type, [])

                if (
                    recommendations and 
                    isinstance(recommendations, list)
                ):
                    advanced_recs.extend(recommendations)
        
        for rec in advanced_recs:
            if isinstance(rec, dict):
                rec['source'] = 'advanced_analyzer'
        
        result.recommendations.extend(advanced_recs)
    
    def get_advanced_metrics(
        self: Self
    ) -> Optional[AdvancedPlanMetrics]:
        """Get advanced analysis metrics."""

        return self.advanced_metrics
