"""Advanced query plan analysis module."""


import re
from typing import Optional, Any, Self
from dataclasses import dataclass, field
from pg_feature_extractor import (
    TableSize,
    IndexInfo,
    PostgresFeatureExtractor
)


@dataclass
class AdvancedPlanMetrics:
    """Enhanced metrics for query analysis."""

    # Join analysis
    join_types: list[str] = field(
        default_factory=list
    )
    join_tables: list[str] = field(
        default_factory=list
    )
    join_conditions: list[str] = field(
        default_factory=list
    )
    missing_join_indexes: list[dict[str, Any]] = field(
        default_factory=list
    )
    nested_loop_on_large_tables: bool = False

    # Aggregation analysis
    aggregation_types: list[str] = field(
        default_factory=list
    )
    aggregation_tables: list[str] = field(
        default_factory=list
    )
    expensive_aggregations: bool = False
    approximate_count_candidate: bool = False
    
    # Partitioning analysis
    large_table_scans: list[dict[str, Any]] = field(
        default_factory=list
    )
    partitioning_candidates: list[dict[str, Any]] = field(
        default_factory=list
    )
    
    # Index recommendations
    covering_index_candidates: list[dict[str, Any]] = field(
        default_factory=list
    )
    foreign_key_indexes_missing: list[dict[str, Any]] = field(
        default_factory=list
    )

    # Recommendations (added during analysis)
    join_recommendations: list[dict[str, Any]] = field(
        default_factory=list
    )
    aggregation_recommendations: list[dict[str, Any]] = field(
        default_factory=list
    )
    partitioning_recommendations: list[dict[str, Any]] = field(
        default_factory=list
    )
    index_recommendations: list[dict[str, Any]] = field(
        default_factory=list
    )


class AdvancedQueryAnalyzer:
    """Enhanced query analyzer with advanced features."""
    
    # Adjusted for ~10,000 row tables
    DEFAULT_LARGE_TABLE_THRESHOLD = 1024 * 1024 * 1        # 1MB (tables with ~1K-10K rows)
    DEFAULT_VERY_LARGE_TABLE_THRESHOLD = 1024 * 1024 * 10  # 10MB (tables with ~100K+ rows)
    DEFAULT_NESTED_LOOP_THRESHOLD = 1000                   # Number of rows
    DEFAULT_APPROXIMATE_COUNT_THRESHOLD = 5000             # Number of rows
    DEFAULT_COVERING_INDEX_THRESHOLD = 5                   # >5 columns

    def __init__(
        self: Self,
        feature_extractor: PostgresFeatureExtractor,
        large_table_threshold: Optional[int] = None,
        very_large_table_threshold: Optional[int] = None,
        nested_loop_threshold: Optional[int] = None,
        approximate_count_threshold: Optional[int] = None,
        covering_index_threshold: Optional[int] = None
    ) -> None:
        self.feature_extractor = feature_extractor
        self._table_cache: dict[str, TableSize] = {}
        self._index_cache: dict[str, list[IndexInfo]] = {}

        # Set thresholds with defaults
        self.LARGE_TABLE_THRESHOLD = (
            large_table_threshold or 
            self.DEFAULT_LARGE_TABLE_THRESHOLD
        )
        self.VERY_LARGE_TABLE_THRESHOLD = (
            very_large_table_threshold or 
            self.DEFAULT_VERY_LARGE_TABLE_THRESHOLD
        )
        self.NESTED_LOOP_THRESHOLD = (
            nested_loop_threshold or 
            self.DEFAULT_NESTED_LOOP_THRESHOLD
        )
        self.APPROXIMATE_COUNT_THRESHOLD = (
            approximate_count_threshold or 
            self.DEFAULT_APPROXIMATE_COUNT_THRESHOLD
        )
        self.COVERING_INDEX_THRESHOLD = (
            covering_index_threshold or 
            self.DEFAULT_COVERING_INDEX_THRESHOLD
        )

    def analyze_advanced_metrics(
        self: Self,
        execution_plan: dict[str, Any]
    ) -> AdvancedPlanMetrics:
        """Perform advanced analysis on execution plan."""

        metrics = AdvancedPlanMetrics()
        
        if (
            not execution_plan or 
            not isinstance(execution_plan, dict)
        ):
            return metrics
        
        if "Plan" in execution_plan:
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

        # Recursively analyze plan nodes
        self._analyze_plan_node(plan_data, metrics)
        
        # Generate additional recommendations
        self._generate_join_recommendations(metrics)
        self._generate_aggregation_recommendations(metrics)
        self._generate_partitioning_recommendations(metrics)
        self._generate_index_recommendations(metrics)

        return metrics
    
    def _analyze_plan_node(
        self: Self,
        node: dict[str, Any],
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Recursively analyze execution plan nodes."""

        node_type = node.get("Node Type", "")
        
        if "Join" in node_type:
            self._analyze_join_node(node, metrics)
        
        elif any(
            agg in node_type 
            for agg in ["Aggregate", "Group", "HashAgg"]
        ):
            self._analyze_aggregation_node(node, metrics)
        
        elif node_type == "Seq Scan":
            self._analyze_sequential_scan_node(node, metrics)

        # Handle different child node structures
        child_nodes = []

        if (
            "Plans" in node and 
            isinstance(node["Plans"], list)
        ):
            child_nodes = node["Plans"]

        elif (
            "Plans" in node and 
            isinstance(node["Plans"], dict)
        ):
            child_nodes = [node["Plans"]]
        
        for child_node in child_nodes:
            self._analyze_plan_node(child_node, metrics)
    
    def _analyze_join_node(
        self: Self,
        node: dict[str, Any],
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Analyze join operations."""

        node_type = node.get("Node Type", "")
        relation_name = node.get("Relation Name", "")

        join_type = ""
        if node_type and isinstance(node_type, str):
            join_type = node_type.replace(" Join", "").lower()
        
        metrics.join_types.append(join_type)
        if relation_name:
            metrics.join_tables.append(relation_name)
        
        if "Nested Loop" in node_type:
            plan_rows = node.get("Plan Rows", 0)
            if plan_rows > self.NESTED_LOOP_THRESHOLD:
                metrics.nested_loop_on_large_tables = True
        
        join_condition = (
            node.get("Join Filter") or 
            node.get("Hash Condition") or
            node.get("Merge Condition")
        )
        if join_condition and isinstance(join_condition, str):
            metrics.join_conditions.append(join_condition)
            self._analyze_join_condition(
                join_condition, relation_name, metrics
            )

    def _analyze_join_condition(
        self: Self,
        condition: str,
        table_name: str,
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Analyze join conditions for missing indexes."""

        column_pattern = r'(\w+)\.(\w+)'
        column_matches = re.findall(
            column_pattern, condition
        )

        for table_ref, column_name in column_matches:
            if table_ref == table_name or not table_ref:
                if not self._has_index_for_column(
                    table_name, column_name
                ):
                    metrics.missing_join_indexes.append({
                        "table": table_name,
                        "column": column_name,
                        "condition": condition
                    })

    def _analyze_aggregation_node(
        self: Self,
        node: dict[str, Any],
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Analyze aggregation operations."""

        node_type = node.get("Node Type", "")
        relation_name = node.get("Relation Name", "")
        plan_rows = node.get("Plan Rows", 0)

        metrics.aggregation_types.append(node_type)
        if relation_name:
            metrics.aggregation_tables.append(relation_name)

        table_size = self._get_table_size(relation_name)
        if (
            table_size and 
            table_size.bytes_size > self.LARGE_TABLE_THRESHOLD
        ):
            metrics.expensive_aggregations = True
        
        if (
            "Aggregate" in node_type and 
            plan_rows > self.APPROXIMATE_COUNT_THRESHOLD
        ):
            metrics.approximate_count_candidate = (
                self._is_count_operation_without_filter(node)
            )

    def _analyze_sequential_scan_node(
        self: Self,
        node: dict[str, Any],
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Analyze sequential scans for partitioning."""

        relation_name = node.get("Relation Name")
        filter_condition = node.get("Filter")
        
        if not relation_name:
            return
        
        table_size = self._get_table_size(relation_name)
        if (
            table_size and 
            table_size.bytes_size > self.VERY_LARGE_TABLE_THRESHOLD
        ):
            scan_info = {
                "table": relation_name,
                "size_bytes": table_size.bytes_size,
                "pretty_size": table_size.pretty_size,
                "filter_condition": filter_condition
            }
            metrics.large_table_scans.append(scan_info)

            if self._is_partitioning_candidate(
                filter_condition
            ):
                metrics.partitioning_candidates.append(scan_info)
    
    def _is_partitioning_candidate(
        self: Self,
        filter_condition: Optional[str]
    ) -> bool:
        """Check if table is a good candidate for partitioning."""

        if (
            not filter_condition or 
            not isinstance(filter_condition, str)
        ):
            return False

        # Look for date/time filters (common partitioning columns)
        date_patterns = [
            r'date\s*[<>=]', r'timestamp\s*[<>=]', 
            r'extract\s*\(.*from', r'date_part\s*\('
        ]

        for pattern in date_patterns:
            if re.search(pattern, filter_condition, re.IGNORECASE):
                return True
        
        # Look for range filters on integer columns
        range_patterns = [
            r'id\s*[<>=]',
            r'number\s*[<>=]',
            r'value\s*[<>=]'
        ]
        for pattern in range_patterns:
            if re.search(pattern, filter_condition, re.IGNORECASE):
                return True
        
        return False
    
    def _generate_join_recommendations(
        self: Self,
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Generate join-specific recommendations."""

        recommendations = []

        if metrics.nested_loop_on_large_tables:
            recommendations.append({
                "type": "join_strategy",
                "priority": "HIGH",
                "message": (
                    f"Nested loop join detected on "
                    f"large result set. "
                    f"Consider using hash joins or "
                    f"merge joins for better performance "
                    f"on large datasets."
                ),
                "suggestion": (
                    f"Enable hash joins with "
                    f"SET enable_nestloop = off; for testing"
                )
            })

        for missing_index in metrics.missing_join_indexes:
            recommendations.append({
                "type": "join_index",
                "priority": "MEDIUM",
                "message": (
                    f"Missing index on "
                    f"{missing_index['table']}."
                    f"{missing_index['column']} "
                    f"used in join condition: "
                    f"{missing_index['condition']}"
                ),
                "table": missing_index["table"],
                "column": missing_index["column"],
                "index_suggestion": (
                    f"CREATE INDEX idx_"
                    f"{missing_index['table']}_"
                    f"{missing_index['column']} "
                    f"ON {missing_index['table']} "
                    f"({missing_index['column']});"
                )
            })
        
        metrics.join_recommendations = recommendations
    
    def _generate_aggregation_recommendations(
        self: Self,
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Generate aggregation-specific recommendations."""

        recommendations = []

        if metrics.expensive_aggregations:
            for table in set(metrics.aggregation_tables):
                table_size = self._get_table_size(table)
                if (
                    table_size and 
                    table_size.bytes_size > self.LARGE_TABLE_THRESHOLD
                ):
                    recommendations.append({
                        "type": "aggregation",
                        "priority": "MEDIUM",
                        "message": (
                            f"Expensive aggregation operation "
                            f"on large table '{table}' "
                            f"({table_size.pretty_size}). "
                            f"Consider using materialized views "
                            f"for pre-aggregated results."
                        ),
                        "table": table,
                        "suggestion": (
                            f"CREATE MATERIALIZED VIEW "
                            f"mv_aggregated_data AS "
                            f"SELECT ... FROM {table} GROUP BY ...;"
                        )
                    })
        
        if metrics.approximate_count_candidate:
            recommendations.append({
                "type": "aggregation",
                "priority": "LOW",
                "message": (
                    f"COUNT(*) operation on "
                    f"large table without filters. "
                    f"If exact count is not required, "
                    f"consider using approximate count "
                    f"techniques for better performance."
                ),
                "suggestion": (
                    f"Use pg_stat_user_tables or "
                    f"sampling for approximate counts"
                )
            })

        metrics.aggregation_recommendations = recommendations
    
    def _generate_partitioning_recommendations(
        self: Self,
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Generate partitioning recommendations."""

        recommendations = []
        
        for candidate in metrics.partitioning_candidates:
            recommendations.append({
                "type": "partitioning",
                "priority": "LOW",
                "message": (
                    f"Large table '{candidate['table']}' "
                    f"({candidate['pretty_size']}) "
                    f"with range-based filters detected. "
                    f"Consider table partitioning "
                    f"for better query performance "
                    f"and maintenance."
                ),
                "table": candidate["table"],
                "size": candidate["pretty_size"],
                "filter_condition": candidate["filter_condition"],
                "suggestion": (
                    f"Investigate partitioning "
                    f"{candidate['table']} "
                    f"by date range or key range"
                )
            })
        
        metrics.partitioning_recommendations = recommendations

    def _generate_index_recommendations(
        self: Self,
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Generate advanced index recommendations."""

        recommendations = []

        has_get_table_columns = hasattr(
            self.feature_extractor, 'get_table_columns'
        )

        for table in set(
            metrics.join_tables + metrics.aggregation_tables
        ):
            if table:
                 if has_get_table_columns:
                    try:
                        table_columns = (
                            self.feature_extractor.get_table_columns(
                                table
                            )
                        )
                        threshold = self.COVERING_INDEX_THRESHOLD
                        if (
                            table_columns and 
                            len(table_columns) > threshold
                        ):
                            recommendations.append({
                                "type": "covering_index",
                                "priority": "LOW",
                                "message": (
                                    f"Table '{table}' has "
                                    f"{len(table_columns)} columns "
                                    f"and is used in operations. "
                                    f"Consider covering indexes for "
                                    f"frequently accessed columns."
                                ),
                                "table": table,
                                "column_count": len(table_columns)
                            })
                    except Exception:
                        pass
        
        for table in set(metrics.join_tables):
            if table:
                fk_indexes_missing = (
                    self._get_missing_foreign_key_indexes(table)
                )
                for fk_info in fk_indexes_missing:
                    recommendations.append({
                        "type": "foreign_key_index",
                        "priority": "MEDIUM",
                        "message": (
                            f"Missing index on foreign key column "
                            f"{fk_info['column']} "
                            f"in table {fk_info['table']}"
                        ),
                        "table": fk_info["table"],
                        "column": fk_info["column"],
                        "index_suggestion": (
                            f"CREATE INDEX idx_"
                            f"{fk_info['table']}_"
                            f"{fk_info['column']} "
                            f"ON {fk_info['table']} "
                            f"({fk_info['column']});"
                        )
                    })

        metrics.index_recommendations = recommendations
    
    def _has_index_for_column(
        self: Self,
        table_name: str,
        column_name: str
    ) -> bool:
        """Check if a column has an index."""

        if not table_name or not column_name:
            return False

        if table_name not in self._index_cache:
            try:
                self._index_cache[table_name] = (
                    self.feature_extractor.get_table_indexes(
                        table_name
                    )
                )
            except Exception:
                self._index_cache[table_name] = []
        
        for index in self._index_cache.get(table_name, []):
            if (
                hasattr(index, 'index_definition') and 
                index.index_definition
            ):
                if self._is_column_in_index_definition(
                    column_name, index.index_definition
                ):
                    return True
        
        return False

    def _is_column_in_index_definition(
        self: Self,
        column_name: str,
        index_definition: str
    ) -> bool:
        """Check if a column is in index definition SQL."""

        if not index_definition or not column_name:
            return False
        
        word_pattern = rf'\b{re.escape(column_name)}\b'
        quoted_pattern = rf'"{re.escape(column_name)}"'      
        parenthesized_pattern = (
            rf'\(\s*{re.escape(column_name)}\s*[\),]'
        )
        
        return (
            re.search(
                word_pattern,
                index_definition,
                re.IGNORECASE
            ) is not None or
            re.search(
                quoted_pattern,
                index_definition
            ) is not None or
            re.search(
                parenthesized_pattern,
                index_definition,
                re.IGNORECASE
            ) is not None
        )

    def _get_table_size(
        self: Self,
        table_name: str
    ) -> Optional[TableSize]:
        """Get table size with caching."""

        if not table_name:
            return None
        
        if table_name not in self._table_cache:
            try:
                self._table_cache[table_name] = (
                    self.feature_extractor.get_table_size(
                        table_name
                    )
                )

            except Exception:
                self._table_cache[table_name] = None
        
        return self._table_cache.get(table_name)
    
    def _is_count_operation_without_filter(
        self: Self,
        node: dict[str, Any]
    ) -> bool:
        """Check if this is COUNT operation without filters."""

        plan_rows = node.get("Plan Rows", 0)
        threshold = self.DEFAULT_APPROXIMATE_COUNT_THRESHOLD

        return (
            plan_rows > threshold and 
            "Filter" not in node and
            str(node).find("count") != -1
        )

    def _get_missing_foreign_key_indexes(
        self: Self,
        table_name: str
    ) -> list[dict[str, Any]]:
        """Get foreign key columns without indexes."""

        missing_indexes = []

        if not table_name:
            return missing_indexes

        try:
            foreign_keys = (
                self.feature_extractor.get_foreign_keys(
                    table_name
                )
            )
            for fk in foreign_keys:

                # Handle different foreign key structures
                column_name = None

                if hasattr(fk, 'column_name'):
                    column_name = fk.column_name

                elif hasattr(fk, 'column'):
                    column_name = fk.column
                
                if (
                    column_name and 
                    not self._has_index_for_column(
                        table_name, column_name
                    )
                ):
                    missing_indexes.append({
                        "table": table_name,
                        "column": column_name,
                        "references": (
                            getattr(
                                fk, 'referenced_table', 'unknown'
                            )
                        )
                    })

        except Exception:
            pass
        
        return missing_indexes
