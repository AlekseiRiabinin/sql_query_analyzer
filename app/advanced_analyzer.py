"""Advanced query plan analysis module."""


import re
from typing import Optional, Any, Self
from dataclasses import dataclass, field
from constants import Defaults
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
            Defaults.LARGE_TABLE_THRESHOLD
        )
        self.VERY_LARGE_TABLE_THRESHOLD = (
            very_large_table_threshold or 
            Defaults.VERY_LARGE_TABLE_THRESHOLD
        )
        self.NESTED_LOOP_THRESHOLD = (
            nested_loop_threshold or 
            Defaults.NESTED_LOOP_THRESHOLD
        )
        self.APPROXIMATE_COUNT_THRESHOLD = (
            approximate_count_threshold or 
            Defaults.APPROXIMATE_COUNT_THRESHOLD
        )
        self.COVERING_INDEX_THRESHOLD = (
            covering_index_threshold or 
            Defaults.COVERING_INDEX_THRESHOLD
        )

    def analyze_advanced_metrics(
        self: Self,
        execution_plan: Any
    ) -> AdvancedPlanMetrics:
        """Perform advanced analysis on execution plan."""
        
        print(
            f"Advanced analysis started - "
            f"execution_plan type: {type(execution_plan)}"
        )
        
        metrics = AdvancedPlanMetrics()
        
        if not execution_plan:
            print(
                f"No execution plan provided "
                f"for advanced analysis"
            )
            return metrics
        
        try:
            plan_data = None
            
            if (
                isinstance(execution_plan, list) and 
                len(execution_plan) > 0
            ):
                print(
                    f"Execution plan is a list, "
                    f"extracting first element"
                )
                first_element = execution_plan[0]
                
                if (
                    isinstance(first_element, dict) and 
                    "Plan" in first_element
                ):
                    plan_data = first_element["Plan"]
                    print("Found Plan in first element")
                else:
                    plan_data = first_element
                    print(
                        f"Using first element directly "
                        f"as plan data"
                    )
                    
            elif isinstance(execution_plan, dict):
                if "Plan" in execution_plan:
                    plan_data = execution_plan["Plan"]
                    print("Found Plan in dict")
                else:
                    plan_data = execution_plan
                    print("Using execution_plan dict directly")
            
            if (
                not plan_data or 
                not isinstance(plan_data, dict)
            ):
                print(
                    f"No valid plan data available "
                    f"for analysis"
                )
                return metrics
            
            print(
                f"Starting recursive plan analysis on: "
                f"{plan_data.get('Node Type', 'Unknown')}"
            )
            
            # Recursively analyze plan nodes
            self._analyze_plan_node(plan_data, metrics)
            
            print(
                f"Join analysis completed. Found "
                f"{len(metrics.join_types)} join types"
            )
            print(
                f"Missing join indexes: "
                f"{len(metrics.missing_join_indexes)}"
            )
            
            # Generate recommendations
            self._generate_join_recommendations(metrics)
            self._generate_aggregation_recommendations(metrics)
            self._generate_partitioning_recommendations(metrics)
            self._generate_index_recommendations(metrics)
            
            print(
                f"Generated {len(metrics.join_recommendations)} "
                f"join recommendations"
            )
            
        except Exception as e:
            print(f"Error in advanced analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        
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
        print(f"Analyzing join node: {node_type}")
        
        join_type = ""
        if (
            node_type and 
            isinstance(node_type, str) and 
            "Join" in node_type
        ):
            join_type = (
                node_type.replace(" Join", "").lower()
            )
        
        metrics.join_types.append(join_type)
        print(f"Join type extracted: {join_type}")
        
        relation_names = []
        if (
            "Plans" in node and 
            isinstance(node["Plans"], list)
        ):
            for plan in node["Plans"]:
                self._extract_relation_names_from_plan(
                    plan, relation_names
                )
        
        metrics.join_tables.extend(relation_names)
        print(f"Found relations in join: {relation_names}")
        
        if (
            isinstance(node_type, str) and 
            "Nested Loop" in node_type
        ):
            plan_rows = node.get("Plan Rows", 0)

            if (
                isinstance(plan_rows, (int, float)) and 
                plan_rows > self.NESTED_LOOP_THRESHOLD
            ):
                metrics.nested_loop_on_large_tables = True
                print("Nested loop on large tables detected!")
    
        join_condition = None
        for condition_key in [
            "Join Filter", "Hash Condition", "Merge Condition"
        ]:
            condition = node.get(condition_key)
            if condition and isinstance(condition, str):
                join_condition = condition
                break
        
        if join_condition and isinstance(join_condition, str):
            print(f"Join condition found: {join_condition}")
            metrics.join_conditions.append(join_condition)

            for table_name in relation_names:
                self._analyze_join_condition(
                    join_condition, table_name, metrics
                )

    def _extract_relation_names_from_plan(
        self: Self,
        plan_node: Any,
        relation_names: list[str]
    ) -> None:
        """Extract relation names from plan nodes."""
        
        if isinstance(plan_node, dict):
            relation_name = None

            if "Relation Name" in plan_node:
                relation_name = plan_node["Relation Name"]
 
            elif (
                hasattr(plan_node, 'get') and 
                callable(plan_node.get)
            ):
                relation_name = plan_node.get("Relation Name")

            if (
                relation_name and 
                isinstance(relation_name, str) and 
                relation_name not in relation_names
            ):
                relation_names.append(relation_name)
                print(f"Found relation: {relation_name}")
            
            # Recursively check child plans
            if (
                "Plans" in plan_node and 
                isinstance(plan_node["Plans"], list)
            ):
                for child_plan in plan_node["Plans"]:
                    self._extract_relation_names_from_plan(
                        child_plan, relation_names
                    )

        elif hasattr(plan_node, '__dict__'):
            plan_dict = plan_node.__dict__
            
            relation_name = None

            if "Relation_Name" in plan_dict:
                relation_name = plan_dict["Relation_Name"]

            elif "relation_name" in plan_dict:
                relation_name = plan_dict["relation_name"]

            elif "Relation Name" in plan_dict:
                relation_name = plan_dict["Relation Name"]

            elif hasattr(plan_node, 'Relation_Name'):
                relation_name = getattr(
                    plan_node, 'Relation_Name', None
                )

            elif hasattr(plan_node, 'relation_name'):
                relation_name = getattr(
                    plan_node, 'relation_name', None
                )

            elif hasattr(plan_node, 'Relation Name'):
                relation_name = getattr(
                    plan_node, 'Relation Name', None
                )
            
            if (
                relation_name and 
                isinstance(relation_name, str) and 
                relation_name not in relation_names
            ):
                relation_names.append(relation_name)
                print(f"Found relation: {relation_name}")

    def _analyze_join_condition(
        self: Self,
        condition: str,
        table_name: str,
        metrics: AdvancedPlanMetrics
    ) -> None:
        """Analyze join conditions for missing indexes."""
        
        print(
            f"Analyzing join condition for "
            f"{table_name}: {condition}"
        )

        column_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
        column_matches = re.findall(column_pattern, condition)

        if not column_matches:
            print(
                f"No column patterns found in condition: "
                f"{condition}"
            )
            return
        
        for table_ref, column_name in column_matches:
            print(
                f"Found column reference: "
                f"{table_ref}.{column_name}"
            )
            
            if table_ref == table_name:
                has_index = self._has_index_for_column(
                    table_name, column_name
                )
                print(
                    f"Index check for "
                    f"{table_name}.{column_name}: "
                    f"{has_index}"
                )
                
                if not has_index:
                    print(
                        f"Missing index detected on "
                        f"{table_name}.{column_name}"
                    )
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
                    f"Nested loop join detected "
                    f"on large result set. "
                    f"Performance improvements: "
                    f"Use hash joins for large datasets; "
                    f"Ensure proper indexes exist; "
                    f"Consider increasing work_mem "
                    f"for hash operations"
                ),
                "suggestions": [
                    (
                        f"SET enable_nestloop = off; "
                        f"-- Test hash joins"
                    ),
                    (
                        f"CREATE INDEX on join columns"
                    ),
                    (
                        f"SET work_mem = '16MB'; "
                        f"-- For better hash performance"
                    )
                ]
            })

        for missing_index in metrics.missing_join_indexes:
            recommendations.append({
                "type": "join_index",
                "priority": "MEDIUM",
                "message": (
                    f"Missing index on "
                    f"{missing_index['table']}."
                    f"{missing_index['column']} "
                    f"used in join condition. "
                    f"This index could improve join performance "
                    f"by {missing_index.get(
                        'potential_improvement', 'significant'
                    )}%."
                ),
                "table": missing_index["table"],
                "column": missing_index["column"],
                "index_suggestion": (
                    f"CREATE INDEX CONCURRENTLY idx_"
                    f"{missing_index['table']}_"
                    f"{missing_index['column']} "
                    f"ON {missing_index['table']} "
                    f"({missing_index['column']});"
                ),
                "estimated_improvement": "20-50% faster joins"
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
                            f"Expensive aggregation "
                            f"on large table '{table}' "
                            f"({table_size.pretty_size}). "
                            f"Optimization strategies: "
                            f"Use materialized views "
                            f"for pre-aggregation; "
                            f"Add appropriate indexes; "
                            f"Consider incremental aggregation"
                        ),
                        "table": table,
                        "suggestions": [
                            (
                                f"CREATE MATERIALIZED "
                                f"VIEW mv_{table}_aggregates "
                                f"AS SELECT ... FROM {table} "
                                f"GROUP BY ...;"
                            ),
                            (
                                f"CREATE INDEX idx_{table}_grouping "
                                f"ON {table} (grouping_columns);"
                            ),
                            (
                                f"SET work_mem = '32MB'; "
                                f"-- For large aggregations"
                            )
                        ]
                    })

        if metrics.approximate_count_candidate:
            recommendations.append({
                "type": "aggregation",
                "priority": "LOW",
                "message": (
                    f"COUNT(*) operation on large "
                    f"table without filters. "
                    f"Approximate count alternatives: "
                    f"Use pg_stat_user_tables.n_live_tup; "
                    f"Use sampling techniques; "
                    f"Maintain counter table if "
                    f"exact counts needed"
                ),
                "suggestions": [
                    (
                        f"SELECT n_live_tup "
                        f"FROM pg_stat_user_tables "
                        f"WHERE relname = 'table_name';"
                    ),
                    (
                        f"Use TABLESAMPLE for "
                        f"approximate counts"
                    ),
                    (
                        f"CREATE TABLE count_stats "
                        f"(table_name text, count bigint, "
                        f"updated_at timestamp);"
                    )
                ]
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
                    f"Partitioning benefits: "
                    f"Faster query performance; "
                    f"Easier maintenance; "
                    f"Better vacuum efficiency"
                ),
                "table": candidate["table"],
                "size": candidate["pretty_size"],
                "filter_condition": candidate["filter_condition"],
                "suggestions": [
                    (
                        f"CREATE TABLE "
                        f"{candidate['table']}_partitioned "
                        f"(LIKE {candidate['table']}) "
                        f"PARTITION BY RANGE (date_column);"
                    ),
                    (
                        f"CREATE INDEX ON {candidate['table']} "
                        f"(partition_key);"
                    ),
                    (
                        f"Consider using pg_partman "
                        f"for automatic partitioning"
                    )
                ]
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

                        idx_threshold = self.COVERING_INDEX_THRESHOLD
                        if (
                            table_columns and 
                            len(table_columns) > idx_threshold
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

                        table_size = self._get_table_size(table)
                        table_threshold = self.VERY_LARGE_TABLE_THRESHOLD
                        if (
                            table_size and 
                            table_size.bytes_size > table_threshold
                        ):
                            recommendations.append({
                                "type": "index_strategy",
                                "priority": "LOW",
                                "message": (
                                    f"Very large table "
                                    f"'{table}' detected. "
                                    f"Consider BRIN indexes "
                                    f"for range queries on "
                                    f"timestamp/date columns "
                                    f"to save space."
                                ),
                                "table": table,
                                "suggestion": (
                                    f"CREATE INDEX "
                                    f"idx_{table}_timestamp_brin "
                                    f"ON {table} USING BRIN "
                                    f"(timestamp_column);"
                                )
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

        print(
            f"Checking index for "
            f"{table_name}.{column_name}"
        )

        if not table_name or not column_name:
            return False

        if table_name not in self._index_cache:
            print(
                f"Loading indexes for table: "
                f"{table_name}"
            )
            try:
                indexes = (
                    self.feature_extractor.get_table_indexes(
                        table_name
                    )
                )
                self._index_cache[table_name] = indexes
                print(
                    f"Found {len(indexes)} "
                    f"indexes for {table_name}"
                )

            except Exception as e:
                print(
                    f"Error loading indexes for "
                    f"{table_name}: {e}"
                )
                self._index_cache[table_name] = []

        for index in self._index_cache.get(table_name, []):
            index_def = None
            
            if isinstance(index, dict):
                if "index_definition" in index:
                    index_def = index["index_definition"]
                elif "indexdef" in index:
                    index_def = index["indexdef"]
        
            if (
                index_def and 
                self._is_column_in_index_definition(
                    column_name, index_def
                )
            ):
                print(
                    f"Index found for "
                    f"{table_name}.{column_name}: "
                    f"{index_def[:100]}..."
                )
                return True
    
        print(
            f"No index found for "
            f"{table_name}.{column_name}"
        )
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
        threshold = self.APPROXIMATE_COUNT_THRESHOLD

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
            
            index_stats = (
                self.feature_extractor.get_index_usage_stats(
                    table_name
                )
            )
            indexed_columns = set()
            
            for index_stat in index_stats:
                index_name = None

                if (
                    isinstance(index_stat, dict) and 
                    "index_name" in index_stat
                ):
                    index_name = index_stat["index_name"]

                elif (
                    hasattr(index_stat, 'get') and 
                    callable(index_stat.get)
                ):
                    index_name = index_stat.get(
                        "index_name", ""
                    )

                elif hasattr(index_stat, 'index_name'):
                    index_name = getattr(
                        index_stat, 'index_name', ""
                    )
                
                if (
                    index_name and 
                    isinstance(index_name, str) and 
                    '_' in index_name
                ):
                    try:
                        possible_columns = index_name.split('_')[1:]
                        indexed_columns.update(
                            possible_columns
                        )

                    except (AttributeError, TypeError, IndexError):
                        pass

            for fk in foreign_keys:
                column_name = None
                referenced_table = "unknown"
                
                if isinstance(fk, dict):
                    column_name = (
                        fk.get("column_name") 
                        if "column_name" in fk 
                        else None
                    )
                    referenced_table = fk.get(
                        "referenced_table", "unknown"
                    )

                elif hasattr(fk, 'column_name'):
                    column_name = getattr(
                        fk, 'column_name', None
                    )
                    referenced_table = getattr(
                        fk, 'referenced_table', "unknown"
                    )
                
                if (
                    column_name and 
                    isinstance(column_name, str) and 
                    column_name not in indexed_columns
                ):
                    missing_indexes.append({
                        "table": table_name,
                        "column": column_name,
                        "references": referenced_table,
                        "reason": "Foreign key column missing index"
                    })
                    print(
                        f"Missing index on foreign key: "
                        f"{table_name}.{column_name}"
                    )

        except Exception as e:
            print(
                f"Error checking foreign key indexes for "
                f"{table_name}: {e}"
            )
        
        return missing_indexes
