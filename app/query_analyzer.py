import psycopg
from typing import Self
from pg_feature_extractor import PostgresFeatureExtractor


class QueryAnalyzer:
    """."""

    def __init__(self: Self, connection: psycopg) -> None:
        self.conn = connection
        self.feature_extractor = PostgresFeatureExtractor(connection)

    def analyze_query(self: Self, sql_query: str):
        """."""
        with self.conn.cursor() as cur:
            cur.execute(f"EXPLAIN (FORMAT JSON, COSTS, BUFFERS) {sql_query}")
            plan_json = cur.fetchone()[0]
        plan_data = plan_json[0]["Plan"]

        analysis_result = {
            "total_cost": plan_data.get("Total Cost"),
            "plan_rows": plan_data.get("Plan Rows"),
            "node_type": plan_data.get("Node Type"),
            "relation_name": plan_data.get("Relation Name"),
            "shared_buffers_hit": plan_data.get("Shared Hit Blocks", 0),
            "shared_buffers_read": plan_data.get("Shared Read Blocks", 0),
            "recommendations": []
        }

        self._generate_recommendations(analysis_result, plan_data)

        return analysis_result

    def _generate_recommendations(
            self: Self,
            analysis_result: dict,
            plan_data: dict
    ) -> :
        """."""
        recs = analysis_result["recommendations"]

        # Sequential Scan of Large Table
        if plan_data.get("Node Type") == "Seq Scan":
            table_name = plan_data.get("Relation Name")
            if table_name:
                size_pretty, size_bytes = (
                    self.feature_extractor.get_table_size(table_name)
                )
                if size_bytes > 1024 * 1024 * 100:
                    filter_condition = plan_data.get("Filter", "unknown condition")
                    recs.append({
                        "type": "index",
                        "priority": "HIGH",
                        "message": (
                            f"Seq Scan on large table '{table_name}' ({size_pretty}). "
                            f"Consider an index for condition: {filter_condition}"
                        )
                    })

        # Low cache efficiency
        total_buffers = (
            analysis_result["shared_buffers_hit"] + 
            analysis_result["shared_buffers_read"]
        )
        if total_buffers > 0:
            cache_hit_ratio = (
                analysis_result["shared_buffers_hit"] / total_buffers
            ) * 100
            if cache_hit_ratio < 90:
                recs.append({
                    "type": "configuration",
                    "priority": "MEDIUM",
                    "message": (
                        f"Low buffer cache hit ratio ({cache_hit_ratio:.2f}%). "
                        f"Query is reading from disk. "
                        f"Consider increasing shared_buffers or optimizing working set."
                    )
                })

        # Sorting without index
        if plan_data.get("Node Type") == "Sort":
            recs.append({
                "type": "query",
                "priority": "MEDIUM",
                "message": (
                    f"Expensive Sort operation. "
                    f"Check if an index on ORDER BY columns is possible."
                )
            })
