import logging
import json
from collections.abc import AsyncIterator
from typing import cast, Any, Self
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp


logger = logging.getLogger(__name__)


def safe_get(
    data: dict[str, Any],
    key: str,
    default: Any = None
) -> Any:
    """Safely get value from dictionary."""

    value = data.get(key, default)

    if value is default and default is not None:
        return default

    return value


class ResponseLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for logging HTTP responses."""
    
    def __init__(self: Self, app: ASGIApp) -> None:
        """Initialize the logging middleware."""
        super().__init__(app)

    async def dispatch(
        self: Self,
        request: Request,
        call_next: Any
    ) -> Response:
        """Process requests and log responses."""

        response = await call_next(request)
        
        if request.url.path in [
            "/analyze", "/stats/historical"
        ]:
            try:
                body = b""
                if hasattr(response, 'body_iterator'):
                    body_iterator = cast(
                        AsyncIterator[bytes],
                        response.body_iterator
                    )
                    async for chunk in body_iterator:
                        body += chunk
                
                if body:
                    response_data = json.loads(body.decode())
                    
                    if request.url.path == "/analyze":
                        self._log_analysis_response(
                            request, response_data
                        )

                    elif request.url.path == "/stats/historical":
                        self._log_historical_response(
                            request, response_data
                        )
                
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
                
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse JSON response: {e}"
                )
                return response

            except Exception as e:
                logger.error(
                    f"Error in response logging: {e}"
                )
                return response
        
        return response

    def _log_analysis_response(
        self: Self,
        request: Request,
        response_data: dict[str, Any]
    ) -> None:
        """Log analysis response data."""
 
        try:
            query = str(
                safe_get(response_data, "query", "N/A")
            )
            timestamp = safe_get(
                response_data, "timestamp", "N/A"
            )
            
            logger.info(f"Analysis Request: {request.url}")
            logger.info(
                f"Query: "
                f"{query[:100]}"
                f"{'...' if len(query) > 100 else ''}"
            )
            logger.info(f"Timestamp: {timestamp}")

            perf = safe_get(
                response_data, "performance_metrics", {}
            ) or {}
            total_cost = safe_get(perf, "total_cost", "N/A")
            node_type = safe_get(perf, "node_type", "N/A")
            relation_name = safe_get(perf, "relation_name", "N/A")
            plan_rows = safe_get(perf, "plan_rows", "N/A")
            
            logger.info(f"Total Cost: {total_cost}")
            logger.info(f"Node Type: {node_type}")
            logger.info(f"Relation: {relation_name}")
            logger.info(f"Plan Rows: {plan_rows}")
            
            buf_hit = int(safe_get(
                perf, "shared_buffers_hit", 0
            )) or 0
            buf_read = int(safe_get(
                perf, "shared_buffers_read", 0
            )) or 0
            logger.info(
                f"Buffer Hits: "
                f"{buf_hit} | Reads: {buf_read}"
            )
            
            if buf_hit + buf_read > 0:
                hit_ratio = (
                    buf_hit / (buf_hit + buf_read)
                ) * 100
                logger.info(
                    f"Buffer Hit Ratio: {hit_ratio:.1f}%"
                )
            
            recommendations = safe_get(
                response_data, "recommendations", []
            )
            if not isinstance(recommendations, list):
                recommendations = []
                
            logger.info(
                f"Optimization Recommendations: "
                f"{len(recommendations)}"
            )
            
            for i, rec in enumerate(recommendations[:3]):
                if isinstance(rec, dict):
                    priority = safe_get(rec, "priority", "N/A")
                    rec_type = safe_get(rec, "type", "N/A")
                    message = str(safe_get(
                        rec, "message", "N/A")
                    )[:60]
                    logger.info(
                        f"  {i+1}. "
                        f"[{priority}] {rec_type}: {message}..."
                    )
            
            if logger.isEnabledFor(logging.DEBUG):
                self._log_debug_details(response_data)
                
        except Exception as e:
            logger.error(
                f"Failed to log analysis response: {str(e)}"
            )

    def _log_debug_details(
        self:Self,
        response_data: dict[str, Any]
    ) -> None:
        """Log detailed debug information."""

        try:
            chars = safe_get(
                response_data, "query_characteristics", {}
            ) or {}
            query_type = safe_get(chars, "query_type")
            contains_join = safe_get(chars, "contains_join")
            contains_sort = safe_get(chars, "contains_sort")
            contains_aggregate = safe_get(
                chars, "contains_aggregate"
            )
            
            logger.debug(f"Query Type: {query_type}")
            logger.debug(
                f"Join: {contains_join} | "
                f"Sort: {contains_sort} | "
                f"Aggregate: {contains_aggregate}"
            )

            resources = safe_get(
                response_data, "resource_metrics", {}
            ) or {}
            predicted_memory = safe_get(
                resources, "predicted_memory_bytes"
            )
            predicted_cpu = safe_get(
                resources, "predicted_cpu_seconds"
            )
            
            logger.debug(
                f"Predicted Memory: "
                f"{predicted_memory} bytes"
            )
            logger.debug(
                f"Predicted CPU: "
                f"{predicted_cpu} seconds"
            )

            containers = safe_get(
                resources, "container_resources", {}
            ) or {}

            if containers:
                app_container = safe_get(
                    containers, "application_container", {}
                ) or {}
                pg_container = safe_get(
                    containers, "postgres_container", {}
                ) or {}
                
                app_cpu_dict = safe_get(
                    app_container, "cpu", {}
                ) or {}
                pg_cpu_dict = safe_get(
                    pg_container, "cpu", {}
                ) or {}
                
                app_cpu = safe_get(
                    app_cpu_dict, "percent_used"
                )
                pg_cpu = safe_get(
                    pg_cpu_dict, "percent_used"
                )
                
                logger.debug(
                    f"CPU Usage - "
                    f"App: {app_cpu}% | PG: {pg_cpu}%"
                )
            
            execution_plan = safe_get(
                response_data, "execution_plan", []
            ) or []

            if execution_plan and isinstance(
                execution_plan, list
            ):
                first_plan = (
                    execution_plan[0] 
                    if execution_plan else {}
                )
                if isinstance(first_plan, dict):
                    plan_data = safe_get(
                        first_plan, "Plan", {}
                    ) or {}
                    startup_cost = safe_get(
                        plan_data, "Startup Cost"
                    )
                    filter_condition = safe_get(
                        plan_data, "Filter", "None"
                    )
                    
                    logger.debug(
                        f"Startup Cost: {startup_cost} | "
                        f"Filter: {filter_condition}"
                    )

        except Exception as e:
            logger.debug(
                f"Failed to log debug details: "
                f"{str(e)}"
            )

    def _log_historical_response(
        self: Self,
        response_data: dict[str, Any]
    ) -> None:
        """Log historical statistics response data."""

        try:
            query_pattern = safe_get(
                response_data, "query_pattern", "N/A"
            )
            stats = safe_get(response_data, "stats")
            top_queries = safe_get(
                response_data, "top_queries"
            )
            
            logger.info(
                f"Historical Stats - Pattern: "
                f"{query_pattern}"
            )
            
            if stats and isinstance(stats, dict):
                query = safe_get(stats, "query", "N/A")
                calls = safe_get(stats, "calls", "N/A")
                total_time = safe_get(
                    stats, "total_exec_time", "N/A"
                )
                logger.info(
                    f"Stats found: {str(query)[:50]}..."
                )
                logger.info(
                    f"Calls: {calls} | "
                    f"Total Time: {total_time}"
                )

            elif (
                top_queries and 
                isinstance(top_queries, list)
            ):
                logger.info(
                    f"Top Queries: {len(top_queries)}"
                )
                for i, query_data in enumerate(
                    top_queries[:3]
                ):
                    if isinstance(query_data, dict):
                        query = safe_get(
                            query_data, "query", "N/A"
                        )
                        logger.info(
                            f"  {i+1}. "
                            f"{str(query)[:50]}..."
                        )
            else:
                logger.info("No historical data found")
                
        except Exception as e:
            logger.error(
                f"Failed to log historical response: "
                f"{str(e)}"
            )
