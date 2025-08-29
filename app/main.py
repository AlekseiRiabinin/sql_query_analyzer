"""Main module running FastAPI app."""


import psycopg
import datetime
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Generator, Optional
from database import get_db
from query_analyzer import QueryAnalyzer


app = FastAPI(
    title="Query Analysis API",
    version="1.0.0",
    description=(
        f"API for analyzing SQL query "
        f"performance with PostgreSQL"
    )
)


def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Database connection for each request."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


def get_query_analyzer(
        conn: psycopg.Connection = Depends(get_connection)
) -> QueryAnalyzer:
    """Get QueryAnalyzer instance for each request."""

    return QueryAnalyzer(conn)


@app.get("/")
async def root():
    """Root endpoint with API information."""

    return {
        "message": "Query Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


@app.get("/health")
async def health_check(
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
):
    """Health check endpoint."""

    try:
        with analyzer.conn.cursor() as cur:
            cur.execute("SELECT 1")
            db_healthy = cur.fetchone()[0] == 1
        
        resource_monitor = analyzer._get_resource_monitor()
        monitor_healthy = resource_monitor is not None
        
        return {
            "status": "healthy",
            "database": (
                "connected" 
                if db_healthy 
                else "disconnected"
            ),
            "resource_monitor": (
                "available" 
                if monitor_healthy 
                else "unavailable"
            ),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {e}"
        )


@app.post("/analyze")
async def analyze_query(
    query: str = Query(
        ...,
        description="SQL query to analyze"
    ),
    include_resources: bool = Query(
        True,
        description="Include container resource metrics"
    ),
    include_historical: bool = Query(
        True,
        description="Include historical performance data"
    ),
    max_cost_threshold: float = Query(
        10000,
        description="Maximum allowed query cost"
    ),
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> dict:
    """Analyze SQL query and give performance recommendations."""

    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if len(query.strip()) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Query is too long for analysis"
            )
        
        analysis_result = analyzer.analyze_query(
            sql_query=query,
            include_resources=include_resources,
            include_historical=include_historical
        )
        
        should_reject = analyzer.should_reject_query(
            analysis_result, max_cost_threshold
        )
        
        result_dict = analysis_result.to_dict()
        result_dict["should_reject"] = should_reject
        
        if should_reject:
            result_dict["rejection_reason"] = (
                f"Query cost {analysis_result.total_cost} "
                f"exceeds threshold {max_cost_threshold}"
            )

        return result_dict
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except psycopg.Error as e:
        if "statement timeout" in str(e):
            raise HTTPException(
                status_code=408, 
                detail=(
                    f"Query analysis timed out. "
                    f"The query may be too complex."
                )
            )
        raise HTTPException(
            status_code=500, detail=f"Database error: {e}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {e}"
        )


@app.get("/stats/historical")
async def get_historical_stats(
    query_pattern: Optional[str] = Query(
        None,
        description="Query pattern to match"
    ),
    limit: int = Query(
        10,
        description="Number of results to return"
    ),
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> dict:
    """Get historical query performance statistics."""

    try:
        if query_pattern:
            stats = analyzer._get_historical_query_stats(
                query_pattern
            )
            return {
                "query_pattern": query_pattern,
                "stats": stats
            }
        else:
            with analyzer.conn.cursor() as cur:
                cur.execute("""
                    SELECT query, calls, total_exec_time, mean_exec_time
                    FROM pg_stat_statements 
                    ORDER BY total_exec_time DESC 
                    LIMIT %s
                """, (limit,))
                results = cur.fetchall()
                
                queries = []
                for row in results:
                    queries.append({
                        "query": row[0],
                        "calls": row[1],
                        "total_exec_time": float(row[2]),
                        "mean_exec_time": float(row[3])
                    })
                
                return {"top_queries": queries}
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get historical stats: {e}"
        )


@app.get("/stats/system")
async def get_system_stats(
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> dict:
    """Get current system resource statistics."""

    try:
        monitor = analyzer._get_resource_monitor()
        if not monitor:
            return {
                "available": False,
                "message": "Resource monitor not available"
            }

        resources = monitor.get_all_container_resources()
        return {
            "available": True,
            "resources": resources,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system stats: {e}"
        )


@app.get("/cache/clear")
async def clear_cache(
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> dict:
    """Clear the query analysis cache."""

    try:
        analyzer._historical_cache.clear()
        analyzer._cache_timestamps.clear()
        return {
            "message": "Cache cleared successfully",
            "cache_size": 0,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {e}"
        )


@app.get("/cache/stats")
async def get_cache_stats(
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> dict:
    """Get cache statistics."""

    return {
        "cache_size": len(analyzer._historical_cache),
        "cache_ttl": analyzer.cache_ttl,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
        conn.close()
        print(f"Connected to PostgreSQL: {version}")

    except Exception as e:
        print(f"Database connection failed: {e}")
        raise
    
    yield
    
    print("Shutting down Query Analysis API")


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
