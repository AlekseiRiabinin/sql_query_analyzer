"""Main module running FastAPI app."""


import os
import logging
import psycopg
import datetime
from logging.config import dictConfig
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Generator, AsyncGenerator
from database import get_db
# from query_analyzer import QueryAnalyzer
from query_analyzer import EnhancedQueryAnalyzer as QueryAnalyzer
from middleware import ResponseLoggingMiddleware


os.makedirs("/app/logs", exist_ok=True)


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': '/app/logs/query_analyzer.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'uvicorn': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.access': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.error': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)
logger.info("Application logging configured successfully")

app = FastAPI(
    title="Query Analysis API",
    version="1.0.0",
    description=(
        f"API for analyzing SQL query "
        f"performance with PostgreSQL"
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(ResponseLoggingMiddleware)


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

    logger.info("Root endpoint accessed")
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

    logger.info("Health check endpoint accessed")
    try:
        with analyzer.conn.cursor() as cur:
            cur.execute("SELECT 1")
            db_healthy = cur.fetchone()[0] == 1
        
        resource_monitor = analyzer._get_resource_monitor()
        monitor_healthy = resource_monitor is not None

        logger.info(
            f"Health check completed - DB: {db_healthy}, "
            f"Monitor: {monitor_healthy}"
        )

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
        logger.error(f"Health check failed: {e}")
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

    logger.info(
        f"Analyze query request received: "
        f"{query[:100]}..."
    )

    try:
        if not query or not query.strip():
            logger.warning("Empty query received")
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if len(query.strip()) > 10000:
            logger.warning("Query too long for analysis")
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

        logger.info(
            f"Query analysis completed - "
            f"Cost: {analysis_result.total_cost}, "
            f"Reject: {should_reject}"
        )

        return result_dict

    except ValueError as e:
        logger.error(f"Value error in query analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except psycopg.Error as e:
        logger.error(f"Database error in query analysis: {e}")
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
        logger.error(f"Unexpected error in query analysis: {e}")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {e}"
        )


@app.get("/cache/clear")
async def clear_cache(
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> dict:
    """Clear the query analysis cache."""

    logger.info("Cache clear request received")

    try:
        analyzer._historical_cache.clear()
        analyzer._cache_timestamps.clear()
        logger.info("Cache cleared successfully")

        return {
            "message": "Cache cleared successfully",
            "cache_size": 0,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {e}"
        )


@app.get("/cache/stats")
async def get_cache_stats(
    analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> dict:
    """Get cache statistics."""

    logger.info("Cache stats request received")

    return {
        "cache_size": len(analyzer._historical_cache),
        "cache_ttl": analyzer.cache_ttl,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test-logging")
async def test_logging():
    """Test if logging is working properly."""

    logger.info(
        "Test log message from /test-logging endpoint"
    )
    
    logger.debug(
        f"Debug message - should not appear "
        f"in file if level is INFO"
    )
    logger.info("Info message - should appear in file")
    logger.warning(
        "Warning message - should appear in file"
    )
    logger.error("Error message - should appear in file")
    
    return {
        "message": "Log test completed",
        "log_file": "/app/logs/query_analyzer.log"
    }

@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Exception handler for unhandled exceptions."""

    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI
) -> AsyncGenerator[None, None]:
    """Async context manager for lifecycle events.."""

    logger.info("Application startup initiated")

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
        conn.close()
        logger.info(f"Connected to PostgreSQL: {version}")

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
    
    yield  # Application runs here
    
    logger.info("Shutting down Query Analysis API")


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
