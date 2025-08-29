import psycopg
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from psycopg import sql
from database import get_db
from typing import Generator


app = FastAPI(title="Query Analysis API", version="1.0.0")


def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Database connection for each request."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


@app.get("/")
async def root():
    return {"message": "Query Analysis API is running"}


@app.get("/analysis/slow-queries")
async def get_slow_queries(conn: psycopg.Connection = Depends(get_connection)):
    """Get the slowest queries from query_performance view"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT query, calls, total_exec_time, mean_exec_time
                FROM query_performance 
                ORDER BY mean_exec_time DESC 
                LIMIT 10
            """)
            results = cur.fetchall()
            
            # Convert to list of dictionaries for JSON response
            slow_queries = []
            for row in results:
                slow_queries.append({
                    "query": row[0],
                    "calls": row[1],
                    "total_exec_time": float(row[2]),
                    "mean_exec_time": float(row[3])
                })
            
            return {"slow_queries": slow_queries}
            
    except psycopg.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/analysis/query-stats")
async def get_query_statistics(conn: psycopg.Connection = Depends(get_connection)):
    """Get aggregated query statistics"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(calls) as total_executions,
                    AVG(mean_exec_time) as avg_exec_time,
                    MAX(mean_exec_time) as max_exec_time
                FROM query_performance
            """)
            stats = cur.fetchone()
            
            return {
                "total_queries": stats[0],
                "total_executions": stats[1],
                "avg_exec_time": float(stats[2]) if stats[2] else 0,
                "max_exec_time": float(stats[3]) if stats[3] else 0
            }
            
    except psycopg.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/analysis/query/{query_hash}")
async def get_query_details(query_hash: str, conn: psycopg.Connection = Depends(get_connection)):
    """Get detailed information about a specific query"""
    try:
        with conn.cursor() as cur:
            # Note: You might want to query by queryid instead of hash
            cur.execute("""
                SELECT *
                FROM query_performance 
                WHERE query LIKE %s
                LIMIT 1
            """, (f"%{query_hash}%",))
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Query not found")
            
            # Get column names
            colnames = [desc[0] for desc in cur.description]
            
            # Create response dictionary
            query_details = {}
            for i, value in enumerate(result):
                query_details[colnames[i]] = value
            
            return query_details
            
    except psycopg.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
