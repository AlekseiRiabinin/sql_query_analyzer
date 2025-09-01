# SQL Query Analyzer

A sophisticated performance analysis tool for PostgreSQL queries that combines execution plan analysis with cross-container resource monitoring to provide comprehensive performance recommendations.

## 🚀 Features

- **Execution Plan Analysis**: Detailed EXPLAIN analysis with cost estimation
- **Resource Prediction**: Empirical-based memory and CPU usage prediction
- **Cross-Container Monitoring**: Real-time Docker container resource utilization
- **Intelligent Recommendations**: AI-driven performance optimization suggestions (* In future release)
- **Historical Context**: Integration with `pg_stat_statements` for trend analysis
- **Safety Features**: Query validation and timeout protection

## 📊 Architecture

```
+-------------------------------------------------------+
|                  Docker Host Machine                  |
| +-----------------------------+ +-------------------+ |
| |     FastAPI App Container   | |  PostgreSQL       | |
| |                             | |  Container        | |
| |  +-----------------------+  | |                   | |
| |  | Resource Monitor      |  | |                   | |
| |  | (Docker API, psutil)  |  | |                   | |
| |  +-----------------------+  | |                   | |
| |                             | |                   | |
| |  +-----------------------+  | | +---------------+ | |
| |  | Query Analyzer        |  | | | PostgreSQL    | | |
| |  |                       |  | | | Process       | | |
| |  +-----------------------+  | | +---------------+ | |
| +-----------------------------+ +-------------------+ |
+-------------------------------------------------------+
```

## 🏗️ Project Structure

```
sql_query-analyzer/
├── docker-compose.yml
├── postgres/
│ ├── Dockerfile
│ ├── init-pg-stat.sql
│ └── init-vtb-db.sql
├── app/
│ ├── Dockerfile
│ ├── \_\_init\_\_.py
│ ├── main.py
│ ├── database.py
│ ├── query_analyzer.py
│ ├── pg_feature_extractor.py
│ ├── resource_monitor.py
├── requirements.txt
├── .env
└── README.md

```
## ⚙️ Teck stack
- Python 3.12
- PostgreSQL 17.5
- Docker
- FastAPI
- Git
- GitHub Actions
- VS Code
- Postman
- DBeaver

## 🛠️ Installation

### Prerequisites
- Linux (Ubuntu)
- Docker and Docker Compose

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/AlekseiRiabinin/sql_query_analyzer.git
cd sql_query_analyzer
```

2. **Build and start containers**
```bash
docker-compose up -d --build
```

3. **Access the API**
```bash
curl http://localhost:8000
```

## 🐘 Generate data
```sql
-- Example table with ~1,000 rows
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary NUMERIC(10,2) NOT NULL,
    hire_date DATE NOT NULL DEFAULT CURRENT_DATE
);

-- Insert dummy data
INSERT INTO employees (name, department, salary)
SELECT
    'Employee_' || g,
    CASE WHEN g % 5 = 0 THEN 'Engineering'
         WHEN g % 5 = 1 THEN 'HR'
         WHEN g % 5 = 2 THEN 'Finance'
         WHEN g % 5 = 3 THEN 'Sales'
         ELSE 'Marketing'
    END,
    (random() * 1000)::NUMERIC(10,2)
FROM generate_series(1, 10000) g;
```

## 🌐 API Routes

1. **Root Endpoint**

```bash
GET / - API information and available endpoints
```

```json
{
    "message": "Query Analysis API",
    "version": "1.0.0",
    "endpoints": {
        "analyze": "POST /analyze",
        "health": "GET /health",
        "stats": "GET /stats"
    }
}
```

2. **Health Check**
```bash
GET /health - System health status
```

```json
{
    "status": "healthy",
    "database": "connected",
    "resource_monitor": "available",
    "timestamp": "2025-09-01T10:30:00.000000"
}
```

3. **Query Analysis**
```bash
POST /analyze - Analyze SQL query performance
```

*Parameters:*

- `query` (required): SQL query to analyze
- `include_resources` (optional, default=`true`): Include container resource metrics
- `include_historical` (optional, default=`true`): Include historical performance data
- `max_cost_threshold` (optional, default=`10000`): Maximum allowed query cost

*Example:*

```bash
curl -X POST "http://localhost:8000/analyze?query=SELECT+*+FROM+employees&include_resources=true&include_historical=true"
```

```json
{
    "query": "SELECT * FROM employees WHERE department = 'Engineering'",
    "performance_metrics": {
        "total_cost": 209.0,
        "plan_rows": 2000,
        "node_type": "Seq Scan",
        "relation_name": "employees",
        "shared_buffers_hit": 0,
        "shared_buffers_read": 0,
        "plan_width": 34,
        "planning_time": null,
        "execution_time": null
    },
    "resource_metrics": {
        "container_resources": {....},
        "postgres_metrics": {....},
        "predicted_memory_bytes": 1116576,
        "predicted_cpu_seconds": 0.0219
    },
    "query_characteristics": {
        "query_type": "SELECT",
        "contains_join": false,
        "contains_sort": false,
        "contains_aggregate": false,
        "is_read_only": true
    },
    "recommendations": [....],
    "historical_context": null,
    "execution_plan": [....],
    "timestamp": "2025-08-31T11:26:55.139418",
    "should_reject": false
}
```

4. **Historical Statistics**
```bash
GET /stats/historical - Get historical query performance data
```

*Parameters:*

- `query_pattern` (optional): Query pattern to match

- `limit` (optional, default=10): Number of results to return


```json
{
    "query_pattern": "SELECT",
    "stats": {
        "query": "SELECT * FROM employees WHERE department = $1",
        "calls": 150,
        "total_exec_time": 1250.5,
        "mean_exec_time": 8.34,
        "rows": 15000,
        "shared_blks_hit": 12000,
        "shared_blks_read": 300,
        "cache_hit_ratio": 97.56
    }
}
```

5. **System Statistics**
```bash
GET /stats/system - Get current system resource usage
```

```json
{
    "available": true,
    "resources": {
        "application_container": {
        "cpu": {
            "percent_used": 0.0,
            "cores_available": 8
        },
        "memory": {
            "used_bytes": 46215168,
            "limit_bytes": 15975219200,
            "percent_used": 0.29
        },
        "network": {...},
        "disk_io": {...},
        "status": "unknown",
        "container_id": "bb54b17ca492",
        "container_type": "API Container"
        },
        "postgres_container": {...},
        "timestamp": 1756639615.057929,
        "system_wide": {
        "load_average": {
            "1min": 0.84326171875,
            "5min": 0.57568359375,
            "15min": 0.35009765625
        }
        }
    },
    "timestamp": "2024-01-15T10:30:00.000000"
}
```

6. **Cache Management**
```bash
GET /cache/clear - Clear query analysis cache
```

```json
{
    "message": "Cache cleared successfully",
    "cache_size": 0,
    "timestamp": "2024-01-15T10:30:00.000000"
}
```

```bash
GET /cache/stats - Get cache statistics
```

```json
{
    "cache_size": 15,
    "cache_ttl": 300,
    "timestamp": "2024-01-15T10:30:00.000000"
}
```
