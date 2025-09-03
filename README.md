# SQL Query Analyzer

A sophisticated performance analysis tool for PostgreSQL queries that combines execution plan analysis with cross-container resource monitoring to provide comprehensive performance recommendations.

## 🚀 Features

- **Execution Plan Analysis**: Detailed `EXPLAIN` analysis with cost estimation
- **Resource Prediction**: Empirical-based memory and CPU usage prediction
- **Cross-Container Monitoring**: Real-time Docker container resource utilization
- **Intelligent Recommendations**: AI-driven performance optimization suggestions (* In future release)
- **Historical Context**: Integration with `pg_stat_statements` for trend analysis
- **Safety Features**: Query validation and timeout protection
- **Comprehensive Logging**: Structured logging of queries, execution plans, resource usage, and errors
- **CI/CD Integration**: Automated build, test, and API validation workflow using GitHub Actions

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
├── .github/
│   └── workflows/
│       └── ci-cd.yml 
├── docker-compose.yml
├── postgres/
│   ├── Dockerfile
│   ├── init-pg-stat.sql
│   └── init-vtb-db.sql
├── app/
│   ├── Dockerfile
│   ├── __init__.py
│   ├── main.py
│   ├── middleware.py
│   ├── database.py
│   ├── query_analyzer.py
│   ├── pg_feature_extractor.py
│   ├── resource_monitor.py
├── logs/    
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
CREATE TABLE vtb_db.public.employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary NUMERIC(10,2) NOT NULL,
    hire_date DATE NOT NULL DEFAULT CURRENT_DATE
);

-- Insert dummy data
INSERT INTO vtb_db.public.employees (name, department, salary)
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
        "container_resources": {  },
        "postgres_metrics": {  },
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
    "recommendations": [  ],
    "historical_context": null,
    "execution_plan": [  ],
    "timestamp": "2025-08-31T11:26:55.139418",
    "should_reject": false
}
```

4. **Cache Management**
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

## 📝 Logging & Monitoring

The application includes comprehensive logging middleware that automatically captures and logs detailed performance metrics for all query analysis requests. The logging system:
Features:

- **Automatic Response Logging:** Captures analysis results from `/analyze` endpoint

- **Structured Logging:** Logs key performance metrics including:

  - Query text and timestamp

  - Execution plan details (cost, node type, relation names)

  - Buffer cache statistics and hit ratios

  - Optimization recommendations

  - Resource predictions (memory, CPU)

- **Debug-Level Details:** Additional debug information including:

  - Query characteristics (joins, sorts, aggregates)

  - Container resource utilization

  - Execution plan startup costs and filter conditions

- **Error Handling:** Robust exception handling with meaningful error messages

*Example:*

```
INFO: Analysis Request: http://localhost:8000/analyze
INFO: Query: SELECT * FROM employees WHERE department = 'Engineering'...
INFO: Timestamp: 2025-09-02T14:26:21.534385
INFO: Total Cost: 323.66
INFO: Node Type: Sort
INFO: Relation: employees
INFO: Plan Rows: 2000
INFO: Buffer Hits: 0 | Reads: 0
INFO: Optimization Recommendations: 0
```

*Configuration:*

- Logs are written to both console and file storage

- Debug logging can be enabled for detailed technical information

- Log files are persisted in the `app_logs` Docker volume

- Middleware automatically handles JSON parsing and error scenarios


## 🤖 CI/CD Pipeline

The project includes an automated CI/CD workflow using GitHub Actions to ensure code quality, deploy Docker containers, and validate functionality of the SQL Query Analyzer.

**Workflow Overview**

The CI/CD workflow is defined in `.github/workflows/ci-cd.yml` and runs on push or pull_request events targeting the main or master branches. It consists of the following steps:

1. **Checkout Code:** Pulls the latest repository code.

2. **Setup Docker Buildx:** Prepares Docker Buildx to build multi-platform images.

3. **Build and Start Services:** Uses `docker compose up --build -d` to build and start the PostgreSQL and API containers.

4. **Generate Test Data:** Creates an employees table and populates it with 1,000 rows of sample data for testing.

5. **Verify Test Data:** Confirms that data was inserted correctly with a `SELECT COUNT(*)` query.

6. **API Endpoint Testing:** Tests the `/analyze` endpoint with multiple queries, including basic `SELECT`, `WHERE`, and `ORDER BY` queries.

7. **Comprehensive API Validation:** Verifies API health and parses JSON responses from `/analyze` to ensure functionality.

8. **Failure Debugging:** Automatically outputs PostgreSQL logs if any step fails, aiding in rapid troubleshooting.


## 📚 References
1. Mason, K. et al. (2018). Predicting host CPU utilization in the cloud using evolutionary neural networks. Future Generation Computer Systems.

2. Liu, X. (2024). Towards CPU Performance Prediction: New Challenge Benchmark Dataset and Novel Approach.

3. Gunther, N. Analyzing Computer Performance with Perl::PDQ.

4. Sites, R. Understanding Software Dynamics.

5. Gregg, B. Systems Performance: Enterprise and the Cloud.
