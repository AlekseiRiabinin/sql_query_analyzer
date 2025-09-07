# SQL Query Analyzer

A sophisticated performance analysis tool for PostgreSQL queries that combines execution plan analysis with cross-container resource monitoring to provide comprehensive performance recommendations.

## 🚀 Features

- **Execution Plan Analysis**: Detailed `EXPLAIN` analysis with cost estimation
- **Resource Prediction**: Empirical-based memory and CPU usage prediction
- **Cross-Container Monitoring**: Real-time Docker container resource utilization
- **Intelligent Recommendations**: Performance optimization suggestions
- **Historical Context**: Integration with `pg_stat_statements` for trend analysis
- **Safety Features**: Query validation and timeout protection
- **Comprehensive Logging**: Structured logging of queries, execution plans, resource usage, and errors
- **CI/CD Integration**: Automated build, test, and API validation workflow using GitHub Actions

## 📊 Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        VTB Infrastructure                            ║
╠═══════════════════════════════════════════════════╦══════════════════╣
║                FastAPI Microservice               ║    PostgreSQL    ║
║                                                   ║     Database     ║
║  ┌─────────────────────────────────────────────┐  ║                  ║
║  │            Resource Monitor                 │  ║                  ║
║  │          (Docker API, psutil)               │  ║                  ║
║  └─────────────────────────────────────────────┘  ║                  ║
║                                                   ║                  ║
║  ┌─────────────────────────────────────────────┐  ║  ┌─────────────┐ ║
║  │              Query Analyzer                 │  ║  │ PostgreSQL  │ ║
║  │                                             │  ║  │  Process    │ ║
║  └─────────────────────────────────────────────┘  ║  └─────────────┘ ║
║                                                   ║                  ║
╚═══════════════════════════════════════════════════╩══════════════════╝
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
│   ├── advanced_analyzer.py
│   ├── pg_feature_extractor.py
│   ├── resource_monitor.py
│   ├── constants.py
├── logs/    
├── requirements.txt
├── .env
└── README.md

```
## ⚙️ Tech stack
- Python 3.12
- PostgreSQL 17.5
- Docker
- FastAPI
- Git
- GitHub Actions
- VS Code
- Postman
- DBeaver

## 📚 PostgreSQL Processes and System Views

| Process/View Name | Type | Description |
|-------------------|------|-------------|
| **pg_stat_statements** | Extension | Tracks execution statistics for all SQL statements executed by the server (requires installation) |
| **pg_stat_activity** | System View | Shows one row per server process with details about current activity and queries |
| **pg_stat_user_tables** | System View | Contains statistics about accesses to each user table, including sequential scans, index scans, and tuple information |
| **pg_stats** | System View | Provides access to per-column statistics about table contents (null fractions, distinct values, etc.) |
| **pg_indexes** | System View | Contains information about all indexes in the database |
| **pg_index** | System Catalog | Stores index information including uniqueness and maintenance data |
| **pg_locks** | System View | Shows the locks currently held or awaited by open transactions |
| **pg_stat_all_tables** | System View | Contains statistics about accesses to each table in the database |
| **pg_stat_all_indexes** | System View | Contains statistics about accesses to specific indexes |
| **pg_stat_wal** | System View | Provides statistics about Write-Ahead Log (WAL) usage and activity |
| **pg_class** | System Catalog | Stores information about tables, indexes, and other relations |
| **information_schema.columns** | System View | Standard SQL view showing column information for all tables |
| **information_schema.table_constraints** | System View | Shows constraint information including foreign keys and primary keys |
| **information_schema.key_column_usage** | System View | Identifies all columns that are constrained as keys |
| **pg_total_relation_size()** | Function | Returns total disk space used by a table including indexes and toasted data |
| **pg_size_pretty()** | Function | Formats a size in bytes into human-readable format (KB, MB, GB, etc.) |

**Required PostgreSQL Configuration**

For full functionality, ensure these settings are enabled in `postgresql.conf`:
- `shared_preload_libraries = 'pg_stat_statements'`
- `track_activities = on`
- `track_counts = on`
- `pg_stat_statements.track = all`

## 🔗 QueryAnalyzer Constants

| Constant | Default Value | Description | Category |
|----------|---------------|-------------|----------|
| `QUERY_LENGTH_LIMIT` | 10000 | Maximum allowed SQL query length in characters | Query Processing |
| `BASE_MEMORY_BYTES` | 1MB | Base memory overhead for query execution | Resource Prediction |
| `BASE_CPU_SECONDS` | 1ms | Base CPU time for query planning and setup | Resource Prediction |
| `COST_TO_CPU_FACTOR` | 0.0001 | Conversion factor from PostgreSQL cost units to CPU seconds | Resource Prediction |
| `MEMORY_THRESHOLD_MEDIUM` | 10MB | Warning threshold for predicted memory usage | Memory Thresholds |
| `MEMORY_THRESHOLD_HIGH` | 50MB | Critical threshold for predicted memory usage | Memory Thresholds |
| `CPU_THRESHOLD_MEDIUM` | 100ms | Warning threshold for predicted CPU time | CPU Thresholds |
| `CPU_THRESHOLD_HIGH` | 500ms | Critical threshold for predicted CPU time | CPU Thresholds |
| `LARGE_TABLE_THRESHOLD` | 100MB | Table size threshold for "large" classification | Table Size |
| `SORT_THRESHOLD` | 10000 | Row count threshold for expensive sort warnings | Performance |
| `CACHE_HIT_THRESHOLD` | 90% | Minimum acceptable buffer cache hit ratio | Performance |
| `CONNECTION_THRESHOLD` | 80% | Maximum connection pool usage before warning | Performance |
| `DISK_WRITE_THRESHOLD` | 50MB/s | Disk write throughput for I/O pressure detection | I/O Monitoring |
| `DISK_IOPS_THRESHOLD` | 1000 | Disk IOPS threshold for high I/O detection | I/O Monitoring |
| `MAX_QUERY_COST` | 10000 | Maximum allowed PostgreSQL query cost for rejection | Query Cost |
| `MEMORY_CRITICAL_THRESHOLD` | 85% | PostgreSQL container memory critical level | Container Resources |
| `CPU_CRITICAL_THRESHOLD` | 90% | PostgreSQL container CPU critical level | Container Resources |
| `APP_MEMORY_HIGH_THRESHOLD` | 90% | Application container memory high usage level | Container Resources |
| `APP_MEMORY_PRESSURE_THRESHOLD` | 80% | Application container memory pressure level | Container Resources |
| `APP_CPU_HIGH_THRESHOLD` | 70% | Application container CPU high usage level | Container Resources |

## ⚡ AdvancedQueryAnalyzer Constants

| Constant | Default Value | Description | Category |
|----------|---------------|-------------|----------|
| `LARGE_TABLE_THRESHOLD` | 1MB | Table size for "large" classification (~1K-10K rows) | Table Size |
| `VERY_LARGE_TABLE_THRESHOLD` | 10MB | Table size for "very large" classification (~100K+ rows) | Table Size |
| `NESTED_LOOP_THRESHOLD` | 1000 | Maximum efficient row count for nested loop joins | Join Operations |
| `APPROXIMATE_COUNT_THRESHOLD` | 5000 | Row count threshold for approximate counting | Aggregation |
| `COVERING_INDEX_THRESHOLD` | 5 | Minimum columns for covering index recommendations | Indexing |

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


## 🎓 References
1. Mason, K. et al. (2018). Predicting host CPU utilization in the cloud using evolutionary neural networks. Future Generation Computer Systems.

2. Liu, X. (2024). Towards CPU Performance Prediction: New Challenge Benchmark Dataset and Novel Approach.

3. Gunther, N. Analyzing Computer Performance with Perl::PDQ.

4. Sites, R. Understanding Software Dynamics.

5. Gregg, B. Systems Performance: Enterprise and the Cloud.
