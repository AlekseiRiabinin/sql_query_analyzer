FROM postgres:17.5

COPY init-pg-stat.sql /docker-entrypoint-initdb.d/10-init-main.sql
COPY init-vtb-db.sql /docker-entrypoint-initdb.d/20-init-vtb.sql

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

EXPOSE 5432

CMD ["postgres", \
    "-c", "shared_preload_libraries=pg_stat_statements", \
    "-c", "pg_stat_statements.max=10000", \
    "-c", "pg_stat_statements.track=all"]


# docker build --no-cache -t alexflames77/custom-postgres-17:latest .
# docker push alexflames77/custom-postgres-17:latest


# docker run -d \
#   --name vtb_postgres_test \
#   -e POSTGRES_USER=postgres \
#   -e POSTGRES_PASSWORD=postgres \
#   -e POSTGRES_DB=vtb_db \
#   -p 5435:5432 \
#   alexflames77/custom-postgres-17:latest

# docker stop vtb_postgres_test && docker rm vtb_postgres_test

# -- Example table with ~1000 rows
# CREATE TABLE employees (
#     id SERIAL PRIMARY KEY,
#     name TEXT NOT NULL,
#     department TEXT NOT NULL,
#     salary NUMERIC(10,2) NOT NULL,
#     hire_date DATE NOT NULL DEFAULT CURRENT_DATE
# );

# -- Insert dummy data
# INSERT INTO employees (name, department, salary)
# SELECT
#     'Employee_' || g,
#     CASE WHEN g % 5 = 0 THEN 'Engineering'
#          WHEN g % 5 = 1 THEN 'HR'
#          WHEN g % 5 = 2 THEN 'Finance'
#          WHEN g % 5 = 3 THEN 'Sales'
#          ELSE 'Marketing'
#     END,
#     (random() * 1000)::NUMERIC(10,2)
# FROM generate_series(1, 10000) g;
