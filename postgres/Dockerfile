FROM postgres:17.5

COPY init-pg-stat.sql /docker-entrypoint-initdb.d/10-init-main.sql
COPY init-vtb-db.sql /docker-entrypoint-initdb.d/20-init-vtb.sql

RUN mkdir -p /var/lib/postgresql/data

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

EXPOSE 5432

CMD ["postgres", \
    "-c", "shared_preload_libraries=pg_stat_statements", \
    "-c", "pg_stat_statements.max=10000", \
    "-c", "pg_stat_statements.track=all"]
