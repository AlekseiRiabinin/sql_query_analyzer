FROM postgres:17.5

# Copy initialization script
COPY init-pg-stat.sql /docker-entrypoint-initdb.d/

# Configure PostgreSQL with pg_stat_statements
# Use environment variables that can be set from docker-compose
RUN echo "shared_preload_libraries = 'pg_stat_statements'" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "pg_stat_statements.max = 10000" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "pg_stat_statements.track = all" >> /usr/share/postgresql/postgresql.conf.sample

# Create volume directory
RUN mkdir -p /var/lib/postgresql/data

# Health check (uses environment variables that will be available at runtime)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

EXPOSE 5432

# Start PostgreSQL with custom configurations
CMD ["postgres", \
    "-c", "shared_preload_libraries=pg_stat_statements", \
    "-c", "pg_stat_statements.max=10000", \
    "-c", "pg_stat_statements.track=all"]