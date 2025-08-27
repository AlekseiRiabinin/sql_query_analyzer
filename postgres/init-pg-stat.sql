-- Ensure vtb_db database exists
DO $$
BEGIN
    -- Check if vtb_db exists by attempting to connect to it
    IF NOT EXISTS (
        SELECT 1 FROM pg_database WHERE datname = 'vtb_db'
    ) THEN
        -- Create the database if it doesn't exist
        PERFORM dblink_exec('dbname=' || current_database(), 'CREATE DATABASE vtb_db');
        RAISE NOTICE 'Database vtb_db created successfully';
    ELSE
        RAISE NOTICE 'Database vtb_db already exists';
    END IF;
EXCEPTION
    WHEN others THEN
        RAISE WARNING 'Failed to create database vtb_db: %', SQLERRM;
END
$$;

-- Switch to vtb_db context to create extensions there
\c vtb_db

-- Enable pg_stat_statements extension for query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create a dedicated user for monitoring
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'monitor') THEN
        CREATE ROLE monitor 
        WITH LOGIN PASSWORD 'monitor_password' 
        NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
END
$$;

-- Grant necessary permissions to monitor user
GRANT pg_monitor TO monitor;
GRANT SELECT ON pg_stat_statements TO monitor;

-- Create a view for easier query performance analysis
CREATE OR REPLACE VIEW public.query_performance AS
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    rows,
    shared_blks_hit,
    shared_blks_read,
    shared_blks_dirtied,
    shared_blks_written,
    local_blks_hit,
    local_blks_read,
    local_blks_dirtied,
    local_blks_written,
    temp_blks_read,
    temp_blks_written,
    blk_read_time,
    blk_write_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC;

-- Grant read access to monitor user
GRANT SELECT ON public.query_performance TO monitor;

-- Reset statistics to start with clean state
SELECT pg_stat_statements_reset();

-- Log extension creation for debugging
DO $$
BEGIN
    RAISE NOTICE 'pg_stat_statements extension initialized successfully';
EXCEPTION
    WHEN others THEN
        RAISE WARNING 'Failed to initialize pg_stat_statements: %', SQLERRM;
END
$$;
