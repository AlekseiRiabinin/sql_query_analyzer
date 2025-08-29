CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

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

GRANT pg_read_all_stats TO postgres;
GRANT SELECT ON pg_stat_statements TO postgres;
GRANT SELECT ON pg_stat_activity TO postgres;
GRANT SELECT ON pg_stats TO postgres;
GRANT SELECT ON pg_locks TO postgres;
