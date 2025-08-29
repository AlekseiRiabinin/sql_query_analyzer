-- Create the vtb_db database if it doesn't exist.
SELECT 'CREATE DATABASE vtb_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'vtb_db')\gexec

-- Enable the extension in the default postgres database
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
