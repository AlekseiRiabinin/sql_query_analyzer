"""
Centralized constants for SQL Query Analyzer application.
All configuration thresholds and defaults are defined here.
"""


from typing import Final


# Query Analysis Constants
QUERY_LENGTH_LIMIT: Final[int] = 10000
BASE_MEMORY_BYTES: Final[int] = 1024 * 1024                # 1MB
BASE_CPU_SECONDS: Final[float] = 0.001                     # 1ms
COST_TO_CPU_FACTOR: Final[float] = 0.0001

# Performance Thresholds
MEMORY_THRESHOLD_MEDIUM: Final[int] = 10                   # 10MB
MEMORY_THRESHOLD_HIGH: Final[int] = 50                     # 50MB
CPU_THRESHOLD_MEDIUM: Final[int] = 100                     # 100ms
CPU_THRESHOLD_HIGH: Final[int] = 500                       # 500ms

# Table Size Thresholds
LARGE_TABLE_THRESHOLD: Final[int] = 1024 * 1024 * 100      # 100MB
VERY_LARGE_TABLE_THRESHOLD: Final[int] = 1024 * 1024 * 10  # 10MB

# Query Processing Thresholds
SORT_THRESHOLD: Final[int] = 10000
NESTED_LOOP_THRESHOLD: Final[int] = 1000
APPROXIMATE_COUNT_THRESHOLD: Final[int] = 5000
COVERING_INDEX_THRESHOLD: Final[int] = 5                   # >5 columns

# Database Metrics Thresholds
CACHE_HIT_THRESHOLD: Final[int] = 90                       # 90%
CONNECTION_THRESHOLD: Final[int] = 80                      # 80%
DISK_WRITE_THRESHOLD: Final[int] = 50 * 1024 * 1024        # 50MB/s
DISK_IOPS_THRESHOLD: Final[int] = 1000
MAX_QUERY_COST: Final[float] = 10000

# System Resource Thresholds (Percentage-based)
MEMORY_CRITICAL_THRESHOLD: Final[int] = 85                 # 85% memory usage
CPU_CRITICAL_THRESHOLD: Final[int] = 90                    # 90% CPU usage
APP_MEMORY_HIGH_THRESHOLD: Final[int] = 90                 # 90% app memory usage
APP_MEMORY_PRESSURE_THRESHOLD: Final[int] = 80             # 80% memory usage
APP_CPU_HIGH_THRESHOLD: Final[int] = 70                    # 70% CPU usage

# Cache Settings
CACHE_TTL: Final[int] = 300                                # 5 minutes TTL for cache

# Environment-specific defaults
class Defaults:
    """
    Default values that can be overridden 
    by environment or configuration.
    """
    
    # Query Analyzer defaults
    QUERY_LENGTH_LIMIT = QUERY_LENGTH_LIMIT
    BASE_MEMORY_BYTES = BASE_MEMORY_BYTES
    BASE_CPU_SECONDS = BASE_CPU_SECONDS
    COST_TO_CPU_FACTOR = COST_TO_CPU_FACTOR
    MEMORY_THRESHOLD_MEDIUM = MEMORY_THRESHOLD_MEDIUM
    MEMORY_THRESHOLD_HIGH = MEMORY_THRESHOLD_HIGH
    CPU_THRESHOLD_MEDIUM = CPU_THRESHOLD_MEDIUM
    CPU_THRESHOLD_HIGH = CPU_THRESHOLD_HIGH
    SORT_THRESHOLD = SORT_THRESHOLD
    CACHE_HIT_THRESHOLD = CACHE_HIT_THRESHOLD
    CONNECTION_THRESHOLD = CONNECTION_THRESHOLD
    DISK_WRITE_THRESHOLD = DISK_WRITE_THRESHOLD
    DISK_IOPS_THRESHOLD = DISK_IOPS_THRESHOLD
    MAX_QUERY_COST = MAX_QUERY_COST
    MEMORY_CRITICAL_THRESHOLD = MEMORY_CRITICAL_THRESHOLD
    CPU_CRITICAL_THRESHOLD = CPU_CRITICAL_THRESHOLD
    APP_MEMORY_HIGH_THRESHOLD = APP_MEMORY_HIGH_THRESHOLD
    APP_MEMORY_PRESSURE_THRESHOLD = APP_MEMORY_PRESSURE_THRESHOLD
    APP_CPU_HIGH_THRESHOLD = APP_CPU_HIGH_THRESHOLD
    LARGE_TABLE_THRESHOLD = LARGE_TABLE_THRESHOLD

    # Advanced Analyzer defaults
    VERY_LARGE_TABLE_THRESHOLD = VERY_LARGE_TABLE_THRESHOLD
    NESTED_LOOP_THRESHOLD = NESTED_LOOP_THRESHOLD
    APPROXIMATE_COUNT_THRESHOLD = APPROXIMATE_COUNT_THRESHOLD
    COVERING_INDEX_THRESHOLD = COVERING_INDEX_THRESHOLD

    # Cache defaults
    CACHE_TTL = CACHE_TTL
