"""Resource monitor of Docker containers."""


import os
import time
import psutil
import docker
from docker import errors
from typing import (
    Self, Any, Optional,
    TypedDict, NotRequired, cast
)
from dataclasses import dataclass


class NetworkStats(TypedDict):
    rx_bytes: int
    tx_bytes: int
    rx_dropped: NotRequired[int]
    tx_dropped: NotRequired[int]
    rx_errors: NotRequired[int]
    tx_errors: NotRequired[int]


class CPUUsage(TypedDict):
    total_usage: int
    percpu_usage: NotRequired[list[int]]


class CPUStats(TypedDict):
    cpu_usage: CPUUsage
    system_cpu_usage: NotRequired[int]


class DockerStats(TypedDict):
    networks: dict[str, NetworkStats]
    cpu_stats: CPUStats
    precpu_stats: CPUStats
    memory_stats: dict
    networks: dict
    blkio_stats: dict
    status: NotRequired[str]
    id: NotRequired[str]


@dataclass
class ContainerStats:
    """Unified container statistics."""
    container_name: str
    cpu_percent: float
    memory_used_bytes: int
    memory_limit_bytes: int
    memory_percent: float
    network_rx_bytes: int
    network_tx_bytes: int
    disk_read_bytes: int
    disk_write_bytes: int
    timestamp: float


class ContainersResourceMonitor:
    """Monitors system resource usage."""
    
    def __init__(
        self: Self,
        postgres_container_name: str = "vtb_postgres",
        api_container_name: str = "vtb_api"
    ) -> None:
        self.docker_client = docker.from_env()
        self.postgres_container_name = postgres_container_name
        self.api_container_name = api_container_name
        self.previous_cpu_stats = {}
        self.previous_network_stats = {}
        
    def get_all_container_resources(self: Self) -> dict[str, Any]:
        """Get resource metrics for FastAPI app and PostgreSQL."""

        try:
            app_stats = self.get_api_container_resources()
            postgres_stats = (
                self.get_postgres_container_resources()
            )
            
            return {
                "application_container": (
                    app_stats or self._get_host_fallback()
                ),
                "postgres_container": postgres_stats,
                "timestamp": time.time(),
                "system_wide": self.get_system_wide_metrics()
            }
            
        except Exception as e:
            print(f"Error getting cross-container metrics: {e}")
            return {}
    
    def get_application_resources(self: Self) -> dict[str, Any]:
        """Get resource metrics for application container."""

        container_stats = self.get_api_container_resources()
        if container_stats:
            return container_stats
        
        return self._get_host_fallback()
    
    def _get_host_fallback(self: Self) -> dict[str, Any]:
        """Fallback to host metrics if Docker isn't available."""

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage('/')
            
            net_io = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'percent_used': cpu_percent,
                    'logical_cores': cpu_count,
                    'load_average': (
                        os.getloadavg() 
                        if hasattr(os, 'getloadavg') 
                        else None
                    )
                },
                'memory': {
                    'total_bytes': memory.total,
                    'available_bytes': memory.available,
                    'percent_used': memory.percent,
                    'used_bytes': memory.used,
                    'free_bytes': memory.free
                },
                'swap': {
                    'total_bytes': swap.total,
                    'used_bytes': swap.used,
                    'free_bytes': swap.free,
                    'percent_used': swap.percent
                },
                'disk': {
                    'total_bytes': disk_usage.total,
                    'used_bytes': disk_usage.used,
                    'free_bytes': disk_usage.free,
                    'percent_used': disk_usage.percent,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                },
                '_note': 'host_metrics_fallback'
            }
            
        except Exception as e:
            print(f"Error getting host fallback resources: {e}")
            return {}
    
    def get_api_container_resources(
        self: Self
    ) -> Optional[dict[str, Any]]:
        """Get resource metrics for API container."""

        try:
            container = self.docker_client.containers.get(
                self.api_container_name
            )
            stats = container.stats(stream=False)
            
            return self._parse_container_stats(
                stats, "API Container"
            )
            
        except errors.NotFound:
            print(
                f"API container '{self.api_container_name}' "
                f"not found"
            )
            return None
 
        except Exception as e:
            print(f"Error getting API container stats: {e}")
            return None

    def get_postgres_container_resources(
        self: Self
    ) -> Optional[dict[str, Any]]:
        """Get resource metrics for PostgreSQL."""
    
        try:
            container = self.docker_client.containers.get(
                self.postgres_container_name
            )
            stats = container.stats(stream=False)
            return self._parse_container_stats(
                stats, "PostgreSQL Container"
            )
            
        except errors.NotFound:
            print(
                f"PostgreSQL container "
                f"'{self.postgres_container_name}' not found"
            )
            return None

        except Exception as e:
            print(
                f"Error getting PostgreSQL "
                f"container stats: {e}"
            )
            return None

    def _parse_container_stats(
        self: Self,
        stats: DockerStats,
        container_type: str
    ) -> dict[str, Any]:
        """Parse Docker container statistics."""

        cpu_stats = stats['cpu_stats']
        precpu_stats = stats['precpu_stats']
        cpu_delta = (
            cpu_stats['cpu_usage']['total_usage'] - 
            precpu_stats['cpu_usage']['total_usage']
        )
        system_delta = (
            cpu_stats['system_cpu_usage'] - 
            precpu_stats['system_cpu_usage']
        )
        cpu_percent = 0.0
        
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (
                (cpu_delta / system_delta) * 
                len(
                    cpu_stats['cpu_usage']['percpu_usage'] 
                    or [1]
                ) * 100
            )

        memory_stats = stats['memory_stats']
        memory_usage = memory_stats.get('usage', 0)
        memory_limit = memory_stats.get('limit', 0)
        memory_percent = (
            (memory_usage / memory_limit * 100) 
            if memory_limit > 0 else 0
        )

        net_stats = stats.get('networks', {})
        network_rx = 0
        network_tx = 0
        
        for iface in cast(
            list[NetworkStats],
            list(net_stats.values())
        ):
            network_rx += iface.get('rx_bytes', 0)
            network_tx += iface.get('tx_bytes', 0)
        
        disk_io_stats = stats['blkio_stats']
        read_bytes = 0
        write_bytes = 0
        
        for entry in disk_io_stats.get(
            'io_service_bytes_recursive', []
        ):
            if entry['op'] == 'Read':
                read_bytes += entry['value']

            elif entry['op'] == 'Write':
                write_bytes += entry['value']
        
        return {
            'cpu': {
                'percent_used': round(cpu_percent, 2),
                'cores_available': (
                    len(
                        cpu_stats['cpu_usage']['percpu_usage'] 
                        or [1]
                    )
                )
            },
            'memory': {
                'used_bytes': memory_usage,
                'limit_bytes': memory_limit,
                'percent_used': round(memory_percent, 2)
            },
            'network': {
                'rx_bytes': network_rx,
                'tx_bytes': network_tx
            },
            'disk_io': {
                'read_bytes': read_bytes,
                'write_bytes': write_bytes
            },
            'status': stats.get('status', 'unknown'),
            'container_id': stats['id'][:12],
            'container_type': container_type
        }
    
    def get_system_wide_metrics(self: Self) -> dict[str, Any]:
        """Get host-level system metrics."""
        try:
            load_avg = (
                os.getloadavg() 
                if hasattr(os, 'getloadavg') 
                else (0, 0, 0)
            )

            return {
                'load_average': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                },
                'timestamp': time.time(),
                'hostname': (
                    os.uname().nodename 
                    if hasattr(os, 'uname') 
                    else 'unknown'
                )
            }
        except Exception:
            return {}
