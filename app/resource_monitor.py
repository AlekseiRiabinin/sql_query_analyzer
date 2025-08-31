"""Resource monitor of Docker containers."""


import os
import time
import psutil
import docker
from docker import errors
from typing import (
    Self, Any, Optional,
    TypedDict, NotRequired
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

        try:
            # CPU calculation
            cpu_percent = 0.0
            cores_available = 1

            try:
                cpu_stats = stats.get('cpu_stats', {})
                precpu_stats = stats.get('precpu_stats', {})
                
                cpu_usage = cpu_stats.get('cpu_usage', {})
                precpu_usage = precpu_stats.get('cpu_usage', {})
                
                cpu_delta = (
                    cpu_usage.get('total_usage', 0) - 
                    precpu_usage.get('total_usage', 0)
                )
                system_delta = (
                    cpu_stats.get('system_cpu_usage', 0) -
                    precpu_stats.get('system_cpu_usage', 0)
                )
                
                percpu_usage = cpu_usage.get('percpu_usage')
                if percpu_usage is not None:
                    cores_available = len(percpu_usage)
                else:
                    cores_available = psutil.cpu_count() or 1
                
                if system_delta > 0 and cpu_delta > 0:
                    cpu_percent = (
                        (cpu_delta / system_delta) * 
                        cores_available * 100
                    )
                else:
                    cpu_percent = 0.0

            except (
                KeyError, TypeError, ZeroDivisionError
            ) as e:
                print(f"Error calculating CPU stats: {e}")
                cpu_percent = 0.0
                cores_available = psutil.cpu_count() or 1

            # Memory calculation
            memory_stats = stats.get('memory_stats', {})
            memory_usage = memory_stats.get('usage', 0)
            memory_limit = memory_stats.get('limit', 0)
            memory_percent = 0.0

            try:
                if memory_limit > 0:
                    memory_percent = (
                        memory_usage / memory_limit
                    ) * 100

            except (TypeError, ZeroDivisionError):
                memory_percent = 0.0

            net_stats = stats.get('networks', {})
            network_rx = 0
            network_tx = 0
            
            try:
                if isinstance(net_stats, dict):
                    for iface_stats in net_stats.values():
                        if isinstance(iface_stats, dict):
                            network_rx += (
                                iface_stats.get('rx_bytes', 0)
                            )
                            network_tx += (
                                iface_stats.get('tx_bytes', 0)
                            )

            except (TypeError, AttributeError) as e:
                print(f"Error parsing network stats: {e}")

            # Disk I/O statistics            
            disk_io_stats = stats.get('blkio_stats', {})
            read_bytes = 0
            write_bytes = 0
            iops = 0
            
            try:
                # Read/Write bytes
                io_entries = disk_io_stats.get(
                    'io_service_bytes_recursive', []
                )
                if isinstance(io_entries, list):
                    for entry in io_entries:

                        if isinstance(entry, dict):
                            op = entry.get('op', '')
                            value = entry.get('value', 0)

                            if op == 'Read':
                                read_bytes += value
                            elif op == 'Write':
                                write_bytes += value
                
                # IOPS calculation
                io_serviced = disk_io_stats.get(
                    'io_serviced_recursive', []
                )
                if isinstance(io_serviced, list):
                    for entry in io_serviced:

                        if isinstance(entry, dict):
                            op = entry.get('op', '')
                            value = entry.get('value', 0)

                            if op in ['Read', 'Write']:
                                iops += value
                                
            except (TypeError, AttributeError, KeyError) as e:
                print(f"Error parsing disk I/O stats: {e}")

            # Container metadata with error handling
            container_id = 'unknown'
            status = 'unknown'
            
            try:
                container_id = stats.get('id', 'unknown')[:12]
                status = stats.get('status', 'unknown')
            except (TypeError, AttributeError, IndexError):
                pass

            return {
                'cpu': {
                    'percent_used': round(cpu_percent, 2),
                    'cores_available': cores_available,
                    '_calculated': True
                },
                'memory': {
                    'used_bytes': memory_usage,
                    'limit_bytes': memory_limit,
                    'percent_used': round(memory_percent, 2),
                    '_calculated': True
                },
                'network': {
                    'rx_bytes': network_rx,
                    'tx_bytes': network_tx,
                    '_calculated': True
                },
                'disk_io': {
                    'read_bytes': read_bytes,
                    'write_bytes': write_bytes,
                    'iops': iops,
                    '_calculated': True
                },
                'status': status,
                'container_id': container_id,
                'container_type': container_type,
                '_source': 'docker_stats',
                '_timestamp': time.time()
            }

        except Exception as e:
            print(
                f"Critical error parsing "
                f"container stats: {e}"
            )
            return self._get_minimal_container_stats(
                stats, container_type
            )

    def _get_minimal_container_stats(
        self: Self, 
        stats: dict, 
        container_type: str
    ) -> dict[str, Any]:
        """Fallback method for minimal container stats."""
        try:
            container_id = stats.get('id', 'unknown')[:12]
            status = stats.get('status', 'unknown')
        except:
            container_id = 'unknown'
            status = 'unknown'
        
        return {
            'cpu': {
                'percent_used': 0,
                'cores_available': psutil.cpu_count() or 1,
                '_calculated': False
            },
            'memory': {
                'used_bytes': 0,
                'limit_bytes': 0,
                'percent_used': 0,
                '_calculated': False
            },
            'network': {
                'rx_bytes': 0,
                'tx_bytes': 0,
                '_calculated': False
            },
            'disk_io': {
                'read_bytes': 0,
                'write_bytes': 0,
                'iops': 0,
                '_calculated': False
            },
            'status': status,
            'container_id': container_id,
            'container_type': container_type,
            '_source': 'minimal_fallback',
            '_error': 'full_parsing_failed',
            '_timestamp': time.time()
        }

    def _calculate_iops(
        self: Self,
        disk_io_stats: dict
    ) -> int:
        """Calculate IOPS from disk I/O stats."""

        try:
            iops = 0
            io_entries = disk_io_stats.get(
                'io_serviced_recursive', []
            )
            
            for entry in io_entries:
                if isinstance(entry, dict):
                    op = entry.get('op')
                    value = entry.get('value', 0)
                    
                    if (
                        op in ['Read', 'Write'] and 
                        isinstance(value, (int, float))
                    ):
                        iops += int(value)                       
            return iops
        
        except Exception as e:
            print(f"Error calculating IOPS: {e}")
            return 0

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
        except Exception as e:
            print(f"Error getting system-wide metrics: {e}")
            return {
                'load_average': {
                    '1min': 0, '5min': 0, '15min': 0
                },
                'timestamp': time.time(),
                'hostname': 'unknown',
                '_error': str(e)
            }
