"""
Enhanced System Information Tool with comprehensive system analysis and monitoring.
Dependencies: psutil (optional), GPUtil (optional)
"""

import platform
import os
import sys
import socket
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Install with: pip install psutil")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPUtil not available. Install with: pip install gputil")

logger = logging.getLogger(__name__)

class SystemInfoTool:
    """Enhanced system information tool with comprehensive system analysis and monitoring."""
    
    name = 'system_info'
    description = 'Get comprehensive system information including hardware, software, performance, and network details'
    
    def run(self, 
            detail_level: str = 'comprehensive',
            include_performance: bool = True,
            include_network: bool = True,
            include_processes: bool = False,
            max_processes: int = 10) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Args:
            detail_level: Level of detail ('basic', 'standard', 'comprehensive')
            include_performance: Whether to include performance metrics
            include_network: Whether to include network information
            include_processes: Whether to include running processes
            max_processes: Maximum number of processes to include
            
        Returns:
            Comprehensive system information dictionary
        """
        try:
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'detail_level': detail_level,
                'collection_duration': None,
            }
            
            start_time = datetime.now()
            
            # Basic system information (always collected)
            system_info.update(self._get_basic_system_info())
            
            # Platform-specific information
            system_info.update(self._get_platform_info())
            
            # Python environment
            system_info.update(self._get_python_info())
            
            # Performance metrics if requested
            if include_performance and PSUTIL_AVAILABLE:
                system_info.update(self._get_performance_metrics())
            
            # Network information if requested
            if include_network:
                system_info.update(self._get_network_info())
            
            # GPU information if available
            if GPU_AVAILABLE:
                system_info.update(self._get_gpu_info())
            
            # Process information if requested
            if include_processes and PSUTIL_AVAILABLE:
                system_info.update(self._get_process_info(max_processes))
            
            # System health assessment
            if detail_level in ['standard', 'comprehensive']:
                system_info.update(self._assess_system_health(system_info))
            
            # Calculate collection duration
            system_info['collection_duration'] = (datetime.now() - start_time).total_seconds()
            
            logger.info("System information collected successfully")
            return system_info
            
        except Exception as e:
            logger.error(f"System information collection failed: {e}")
            return self._error_result(f"System information collection failed: {str(e)}")
    
    def _get_basic_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            'system': {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'architecture': platform.architecture()[0],
                'machine': platform.machine(),
                'processor': platform.processor(),
                'hostname': socket.gethostname(),
                'fqdn': socket.getfqdn(),
            }
        }
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get platform-specific information."""
        platform_info = {}
        
        try:
            # Windows-specific information
            if platform.system() == 'Windows':
                platform_info['windows'] = {
                    'edition': platform.win32_edition() if hasattr(platform, 'win32_edition') else 'N/A',
                    'version': platform.win32_ver(),
                }
            
            # Linux-specific information
            elif platform.system() == 'Linux':
                platform_info['linux'] = self._get_linux_info()
            
            # macOS-specific information
            elif platform.system() == 'Darwin':
                platform_info['macos'] = self._get_macos_info()
            
        except Exception as e:
            logger.warning(f"Platform-specific info collection failed: {e}")
            platform_info['platform_info_error'] = str(e)
        
        return platform_info
    
    def _get_linux_info(self) -> Dict[str, Any]:
        """Get Linux-specific information."""
        linux_info = {}
        
        try:
            # Try to get distribution information
            if hasattr(platform, 'linux_distribution'):
                dist_info = platform.linux_distribution()
                linux_info['distribution'] = {
                    'name': dist_info[0],
                    'version': dist_info[1],
                    'codename': dist_info[2],
                }
            
            # Try to read /etc/os-release
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            linux_info[key.lower()] = value.strip('"')
            
            # Get kernel version
            linux_info['kernel'] = platform.release()
            
        except Exception as e:
            logger.warning(f"Linux info collection failed: {e}")
            linux_info['error'] = str(e)
        
        return linux_info
    
    def _get_macos_info(self) -> Dict[str, Any]:
        """Get macOS-specific information."""
        macos_info = {}
        
        try:
            # Try to get macOS version
            if hasattr(platform, 'mac_ver'):
                mac_ver = platform.mac_ver()
                macos_info['version'] = mac_ver[0]
                macos_info['build'] = mac_ver[2] if len(mac_ver) > 2 else 'N/A'
            
            # Try to get system profiler info
            try:
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    macos_info['hardware_overview'] = result.stdout[:1000]  # First 1000 chars
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
        except Exception as e:
            logger.warning(f"macOS info collection failed: {e}")
            macos_info['error'] = str(e)
        
        return macos_info
    
    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python environment information."""
        return {
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler(),
                'build': platform.python_build(),
                'executable': sys.executable,
                'path': sys.path,
                'prefix': sys.prefix,
                'byteorder': sys.byteorder,
                'modules_loaded': len(sys.modules),
            }
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics using psutil."""
        if not PSUTIL_AVAILABLE:
            return {'performance_metrics': {'error': 'psutil not available'}}
        
        try:
            # CPU information
            cpu_times = psutil.cpu_times()
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'usage_percent': psutil.cpu_percent(interval=0.1),
                'usage_per_core': psutil.cpu_percent(interval=0.1, percpu=True),
                'times': {
                    'user': cpu_times.user,
                    'system': cpu_times.system,
                    'idle': cpu_times.idle,
                }
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            memory_info = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent,
            }
            
            # Disk information
            disk_info = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info[partition.mountpoint] = {
                        'device': partition.device,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': usage.percent,
                    }
                except PermissionError:
                    continue
            
            # Load average (Unix-like systems)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            return {
                'performance_metrics': {
                    'cpu': cpu_info,
                    'memory': memory_info,
                    'disk': disk_info,
                    'load_average': load_avg,
                    'boot_time': psutil.boot_time(),
                    'users': [user.name for user in psutil.users()],
                }
            }
            
        except Exception as e:
            logger.warning(f"Performance metrics collection failed: {e}")
            return {'performance_metrics': {'error': str(e)}}
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        network_info = {}
        
        try:
            # Network interfaces
            interfaces = {}
            for interface, addrs in psutil.net_if_addrs().items() if PSUTIL_AVAILABLE else {}:
                interfaces[interface] = []
                for addr in addrs:
                    interfaces[interface].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast,
                    })
            
            # Network I/O statistics
            io_counters = {}
            if PSUTIL_AVAILABLE:
                for interface, counters in psutil.net_io_counters(pernic=True).items():
                    io_counters[interface] = {
                        'bytes_sent': counters.bytes_sent,
                        'bytes_recv': counters.bytes_recv,
                        'packets_sent': counters.packets_sent,
                        'packets_recv': counters.packets_recv,
                    }
            
            network_info['network'] = {
                'hostname': socket.gethostname(),
                'fqdn': socket.getfqdn(),
                'interfaces': interfaces,
                'io_counters': io_counters,
            }
            
        except Exception as e:
            logger.warning(f"Network info collection failed: {e}")
            network_info['network'] = {'error': str(e)}
        
        return network_info
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information using GPUtil."""
        if not GPU_AVAILABLE:
            return {'gpu': {'error': 'GPUtil not available'}}
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # Convert to percentage
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'temperature': gpu.temperature,
                    'driver': gpu.driver,
                })
            
            return {'gpu': {'gpus': gpu_info, 'total_gpus': len(gpus)}}
            
        except Exception as e:
            logger.warning(f"GPU info collection failed: {e}")
            return {'gpu': {'error': str(e)}}
    
    def _get_process_info(self, max_processes: int) -> Dict[str, Any]:
        """Get information about running processes."""
        if not PSUTIL_AVAILABLE:
            return {'processes': {'error': 'psutil not available'}}
        
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                try:
                    process_info = proc.info
                    processes.append(process_info)
                    if len(processes) >= max_processes:
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x.get('cpu_percent', 0) or 0, reverse=True)
            
            return {
                'processes': {
                    'total_processes': len(processes),
                    'top_processes': processes,
                    'total_cpu_usage': sum(p.get('cpu_percent', 0) or 0 for p in processes),
                    'total_memory_usage': sum(p.get('memory_percent', 0) or 0 for p in processes),
                }
            }
            
        except Exception as e:
            logger.warning(f"Process info collection failed: {e}")
            return {'processes': {'error': str(e)}}
    
    def _assess_system_health(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system health based on collected metrics."""
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': [],
            'score': 100,  # Start with perfect score
        }
        
        try:
            # Check memory usage
            if PSUTIL_AVAILABLE:
                memory = system_info.get('performance_metrics', {}).get('memory', {})
                if memory.get('percent', 0) > 90:
                    health['issues'].append('High memory usage (>90%)')
                    health['score'] -= 20
                    health['recommendations'].append('Consider closing unused applications or adding more RAM')
                elif memory.get('percent', 0) > 80:
                    health['issues'].append('Elevated memory usage (>80%)')
                    health['score'] -= 10
            
            # Check CPU usage
            cpu = system_info.get('performance_metrics', {}).get('cpu', {})
            if cpu.get('usage_percent', 0) > 90:
                health['issues'].append('High CPU usage (>90%)')
                health['score'] -= 20
                health['recommendations'].append('Check for CPU-intensive processes')
            elif cpu.get('usage_percent', 0) > 80:
                health['issues'].append('Elevated CPU usage (>80%)')
                health['score'] -= 10
            
            # Check disk space
            disk = system_info.get('performance_metrics', {}).get('disk', {})
            for mount, info in disk.items():
                if info.get('percent', 0) > 95:
                    health['issues'].append(f'Critical disk space on {mount} (>95%)')
                    health['score'] -= 25
                    health['recommendations'].append(f'Free up space on {mount}')
                elif info.get('percent', 0) > 90:
                    health['issues'].append(f'Low disk space on {mount} (>90%)')
                    health['score'] -= 15
            
            # Update status based on score
            if health['score'] >= 80:
                health['status'] = 'healthy'
            elif health['score'] >= 60:
                health['status'] = 'degraded'
            else:
                health['status'] = 'critical'
            
        except Exception as e:
            logger.warning(f"System health assessment failed: {e}")
            health['assessment_error'] = str(e)
        
        return {'system_health': health}
    
    def monitor_system(self, 
                      duration: int = 60, 
                      interval: int = 5,
                      metrics: List[str] = None) -> Dict[str, Any]:
        """
        Monitor system performance over time.
        
        Args:
            duration: Monitoring duration in seconds
            interval: Sampling interval in seconds
            metrics: List of metrics to monitor
            
        Returns:
            System monitoring results
        """
        if not PSUTIL_AVAILABLE:
            return self._error_result("psutil required for system monitoring")
        
        if metrics is None:
            metrics = ['cpu', 'memory', 'disk']
        
        monitoring_data = {
            'start_time': datetime.now().isoformat(),
            'duration': duration,
            'interval': interval,
            'samples': [],
            'summary': {}
        }
        
        import time
        
        try:
            samples = []
            start_time = time.time()
            
            while time.time() - start_time < duration:
                sample = {
                    'timestamp': datetime.now().isoformat(),
                    'elapsed': time.time() - start_time,
                }
                
                if 'cpu' in metrics:
                    sample['cpu'] = psutil.cpu_percent(interval=1, percpu=True)
                
                if 'memory' in metrics:
                    memory = psutil.virtual_memory()
                    sample['memory'] = {
                        'percent': memory.percent,
                        'used': memory.used,
                        'available': memory.available,
                    }
                
                if 'disk' in metrics:
                    disk = psutil.disk_usage('/')
                    sample['disk'] = {
                        'percent': disk.percent,
                        'used': disk.used,
                        'free': disk.free,
                    }
                
                samples.append(sample)
                time.sleep(interval)
            
            monitoring_data['samples'] = samples
            monitoring_data['end_time'] = datetime.now().isoformat()
            
            # Calculate summary statistics
            monitoring_data['summary'] = self._calculate_monitoring_summary(samples, metrics)
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"System monitoring failed: {e}")
            return self._error_result(f"System monitoring failed: {str(e)}")
    
    def _calculate_monitoring_summary(self, samples: List[Dict], metrics: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics from monitoring data."""
        summary = {}
        
        if 'cpu' in metrics and samples:
            cpu_values = [max(sample.get('cpu', [0])) for sample in samples]
            summary['cpu'] = {
                'average': sum(cpu_values) / len(cpu_values),
                'maximum': max(cpu_values),
                'minimum': min(cpu_values),
            }
        
        if 'memory' in metrics and samples:
            memory_values = [sample.get('memory', {}).get('percent', 0) for sample in samples]
            summary['memory'] = {
                'average': sum(memory_values) / len(memory_values),
                'maximum': max(memory_values),
                'minimum': min(memory_values),
            }
        
        return summary
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy system info tool (maintains original interface)."""
    
    name = 'system_info'
    description = 'Get system information'
    
    def __init__(self):
        self.enhanced_tool = SystemInfoTool()
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get system information with enhanced capabilities.
        
        Args:
            **kwargs: Additional parameters for detailed system analysis
            
        Returns:
            System information dictionary
        """
        return self.enhanced_tool.run(**kwargs)


# Utility functions
def get_system_benchmark() -> Dict[str, Any]:
    """Run basic system performance benchmark."""
    import time
    
    benchmark = {
        'start_time': datetime.now().isoformat(),
        'tests': {}
    }
    
    # CPU benchmark (simple calculation test)
    start = time.time()
    for _ in range(10**7):
        _ = 123.456 * 789.012
    benchmark['tests']['cpu_calculation'] = time.time() - start
    
    # Memory benchmark (list operations)
    start = time.time()
    test_list = list(range(10**6))
    test_list.reverse()
    benchmark['tests']['memory_operations'] = time.time() - start
    
    benchmark['end_time'] = datetime.now().isoformat()
    benchmark['total_duration'] = benchmark['end_time'] - benchmark['start_time']
    
    return benchmark


# Example usage
if __name__ == "__main__":
    tool = SystemInfoTool()
    
    # Get comprehensive system information
    system_info = tool.run(
        detail_level='comprehensive',
        include_performance=True,
        include_network=True
    )
    
    print("System Information Summary:")
    print(f"Platform: {system_info['system']['platform']}")
    print(f"CPU Cores: {system_info.get('performance_metrics', {}).get('cpu', {}).get('logical_cores', 'N/A')}")
    print(f"Memory: {system_info.get('performance_metrics', {}).get('memory', {}).get('percent', 'N/A')}% used")
    print(f"System Health: {system_info.get('system_health', {}).get('status', 'N/A')}")
    
    # Show available metrics
    if system_info.get('performance_metrics'):
        print("\nPerformance Metrics Available:")
        for metric in system_info['performance_metrics'].keys():
            print(f"  - {metric}")