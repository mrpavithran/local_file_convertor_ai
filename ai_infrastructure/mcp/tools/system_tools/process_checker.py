"""
Enhanced Process Checker Tool with comprehensive process analysis and management.
Dependencies: psutil
"""

import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ProcessCheckerTool:
    """Enhanced process checker tool with comprehensive process analysis and management."""
    
    name = 'process_check'
    description = 'Comprehensive process analysis including monitoring, resource usage, and management'
    
    def run(self, 
            name: Optional[str] = None,
            pid: Optional[int] = None,
            detail_level: str = 'basic',
            include_children: bool = True,
            resource_metrics: bool = True,
            max_results: int = 50) -> Dict[str, Any]:
        """
        Analyze processes with comprehensive details and filtering.
        
        Args:
            name: Process name to search for (partial match, case-insensitive)
            pid: Specific process ID to analyze
            detail_level: Level of detail ('basic', 'detailed', 'comprehensive')
            include_children: Whether to include child process information
            resource_metrics: Whether to include resource usage metrics
            max_results: Maximum number of processes to return
            
        Returns:
            Process analysis results
        """
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'search_criteria': {
                    'name': name,
                    'pid': pid,
                    'detail_level': detail_level
                },
                'total_processes': 0,
                'matching_processes': 0,
                'processes': []
            }
            
            # Get process information based on criteria
            if pid:
                processes = self._get_process_by_pid(pid)
            elif name:
                processes = self._get_processes_by_name(name, max_results)
            else:
                processes = self._get_all_processes(max_results)
            
            results['total_processes'] = len(processes)
            
            # Analyze each process
            for proc_info in processes:
                try:
                    process_analysis = self._analyze_process(
                        proc_info, 
                        detail_level, 
                        include_children, 
                        resource_metrics
                    )
                    if process_analysis:
                        results['processes'].append(process_analysis)
                        results['matching_processes'] += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Could not analyze process {proc_info.get('pid')}: {e}")
                    continue
            
            # Add summary statistics
            results.update(self._calculate_process_statistics(results['processes']))
            
            logger.info(f"Process analysis completed: {results['matching_processes']} processes analyzed")
            return results
            
        except Exception as e:
            logger.error(f"Process analysis failed: {e}")
            return self._error_result(f"Process analysis failed: {str(e)}")
    
    def _get_process_by_pid(self, pid: int) -> List[Dict[str, Any]]:
        """Get process information by PID."""
        try:
            process = psutil.Process(pid)
            return [{
                'pid': pid,
                'name': process.name(),
                'status': process.status()
            }]
        except psutil.NoSuchProcess:
            return []
    
    def _get_processes_by_name(self, name: str, max_results: int) -> List[Dict[str, Any]]:
        """Get processes by name (partial match, case-insensitive)."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                proc_info = proc.info
                if (proc_info['name'] and 
                    name.lower() in proc_info['name'].lower()):
                    processes.append(proc_info)
                    
                    if len(processes) >= max_results:
                        break
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def _get_all_processes(self, max_results: int) -> List[Dict[str, Any]]:
        """Get all processes with basic information."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                processes.append(proc.info)
                if len(processes) >= max_results:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def _analyze_process(self, 
                        proc_info: Dict[str, Any],
                        detail_level: str,
                        include_children: bool,
                        resource_metrics: bool) -> Optional[Dict[str, Any]]:
        """Analyze a single process with comprehensive details."""
        try:
            process = psutil.Process(proc_info['pid'])
            analysis = {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'basic_info': self._get_basic_process_info(process)
            }
            
            if detail_level in ['detailed', 'comprehensive']:
                analysis.update(self._get_detailed_process_info(process))
            
            if resource_metrics:
                analysis.update(self._get_resource_metrics(process))
            
            if include_children and detail_level == 'comprehensive':
                analysis['children'] = self._get_child_processes(process)
            
            if detail_level == 'comprehensive':
                analysis['advanced_metrics'] = self._get_advanced_metrics(process)
            
            return analysis
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not analyze process {proc_info.get('pid')}: {e}")
            return None
    
    def _get_basic_process_info(self, process: psutil.Process) -> Dict[str, Any]:
        """Get basic process information."""
        try:
            with process.oneshot():
                return {
                    'ppid': process.ppid(),
                    'exe': process.exe(),
                    'command_line': process.cmdline(),
                    'create_time': datetime.fromtimestamp(process.create_time()).isoformat() if process.create_time() else None,
                    'username': process.username(),
                    'terminal': process.terminal(),
                }
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return {'error': 'Access denied or process no longer exists'}
    
    def _get_detailed_process_info(self, process: psutil.Process) -> Dict[str, Any]:
        """Get detailed process information."""
        detailed_info = {}
        
        try:
            with process.oneshot():
                # Environment and working directory
                detailed_info['environment'] = {
                    'cwd': process.cwd(),
                    'environ_keys': list(process.environ().keys()) if process.environ() else [],
                    'num_env_vars': len(process.environ()) if process.environ() else 0
                }
                
                # Memory maps (if available)
                try:
                    memory_maps = process.memory_maps()
                    detailed_info['memory_maps'] = {
                        'total_mappings': len(memory_maps),
                        'sample_mappings': [str(map) for map in memory_maps[:3]]  # First 3 as sample
                    }
                except (psutil.AccessDenied, AttributeError):
                    detailed_info['memory_maps'] = {'available': False}
                
                # Open files
                try:
                    open_files = process.open_files()
                    detailed_info['open_files'] = {
                        'total_files': len(open_files),
                        'files': [file.path for file in open_files[:5]]  # First 5 files
                    }
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    detailed_info['open_files'] = {'available': False}
                
                # Connections
                try:
                    connections = process.connections()
                    detailed_info['network_connections'] = {
                        'total_connections': len(connections),
                        'connections': [
                            {
                                'family': conn.family.name,
                                'type': conn.type.name,
                                'laddr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                                'raddr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                                'status': conn.status
                            } for conn in connections[:5]  # First 5 connections
                        ]
                    }
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    detailed_info['network_connections'] = {'available': False}
        
        except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
            detailed_info['error'] = f"Could not get detailed info: {str(e)}"
        
        return {'detailed_info': detailed_info}
    
    def _get_resource_metrics(self, process: psutil.Process) -> Dict[str, Any]:
        """Get process resource usage metrics."""
        metrics = {}
        
        try:
            with process.oneshot():
                # CPU usage
                cpu_times = process.cpu_times()
                metrics['cpu'] = {
                    'percent': process.cpu_percent(),
                    'user_time': cpu_times.user,
                    'system_time': cpu_times.system,
                    'children_user_time': getattr(cpu_times, 'children_user', 0),
                    'children_system_time': getattr(cpu_times, 'children_system', 0),
                }
                
                # Memory usage
                memory_info = process.memory_info()
                memory_full_info = process.memory_full_info() if hasattr(process, 'memory_full_info') else None
                
                metrics['memory'] = {
                    'rss': memory_info.rss,  # Resident Set Size
                    'vms': memory_info.vms,  # Virtual Memory Size
                    'percent': process.memory_percent(),
                }
                
                # Additional memory metrics if available
                if memory_full_info:
                    metrics['memory'].update({
                        'uss': getattr(memory_full_info, 'uss', 0),  # Unique Set Size
                        'pss': getattr(memory_full_info, 'pss', 0),  # Proportional Set Size
                        'swap': getattr(memory_full_info, 'swap', 0),
                    })
                
                # I/O statistics
                try:
                    io_counters = process.io_counters()
                    metrics['io'] = {
                        'read_count': io_counters.read_count,
                        'write_count': io_counters.write_count,
                        'read_bytes': io_counters.read_bytes,
                        'write_bytes': io_counters.write_bytes,
                    }
                except (psutil.AccessDenied, AttributeError):
                    metrics['io'] = {'available': False}
                
                # Thread information
                metrics['threads'] = {
                    'num_threads': process.num_threads(),
                    'thread_ids': process.threads(),
                }
        
        except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
            metrics['error'] = f"Could not get resource metrics: {str(e)}"
        
        return {'resource_metrics': metrics}
    
    def _get_child_processes(self, process: psutil.Process) -> List[Dict[str, Any]]:
        """Get child processes information."""
        children = []
        
        try:
            for child in process.children(recursive=True):
                try:
                    children.append({
                        'pid': child.pid,
                        'name': child.name(),
                        'status': child.status(),
                        'cpu_percent': child.cpu_percent(),
                        'memory_percent': child.memory_percent(),
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return children
    
    def _get_advanced_metrics(self, process: psutil.Process) -> Dict[str, Any]:
        """Get advanced process metrics."""
        advanced = {}
        
        try:
            with process.oneshot():
                # Nice value (priority)
                try:
                    advanced['nice'] = process.nice()
                except (psutil.AccessDenied, AttributeError):
                    advanced['nice'] = 'N/A'
                
                # I/O priority (Windows/Linux)
                try:
                    advanced['ionice'] = process.ionice()
                except (psutil.AccessDenied, AttributeError):
                    advanced['ionice'] = 'N/A'
                
                # CPU affinity
                try:
                    advanced['cpu_affinity'] = process.cpu_affinity()
                except (psutil.AccessDenied, AttributeError):
                    advanced['cpu_affinity'] = 'N/A'
                
                # Process limits (Unix-like systems)
                try:
                    advanced['rlimit'] = {
                        'soft_limits': {},
                        'hard_limits': {}
                    }
                    # Sample common limits
                    for resource in [psutil.RLIMIT_CPU, psutil.RLIMIT_AS, psutil.RLIMIT_NOFILE]:
                        try:
                            soft, hard = process.rlimit(resource)
                            resource_name = resource.name if hasattr(resource, 'name') else str(resource)
                            advanced['rlimit']['soft_limits'][resource_name] = soft
                            advanced['rlimit']['hard_limits'][resource_name] = hard
                        except (psutil.AccessDenied, AttributeError):
                            continue
                except (psutil.AccessDenied, AttributeError):
                    advanced['rlimit'] = {'available': False}
        
        except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
            advanced['error'] = f"Could not get advanced metrics: {str(e)}"
        
        return advanced
    
    def _calculate_process_statistics(self, processes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from process analysis."""
        if not processes:
            return {}
        
        stats = {
            'status_distribution': {},
            'total_cpu_usage': 0,
            'total_memory_usage': 0,
            'average_cpu_usage': 0,
            'average_memory_usage': 0,
        }
        
        cpu_usages = []
        memory_usages = []
        
        for process in processes:
            # Status distribution
            status = process.get('status', 'unknown')
            stats['status_distribution'][status] = stats['status_distribution'].get(status, 0) + 1
            
            # Resource usage
            metrics = process.get('resource_metrics', {})
            cpu_percent = metrics.get('cpu', {}).get('percent', 0)
            memory_percent = metrics.get('memory', {}).get('percent', 0)
            
            stats['total_cpu_usage'] += cpu_percent
            stats['total_memory_usage'] += memory_percent
            cpu_usages.append(cpu_percent)
            memory_usages.append(memory_percent)
        
        if cpu_usages:
            stats['average_cpu_usage'] = sum(cpu_usages) / len(cpu_usages)
        if memory_usages:
            stats['average_memory_usage'] = sum(memory_usages) / len(memory_usages)
        
        return {'statistics': stats}
    
    def monitor_process(self, 
                       identifier: str,
                       duration: int = 60,
                       interval: int = 2) -> Dict[str, Any]:
        """
        Monitor a process over time.
        
        Args:
            identifier: Process name or PID
            duration: Monitoring duration in seconds
            interval: Sampling interval in seconds
        """
        import time
        
        monitoring_data = {
            'identifier': identifier,
            'start_time': datetime.now().isoformat(),
            'duration': duration,
            'interval': interval,
            'samples': []
        }
        
        try:
            # Determine if identifier is PID or name
            try:
                pid = int(identifier)
                process = psutil.Process(pid)
            except ValueError:
                # It's a name, find the first matching process
                processes = self._get_processes_by_name(identifier, 1)
                if not processes:
                    return self._error_result(f"No process found with name: {identifier}")
                process = psutil.Process(processes[0]['pid'])
            
            start_time = time.time()
            samples = []
            
            while time.time() - start_time < duration:
                try:
                    with process.oneshot():
                        sample = {
                            'timestamp': datetime.now().isoformat(),
                            'elapsed': time.time() - start_time,
                            'cpu_percent': process.cpu_percent(),
                            'memory_percent': process.memory_percent(),
                            'memory_rss': process.memory_info().rss,
                            'num_threads': process.num_threads(),
                            'status': process.status(),
                        }
                        
                        # Get I/O if available
                        try:
                            io = process.io_counters()
                            sample['io_read_bytes'] = io.read_bytes
                            sample['io_write_bytes'] = io.write_bytes
                        except (psutil.AccessDenied, AttributeError):
                            pass
                        
                        samples.append(sample)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    monitoring_data['error'] = 'Process terminated or access denied during monitoring'
                    break
                
                time.sleep(interval)
            
            monitoring_data['samples'] = samples
            monitoring_data['end_time'] = datetime.now().isoformat()
            monitoring_data['summary'] = self._calculate_monitoring_summary(samples)
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Process monitoring failed: {e}")
            return self._error_result(f"Process monitoring failed: {str(e)}")
    
    def _calculate_monitoring_summary(self, samples: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from monitoring data."""
        if not samples:
            return {}
        
        summary = {}
        
        # CPU statistics
        cpu_values = [s.get('cpu_percent', 0) for s in samples]
        summary['cpu'] = {
            'average': sum(cpu_values) / len(cpu_values),
            'maximum': max(cpu_values),
            'minimum': min(cpu_values),
        }
        
        # Memory statistics
        memory_values = [s.get('memory_percent', 0) for s in samples]
        summary['memory'] = {
            'average': sum(memory_values) / len(memory_values),
            'maximum': max(memory_values),
            'minimum': min(memory_values),
        }
        
        # RSS memory statistics
        rss_values = [s.get('memory_rss', 0) for s in samples]
        summary['memory_rss'] = {
            'average': sum(rss_values) / len(rss_values),
            'maximum': max(rss_values),
            'minimum': min(rss_values),
        }
        
        return summary
    
    def find_resource_hogs(self, 
                          sort_by: str = 'cpu',
                          limit: int = 10) -> Dict[str, Any]:
        """
        Find processes using the most resources.
        
        Args:
            sort_by: Sort by 'cpu', 'memory', or 'io'
            limit: Number of top processes to return
        """
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                with proc.oneshot():
                    process_info = {
                        'pid': proc.pid,
                        'name': proc.name(),
                        'cpu_percent': proc.cpu_percent(),
                        'memory_percent': proc.memory_percent(),
                    }
                    
                    # Add I/O if available and requested
                    if sort_by == 'io':
                        try:
                            io = proc.io_counters()
                            process_info['io_total'] = io.read_bytes + io.write_bytes
                        except (psutil.AccessDenied, AttributeError):
                            process_info['io_total'] = 0
                    
                    processes.append(process_info)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort processes
        if sort_by == 'cpu':
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        elif sort_by == 'memory':
            processes.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
        elif sort_by == 'io':
            processes.sort(key=lambda x: x.get('io_total', 0), reverse=True)
        
        return {
            'resource_hogs': processes[:limit],
            'sort_by': sort_by,
            'timestamp': datetime.now().isoformat()
        }
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy process checker tool (maintains original interface)."""
    
    name = 'process_check'
    description = 'Check and analyze processes'
    
    def __init__(self):
        self.enhanced_tool = ProcessCheckerTool()
    
    def run(self, name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Check processes with enhanced capabilities.
        
        Args:
            name: Process name to search for
            **kwargs: Additional analysis parameters
            
        Returns:
            Process analysis results
        """
        return self.enhanced_tool.run(name=name, **kwargs)


# Example usage
if __name__ == "__main__":
    tool = ProcessCheckerTool()
    
    # Find Python processes with detailed analysis
    results = tool.run(
        name='python',
        detail_level='detailed',
        resource_metrics=True,
        max_results=5
    )
    
    print(f"Found {results['matching_processes']} Python processes")
    print(f"Total CPU usage: {results['statistics']['total_cpu_usage']:.2f}%")
    print(f"Total memory usage: {results['statistics']['total_memory_usage']:.2f}%")
    
    # Monitor a specific process
    if results['processes']:
        first_pid = results['processes'][0]['pid']
        monitor_results = tool.monitor_process(str(first_pid), duration=10, interval=1)
        print(f"Monitoring completed: {len(monitor_results.get('samples', []))} samples")
    
    # Find resource hogs
    hogs = tool.find_resource_hogs(sort_by='cpu', limit=5)
    print(f"Top 5 CPU hogs:")
    for hog in hogs['resource_hogs']:
        print(f"  {hog['name']} (PID {hog['pid']}): {hog['cpu_percent']}% CPU")