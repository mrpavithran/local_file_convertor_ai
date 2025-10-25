"""
Enhanced Network Check Tool with comprehensive network diagnostics and analysis.
Dependencies: psutil (optional), ping3 (optional), requests (optional)
"""

import socket
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Install with: pip install psutil")

try:
    import ping3
    PING3_AVAILABLE = True
except ImportError:
    PING3_AVAILABLE = False
    logging.warning("ping3 not available. Install with: pip install ping3")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available. Install with: pip install requests")

logger = logging.getLogger(__name__)

class NetworkCheckTool:
    """Enhanced network check tool with comprehensive diagnostics and analysis."""
    
    name = 'network_check'
    description = 'Comprehensive network diagnostics including connectivity, DNS, latency, and performance analysis'
    
    def run(self, 
            target: str = '8.8.8.8',
            checks: List[str] = None,
            timeout: int = 5,
            detailed: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive network diagnostics.
        
        Args:
            target: Hostname or IP address to check
            checks: Specific checks to perform ('dns', 'ping', 'http', 'ports', 'traceroute')
            timeout: Timeout for network operations in seconds
            detailed: Whether to include detailed analysis
            
        Returns:
            Comprehensive network diagnostics results
        """
        if checks is None:
            checks = ['dns', 'ping', 'http', 'ports']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'checks_performed': checks,
            'timeout': timeout,
            'results': {}
        }
        
        try:
            # Basic connectivity checks
            if 'dns' in checks:
                results['results']['dns'] = self._check_dns(target, timeout, detailed)
            
            if 'ping' in checks:
                results['results']['ping'] = self._check_ping(target, timeout, detailed)
            
            if 'http' in checks:
                results['results']['http'] = self._check_http(target, timeout, detailed)
            
            if 'ports' in checks:
                results['results']['ports'] = self._check_common_ports(target, timeout, detailed)
            
            if 'traceroute' in checks:
                results['results']['traceroute'] = self._check_traceroute(target, timeout)
            
            # Network interface information
            if detailed and PSUTIL_AVAILABLE:
                results['network_interfaces'] = self._get_network_interfaces()
            
            # Overall connectivity assessment
            results['connectivity_assessment'] = self._assess_connectivity(results['results'])
            
            logger.info(f"Network diagnostics completed for {target}")
            return results
            
        except Exception as e:
            logger.error(f"Network diagnostics failed: {e}")
            return self._error_result(f"Network diagnostics failed: {str(e)}")
    
    def _check_dns(self, host: str, timeout: int, detailed: bool) -> Dict[str, Any]:
        """Perform DNS resolution checks."""
        dns_results = {
            'resolvable': False,
            'resolved_ips': [],
            'reverse_dns': {}
        }
        
        try:
            # Forward DNS lookup
            start_time = time.time()
            ip_addresses = socket.getaddrinfo(host, None)
            resolve_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            unique_ips = set()
            for family, type, proto, canonname, sockaddr in ip_addresses:
                ip = sockaddr[0]
                unique_ips.add(ip)
                
                # Reverse DNS lookup if detailed
                if detailed:
                    try:
                        reverse_dns = socket.getnameinfo((ip, 0), 0)
                        dns_results['reverse_dns'][ip] = reverse_dns[0]
                    except socket.herror:
                        dns_results['reverse_dns'][ip] = None
            
            dns_results.update({
                'resolvable': True,
                'resolved_ips': list(unique_ips),
                'resolve_time_ms': round(resolve_time, 2),
                'ip_count': len(unique_ips)
            })
            
        except socket.gaierror as e:
            dns_results['error'] = f"DNS resolution failed: {str(e)}"
        except Exception as e:
            dns_results['error'] = f"DNS check failed: {str(e)}"
        
        return dns_results
    
    def _check_ping(self, target: str, timeout: int, detailed: bool) -> Dict[str, Any]:
        """Perform ping checks with latency measurement."""
        ping_results = {
            'reachable': False,
            'latency_ms': None
        }
        
        # Try using ping3 library first
        if PING3_AVAILABLE:
            try:
                start_time = time.time()
                latency = ping3.ping(target, timeout=timeout)
                if latency is not None:
                    ping_results.update({
                        'reachable': True,
                        'latency_ms': round(latency * 1000, 2),
                        'method': 'ping3'
                    })
                else:
                    ping_results['error'] = 'Host unreachable via ping3'
                
            except Exception as e:
                ping_results['error'] = f"Ping3 failed: {str(e)}"
        
        # Fallback to system ping command
        if not ping_results.get('reachable') and not ping_results.get('error'):
            try:
                # Platform-specific ping command
                if subprocess.os.name == 'nt':  # Windows
                    cmd = ['ping', '-n', '1', '-w', str(timeout * 1000), target]
                else:  # Unix-like
                    cmd = ['ping', '-c', '1', '-W', str(timeout), target]
                
                start_time = time.time()
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout + 2
                )
                
                if result.returncode == 0:
                    # Extract latency from ping output
                    latency = self._extract_latency_from_ping(result.stdout)
                    ping_results.update({
                        'reachable': True,
                        'latency_ms': latency,
                        'method': 'system_ping',
                        'output': result.stdout
                    })
                else:
                    ping_results['error'] = f"System ping failed: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                ping_results['error'] = 'Ping timeout'
            except Exception as e:
                ping_results['error'] = f"System ping failed: {str(e)}"
        
        # Additional detailed metrics
        if detailed and ping_results.get('reachable'):
            ping_results['latency_assessment'] = self._assess_latency(ping_results['latency_ms'])
        
        return ping_results
    
    def _extract_latency_from_ping(self, ping_output: str) -> Optional[float]:
        """Extract latency from ping command output."""
        try:
            # Windows ping output: "Reply from 8.8.8.8: bytes=32 time=12ms TTL=117"
            if 'time=' in ping_output:
                for line in ping_output.split('\n'):
                    if 'time=' in line:
                        time_str = line.split('time=')[1].split(' ')[0]
                        if 'ms' in time_str:
                            return float(time_str.replace('ms', ''))
                        else:
                            return float(time_str) * 1000  # Convert seconds to ms
            
            # Unix ping output: "64 bytes from 8.8.8.8: icmp_seq=1 ttl=117 time=12.3 ms"
            if 'time=' in ping_output:
                for line in ping_output.split('\n'):
                    if 'time=' in line and 'bytes from' in line:
                        time_str = line.split('time=')[1].split(' ')[0]
                        return float(time_str)
        
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _check_http(self, target: str, timeout: int, detailed: bool) -> Dict[str, Any]:
        """Perform HTTP/HTTPS connectivity checks."""
        http_results = {
            'http_accessible': False,
            'https_accessible': False,
            'response_times': {}
        }
        
        if not REQUESTS_AVAILABLE:
            http_results['error'] = 'requests library not available'
            return http_results
        
        # Test HTTP
        http_url = f"http://{target}" if not target.startswith(('http://', 'https://')) else target
        if not http_url.startswith('http://'):
            http_url = f"http://{target}"
        
        try:
            start_time = time.time()
            response = requests.get(http_url, timeout=timeout, allow_redirects=False)
            http_time = (time.time() - start_time) * 1000
            
            http_results.update({
                'http_accessible': True,
                'http_status_code': response.status_code,
                'http_response_time_ms': round(http_time, 2),
                'http_headers': dict(response.headers)
            })
            
        except requests.RequestException as e:
            http_results['http_error'] = str(e)
        
        # Test HTTPS
        https_url = http_url.replace('http://', 'https://')
        try:
            start_time = time.time()
            response = requests.get(https_url, timeout=timeout, allow_redirects=False, verify=False)
            https_time = (time.time() - start_time) * 1000
            
            http_results.update({
                'https_accessible': True,
                'https_status_code': response.status_code,
                'https_response_time_ms': round(https_time, 2),
                'https_headers': dict(response.headers)
            })
            
        except requests.RequestException as e:
            http_results['https_error'] = str(e)
        
        # SSL certificate check for HTTPS
        if http_results.get('https_accessible') and detailed:
            try:
                response = requests.get(https_url, timeout=timeout, verify=True)
                http_results['ssl_valid'] = True
                if hasattr(response.raw, 'connection'):
                    if hasattr(response.raw.connection, 'sock'):
                        if hasattr(response.raw.connection.sock, 'getpeercert'):
                            cert = response.raw.connection.sock.getpeercert()
                            http_results['ssl_certificate'] = {
                                'subject': cert.get('subject', []),
                                'issuer': cert.get('issuer', []),
                                'expires': cert.get('notAfter', '')
                            }
            except requests.RequestException:
                http_results['ssl_valid'] = False
        
        return http_results
    
    def _check_common_ports(self, target: str, timeout: int, detailed: bool) -> Dict[str, Any]:
        """Check common network ports."""
        common_ports = {
            21: 'FTP',
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            53: 'DNS',
            80: 'HTTP',
            110: 'POP3',
            143: 'IMAP',
            443: 'HTTPS',
            587: 'SMTP SSL',
            993: 'IMAP SSL',
            995: 'POP3 SSL',
            3306: 'MySQL',
            3389: 'RDP',
            5432: 'PostgreSQL',
            8080: 'HTTP Alt'
        }
        
        port_results = {
            'open_ports': [],
            'closed_ports': [],
            'filtered_ports': []
        }
        
        for port, service in common_ports.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                
                start_time = time.time()
                result = sock.connect_ex((target, port))
                connect_time = (time.time() - start_time) * 1000
                
                if result == 0:
                    port_info = {
                        'port': port,
                        'service': service,
                        'connect_time_ms': round(connect_time, 2)
                    }
                    port_results['open_ports'].append(port_info)
                else:
                    port_results['closed_ports'].append({
                        'port': port,
                        'service': service
                    })
                
                sock.close()
                
            except socket.timeout:
                port_results['filtered_ports'].append({
                    'port': port,
                    'service': service,
                    'reason': 'timeout'
                })
            except Exception as e:
                port_results['filtered_ports'].append({
                    'port': port,
                    'service': service,
                    'reason': str(e)
                })
        
        # Sort open ports by port number
        port_results['open_ports'].sort(key=lambda x: x['port'])
        
        if detailed:
            port_results['summary'] = {
                'total_checked': len(common_ports),
                'open_count': len(port_results['open_ports']),
                'closed_count': len(port_results['closed_ports']),
                'filtered_count': len(port_results['filtered_ports'])
            }
        
        return port_results
    
    def _check_traceroute(self, target: str, timeout: int) -> Dict[str, Any]:
        """Perform traceroute to target."""
        traceroute_results = {
            'hops': [],
            'completed': False
        }
        
        try:
            # Platform-specific traceroute command
            if subprocess.os.name == 'nt':  # Windows
                cmd = ['tracert', '-d', '-w', str(timeout * 1000), target]
            else:  # Unix-like
                cmd = ['traceroute', '-w', str(timeout), '-q', '1', target]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout * 30  # Longer timeout for traceroute
            )
            
            if result.returncode == 0:
                hops = self._parse_traceroute_output(result.stdout)
                traceroute_results.update({
                    'hops': hops,
                    'completed': True,
                    'hop_count': len(hops)
                })
            else:
                traceroute_results['error'] = f"Traceroute failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            traceroute_results['error'] = 'Traceroute timeout'
        except FileNotFoundError:
            traceroute_results['error'] = 'traceroute/tracert command not found'
        except Exception as e:
            traceroute_results['error'] = f"Traceroute failed: {str(e)}"
        
        return traceroute_results
    
    def _parse_traceroute_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse traceroute command output."""
        hops = []
        
        for line in output.split('\n'):
            line = line.strip()
            if not line or line.startswith('traceroute') or line.startswith('tracing'):
                continue
            
            # Parse hop information
            parts = line.split()
            if len(parts) >= 2:
                try:
                    hop_num = int(parts[0])
                    hop_info = {'hop': hop_num}
                    
                    # Extract IP addresses and latency
                    for part in parts[1:]:
                        if self._is_ip_address(part):
                            hop_info['ip'] = part
                        elif part.endswith('ms'):
                            latency = float(part.replace('ms', ''))
                            hop_info['latency_ms'] = latency
                    
                    if 'ip' in hop_info:
                        # Try reverse DNS lookup
                        try:
                            hostname = socket.getfqdn(hop_info['ip'])
                            if hostname != hop_info['ip']:
                                hop_info['hostname'] = hostname
                        except socket.herror:
                            pass
                        
                        hops.append(hop_info)
                        
                except ValueError:
                    continue
        
        return hops
    
    def _is_ip_address(self, text: str) -> bool:
        """Check if text is an IP address."""
        try:
            socket.inet_pton(socket.AF_INET, text)
            return True
        except socket.error:
            try:
                socket.inet_pton(socket.AF_INET6, text)
                return True
            except socket.error:
                return False
    
    def _get_network_interfaces(self) -> Dict[str, Any]:
        """Get network interface information."""
        if not PSUTIL_AVAILABLE:
            return {'error': 'psutil not available'}
        
        interfaces = {}
        
        try:
            for interface, addrs in psutil.net_if_addrs().items():
                interfaces[interface] = {
                    'addresses': [],
                    'stats': {}
                }
                
                for addr in addrs:
                    interfaces[interface]['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
                
                # Get interface statistics
                try:
                    stats = psutil.net_if_stats()[interface]
                    interfaces[interface]['stats'] = {
                        'is_up': stats.isup,
                        'duplex': stats.duplex,
                        'speed': stats.speed,
                        'mtu': stats.mtu
                    }
                except KeyError:
                    pass
            
            # Get I/O statistics
            io_counters = psutil.net_io_counters(pernic=True)
            for interface, counters in io_counters.items():
                if interface in interfaces:
                    interfaces[interface]['io'] = {
                        'bytes_sent': counters.bytes_sent,
                        'bytes_recv': counters.bytes_recv,
                        'packets_sent': counters.packets_sent,
                        'packets_recv': counters.packets_recv
                    }
        
        except Exception as e:
            return {'error': f"Failed to get interface info: {str(e)}"}
        
        return interfaces
    
    def _assess_latency(self, latency_ms: float) -> str:
        """Assess latency quality."""
        if latency_ms < 50:
            return 'excellent'
        elif latency_ms < 100:
            return 'good'
        elif latency_ms < 200:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_connectivity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall network connectivity."""
        assessment = {
            'overall': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        # Check DNS
        dns = results.get('dns', {})
        if not dns.get('resolvable'):
            assessment['issues'].append('DNS resolution failed')
            assessment['recommendations'].append('Check DNS configuration and network connectivity')
        else:
            assessment['recommendations'].append('DNS resolution is working correctly')
        
        # Check ping
        ping = results.get('ping', {})
        if not ping.get('reachable'):
            assessment['issues'].append('Host is not reachable via ping')
            assessment['recommendations'].append('Check firewall settings and network routing')
        else:
            latency = ping.get('latency_ms')
            if latency:
                latency_assessment = self._assess_latency(latency)
                assessment['recommendations'].append(f'Network latency is {latency_assessment} ({latency}ms)')
        
        # Check HTTP
        http = results.get('http', {})
        if not http.get('http_accessible') and not http.get('https_accessible'):
            assessment['issues'].append('HTTP/HTTPS services are not accessible')
            assessment['recommendations'].append('Check if web services are running and accessible')
        
        # Determine overall status
        if assessment['issues']:
            assessment['overall'] = 'degraded'
        else:
            assessment['overall'] = 'healthy'
        
        return assessment
    
    def monitor_network(self, 
                       target: str = '8.8.8.8',
                       duration: int = 300,
                       interval: int = 10) -> Dict[str, Any]:
        """
        Monitor network connectivity over time.
        
        Args:
            target: Hostname or IP to monitor
            duration: Monitoring duration in seconds
            interval: Check interval in seconds
        """
        import time
        
        monitoring_data = {
            'target': target,
            'start_time': datetime.now().isoformat(),
            'duration': duration,
            'interval': interval,
            'samples': []
        }
        
        try:
            start_time = time.time()
            samples = []
            
            while time.time() - start_time < duration:
                sample = {
                    'timestamp': datetime.now().isoformat(),
                    'elapsed': time.time() - start_time,
                }
                
                # Quick ping check
                ping_result = self._check_ping(target, 2, False)
                sample['ping'] = {
                    'reachable': ping_result.get('reachable', False),
                    'latency_ms': ping_result.get('latency_ms')
                }
                
                # Quick DNS check
                dns_result = self._check_dns(target, 2, False)
                sample['dns'] = {
                    'resolvable': dns_result.get('resolvable', False),
                    'resolve_time_ms': dns_result.get('resolve_time_ms')
                }
                
                samples.append(sample)
                time.sleep(interval)
            
            monitoring_data['samples'] = samples
            monitoring_data['end_time'] = datetime.now().isoformat()
            monitoring_data['summary'] = self._calculate_network_monitoring_summary(samples)
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Network monitoring failed: {e}")
            return self._error_result(f"Network monitoring failed: {str(e)}")
    
    def _calculate_network_monitoring_summary(self, samples: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from network monitoring data."""
        if not samples:
            return {}
        
        summary = {}
        
        # Ping statistics
        ping_latencies = [s['ping']['latency_ms'] for s in samples if s['ping']['latency_ms']]
        if ping_latencies:
            summary['ping'] = {
                'availability_percent': (len(ping_latencies) / len(samples)) * 100,
                'average_latency_ms': sum(ping_latencies) / len(ping_latencies),
                'max_latency_ms': max(ping_latencies),
                'min_latency_ms': min(ping_latencies),
            }
        
        # DNS statistics
        dns_times = [s['dns']['resolve_time_ms'] for s in samples if s['dns']['resolve_time_ms']]
        if dns_times:
            summary['dns'] = {
                'success_rate_percent': (len(dns_times) / len(samples)) * 100,
                'average_resolve_time_ms': sum(dns_times) / len(dns_times),
                'max_resolve_time_ms': max(dns_times),
                'min_resolve_time_ms': min(dns_times),
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
    """Legacy network check tool (maintains original interface)."""
    
    name = 'network_check'
    description = 'Check network connectivity'
    
    def __init__(self):
        self.enhanced_tool = NetworkCheckTool()
    
    def run(self, host: str = '8.8.8.8', **kwargs) -> Dict[str, Any]:
        """
        Check network connectivity with enhanced capabilities.
        
        Args:
            host: Hostname or IP to check
            **kwargs: Additional diagnostic parameters
            
        Returns:
            Network diagnostics results
        """
        return self.enhanced_tool.run(target=host, **kwargs)


# Example usage
if __name__ == "__main__":
    tool = NetworkCheckTool()
    
    # Comprehensive network diagnostics
    results = tool.run(
        target='google.com',
        checks=['dns', 'ping', 'http', 'ports'],
        detailed=True
    )
    
    print(f"Network Diagnostics for {results['target']}:")
    print(f"Overall Connectivity: {results['connectivity_assessment']['overall']}")
    
    dns = results['results']['dns']
    if dns['resolvable']:
        print(f"DNS: Resolved to {len(dns['resolved_ips'])} IPs in {dns['resolve_time_ms']}ms")
    
    ping = results['results']['ping']
    if ping['reachable']:
        print(f"Ping: Reachable with {ping['latency_ms']}ms latency")
    
    # Monitor network over time
    monitor_results = tool.monitor_network('8.8.8.8', duration=30, interval=5)
    print(f"Monitoring completed: {len(monitor_results['samples'])} samples")