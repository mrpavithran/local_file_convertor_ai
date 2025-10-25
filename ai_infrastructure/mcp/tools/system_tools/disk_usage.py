import shutil
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

class DiskUsageTool:
    """Enhanced disk usage analysis tool with comprehensive storage diagnostics."""
    
    name = 'disk_usage'
    description = 'Comprehensive disk usage analysis including partitions, trends, and recommendations'
    
    def run(self, 
            path: str = '/',
            detailed: bool = True,
            analyze_trends: bool = False,
            threshold_warning: int = 80,
            threshold_critical: int = 90) -> Dict[str, Any]:
        """
        Perform comprehensive disk usage analysis.
        
        Args:
            path: Path to analyze (default: root)
            detailed: Include detailed partition information
            analyze_trends: Analyze usage trends over time
            threshold_warning: Usage percentage for warning (default: 80%)
            threshold_critical: Usage percentage for critical (default: 90%)
            
        Returns:
            Comprehensive disk usage analysis
        """
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'analyzed_path': path,
                'basic_usage': self._get_basic_usage(path),
                'analysis': {}
            }
            
            if detailed:
                results['detailed_analysis'] = self._get_detailed_analysis(path)
                results['partition_info'] = self._get_partition_info()
                results['io_statistics'] = self._get_io_statistics()
            
            results['analysis'] = self._analyze_usage(results, threshold_warning, threshold_critical)
            
            if analyze_trends:
                results['trend_analysis'] = self._analyze_usage_trends(path)
            
            logger.info(f"Disk usage analysis completed for {path}")
            return results
            
        except Exception as e:
            logger.error(f"Disk usage analysis failed: {e}")
            return self._error_result(f"Disk analysis failed: {str(e)}")
    
    def _get_basic_usage(self, path: str) -> Dict[str, Any]:
        """Get basic disk usage statistics."""
        try:
            total, used, free = shutil.disk_usage(path)
            
            return {
                'total_bytes': total,
                'used_bytes': used,
                'free_bytes': free,
                'total_gb': round(total / (1024**3), 2),
                'used_gb': round(used / (1024**3), 2),
                'free_gb': round(free / (1024**3), 2),
                'usage_percent': round((used / total) * 100, 2) if total > 0 else 0
            }
        except Exception as e:
            return {'error': f"Failed to get disk usage: {str(e)}"}
    
    def _get_detailed_analysis(self, path: str) -> Dict[str, Any]:
        """Get detailed disk analysis using psutil."""
        try:
            if not hasattr(psutil, 'disk_usage'):
                return {'error': 'psutil not available for detailed analysis'}
            
            usage = psutil.disk_usage(path)
            
            return {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'percent': usage.percent,
                'inodes': self._get_inode_usage(path) if hasattr(os, 'statvfs') else None
            }
        except Exception as e:
            return {'error': f"Detailed analysis failed: {str(e)}"}
    
    def _get_partition_info(self) -> List[Dict[str, Any]]:
        """Get information about all disk partitions."""
        partitions = []
        
        try:
            for partition in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    partition_info = {
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'opts': partition.opts,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'usage_percent': usage.percent
                    }
                    partitions.append(partition_info)
                except (PermissionError, OSError):
                    # Skip partitions we can't access
                    continue
            
            # Sort by usage percentage (descending)
            partitions.sort(key=lambda x: x.get('usage_percent', 0), reverse=True)
            
        except Exception as e:
            logger.warning(f"Could not get partition info: {e}")
        
        return partitions
    
    def _get_io_statistics(self) -> Dict[str, Any]:
        """Get disk I/O statistics."""
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                return {
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_time_ms': io_counters.read_time,
                    'write_time_ms': io_counters.write_time
                }
        except Exception as e:
            logger.warning(f"Could not get I/O statistics: {e}")
        
        return {'error': 'I/O statistics not available'}
    
    def _get_inode_usage(self, path: str) -> Optional[Dict[str, Any]]:
        """Get inode usage statistics (Unix/Linux only)."""
        try:
            stat = os.statvfs(path)
            inode_total = stat.f_files
            inode_free = stat.f_ffree
            inode_used = inode_total - inode_free
            
            return {
                'total': inode_total,
                'used': inode_used,
                'free': inode_free,
                'usage_percent': round((inode_used / inode_total) * 100, 2) if inode_total > 0 else 0
            }
        except (AttributeError, OSError):
            return None
    
    def _analyze_usage(self, 
                      results: Dict[str, Any], 
                      warning_threshold: int, 
                      critical_threshold: int) -> Dict[str, Any]:
        """Analyze disk usage and provide recommendations."""
        basic_usage = results.get('basic_usage', {})
        usage_percent = basic_usage.get('usage_percent', 0)
        
        analysis = {
            'status': 'normal',
            'level': 'info',
            'message': 'Disk usage is within normal limits',
            'recommendations': [],
            'warnings': []
        }
        
        # Determine status based on thresholds
        if usage_percent >= critical_threshold:
            analysis.update({
                'status': 'critical',
                'level': 'error',
                'message': f'Disk usage is critical ({usage_percent}%)'
            })
            analysis['warnings'].append(f"Immediate action required: Disk usage at {usage_percent}%")
            
        elif usage_percent >= warning_threshold:
            analysis.update({
                'status': 'warning',
                'level': 'warning',
                'message': f'Disk usage is high ({usage_percent}%)'
            })
            analysis['warnings'].append(f"Warning: Disk usage at {usage_percent}%")
        
        # Generate recommendations
        free_gb = basic_usage.get('free_gb', 0)
        if free_gb < 1:
            analysis['recommendations'].append("Less than 1GB free - consider immediate cleanup")
        elif free_gb < 5:
            analysis['recommendations'].append("Low free space - consider cleaning temporary files")
        
        if usage_percent > 50:
            analysis['recommendations'].append("Consider archiving old files or expanding storage")
        
        # Add partition-specific recommendations
        partitions = results.get('partition_info', [])
        critical_partitions = [p for p in partitions if p.get('usage_percent', 0) >= critical_threshold]
        if critical_partitions:
            analysis['recommendations'].extend([
                f"Critical usage on {len(critical_partitions)} partition(s)",
                "Check largest directories for cleanup opportunities"
            ])
        
        return analysis
    
    def _analyze_usage_trends(self, path: str) -> Dict[str, Any]:
        """Analyze disk usage trends (placeholder for historical analysis)."""
        # In a real implementation, this would compare with historical data
        return {
            'note': 'Trend analysis requires historical data collection',
            'suggestions': [
                'Implement periodic disk usage logging',
                'Compare with previous measurements',
                'Monitor growth rate over time'
            ]
        }
    
    def find_large_files(self, 
                        path: str = '/', 
                        limit: int = 20, 
                        min_size_mb: int = 100) -> Dict[str, Any]:
        """
        Find large files in the specified directory.
        
        Args:
            path: Directory to search
            limit: Maximum number of files to return
            min_size_mb: Minimum file size in MB to include
            
        Returns:
            List of large files with sizes
        """
        large_files = []
        min_size_bytes = min_size_mb * 1024 * 1024
        
        try:
            for root, dirs, files in os.walk(path):
                # Skip system directories that might cause permission issues
                if any(skip in root for skip in ['/proc', '/sys', '/dev']):
                    continue
                    
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        size = os.path.getsize(filepath)
                        if size >= min_size_bytes:
                            large_files.append({
                                'path': filepath,
                                'size_bytes': size,
                                'size_mb': round(size / (1024 * 1024), 2),
                                'size_gb': round(size / (1024 ** 3), 2)
                            })
                            
                            # Stop if we've found enough files
                            if len(large_files) >= limit * 2:  # Collect extra for sorting
                                break
                                
                    except (OSError, PermissionError):
                        continue
                
                if len(large_files) >= limit * 2:
                    break
            
            # Sort by size descending and limit results
            large_files.sort(key=lambda x: x['size_bytes'], reverse=True)
            large_files = large_files[:limit]
            
            return {
                'large_files': large_files,
                'total_found': len(large_files),
                'min_size_mb': min_size_mb,
                'search_path': path
            }
            
        except Exception as e:
            return self._error_result(f"Large file search failed: {str(e)}")
    
    def get_directory_sizes(self, 
                           path: str = '/', 
                           top_n: int = 10) -> Dict[str, Any]:
        """
        Get sizes of top-level directories.
        
        Args:
            path: Path to analyze
            top_n: Number of largest directories to return
            
        Returns:
            Directory size analysis
        """
        try:
            dir_sizes = []
            
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    try:
                        total_size = 0
                        for dirpath, dirnames, filenames in os.walk(item_path):
                            for filename in filenames:
                                try:
                                    filepath = os.path.join(dirpath, filename)
                                    total_size += os.path.getsize(filepath)
                                except (OSError, PermissionError):
                                    continue
                        
                        dir_sizes.append({
                            'directory': item,
                            'size_bytes': total_size,
                            'size_mb': round(total_size / (1024 * 1024), 2),
                            'size_gb': round(total_size / (1024 ** 3), 2)
                        })
                    except (OSError, PermissionError):
                        continue
            
            # Sort by size descending
            dir_sizes.sort(key=lambda x: x['size_bytes'], reverse=True)
            
            return {
                'directory_sizes': dir_sizes[:top_n],
                'analyzed_path': path,
                'total_directories': len(dir_sizes)
            }
            
        except Exception as e:
            return self._error_result(f"Directory size analysis failed: {str(e)}")
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy disk usage tool (maintains original interface)."""
    
    name = 'disk_usage'
    description = 'Disk usage statistics'
    
    def __init__(self):
        self.enhanced_tool = DiskUsageTool()
    
    def run(self, path: str = '/', **kwargs) -> Dict[str, Any]:
        """
        Get disk usage statistics with enhanced capabilities.
        
        Args:
            path: Path to analyze
            **kwargs: Additional parameters for enhanced analysis
            
        Returns:
            Disk usage statistics
        """
        # Maintain backward compatibility with simple output
        if not kwargs:
            total, used, free = shutil.disk_usage(path)
            return {
                'total': total,
                'used': used,
                'free': free,
                'total_gb': round(total / (1024**3), 2),
                'used_gb': round(used / (1024**3), 2),
                'free_gb': round(free / (1024**3), 2),
                'usage_percent': round((used / total) * 100, 2)
            }
        else:
            return self.enhanced_tool.run(path=path, **kwargs)


# Example usage
if __name__ == "__main__":
    tool = DiskUsageTool()
    
    # Basic usage (like original tool)
    basic_results = tool.run('/')
    print(f"Basic disk usage: {basic_results['basic_usage']['usage_percent']}%")
    
    # Detailed analysis
    detailed_results = tool.run('/', detailed=True)
    print(f"Status: {detailed_results['analysis']['status']}")
    
    # Find large files
    large_files = tool.find_large_files('/', limit=5, min_size_mb=100)
    print(f"Found {len(large_files['large_files'])} large files")
    
    # Directory analysis
    dir_sizes = tool.get_directory_sizes('/', top_n=5)
    for dir_info in dir_sizes['directory_sizes']:
        print(f"{dir_info['directory']}: {dir_info['size_gb']} GB")