"""
Enhanced Directory Scanner Tool
Provides comprehensive directory scanning with filtering, analysis, and metadata.
"""
import os
import fnmatch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DirectoryScannerTool:
    """Enhanced directory scanner with advanced filtering and analysis capabilities."""
    
    name = 'scan_directory'
    description = 'Scan directory contents with advanced filtering, sorting, and analysis'
    
    def run(self, 
            path: str, 
            pattern: str = '*',
            recursive: bool = False,
            include_files: bool = True,
            include_dirs: bool = True,
            file_types: Optional[List[str]] = None,
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
            modified_after: Optional[str] = None,
            modified_before: Optional[str] = None,
            sort_by: str = 'name',
            sort_order: str = 'asc',
            limit: Optional[int] = None,
            include_metadata: bool = False) -> Dict[str, Any]:
        """
        Scan directory with comprehensive filtering and analysis.
        
        Args:
            path: Directory path to scan
            pattern: File pattern to match (e.g., "*.txt", "image*.*")
            recursive: Whether to scan subdirectories
            include_files: Whether to include files in results
            include_dirs: Whether to include directories in results
            file_types: Filter by file extensions (e.g., ['.txt', '.py'])
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            modified_after: Filter files modified after date (YYYY-MM-DD)
            modified_before: Filter files modified before date (YYYY-MM-DD)
            sort_by: Sort by 'name', 'size', 'modified', 'type'
            sort_order: Sort order 'asc' or 'desc'
            limit: Maximum number of items to return
            include_metadata: Whether to include detailed file metadata
            
        Returns:
            Directory scanning results with comprehensive analysis
        """
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                return self._error_result(f"Path does not exist: {path}")
            
            if not path_obj.is_dir():
                return self._error_result(f"Path is not a directory: {path}")
            
            # Scan directory
            items = self._scan_directory(
                path_obj, pattern, recursive, include_files, include_dirs
            )
            
            # Apply filters
            filtered_items = self._apply_filters(
                items, file_types, min_size, max_size, modified_after, modified_before
            )
            
            # Sort results
            sorted_items = self._sort_items(filtered_items, sort_by, sort_order)
            
            # Apply limit
            if limit and limit > 0:
                sorted_items = sorted_items[:limit]
            
            # Add metadata if requested
            if include_metadata:
                sorted_items = self._add_metadata(sorted_items)
            
            # Generate statistics
            statistics = self._generate_statistics(sorted_items, path_obj)
            
            return {
                'directory': str(path_obj.absolute()),
                'scan_timestamp': datetime.now().isoformat(),
                'scan_parameters': {
                    'pattern': pattern,
                    'recursive': recursive,
                    'include_files': include_files,
                    'include_dirs': include_dirs,
                    'file_types': file_types,
                    'min_size': min_size,
                    'max_size': max_size,
                    'modified_after': modified_after,
                    'modified_before': modified_before,
                    'sort_by': sort_by,
                    'sort_order': sort_order,
                    'limit': limit
                },
                'statistics': statistics,
                'items': sorted_items,
                'success': True
            }
            
        except PermissionError as e:
            return self._error_result(f"Permission denied: {path}")
        except Exception as e:
            logger.error(f"Directory scan failed: {e}")
            return self._error_result(f"Scan failed: {str(e)}")
    
    def _scan_directory(self, 
                       path_obj: Path, 
                       pattern: str, 
                       recursive: bool,
                       include_files: bool,
                       include_dirs: bool) -> List[Dict[str, Any]]:
        """Scan directory and collect items."""
        items = []
        
        try:
            # Choose scan method
            if recursive:
                scanner = path_obj.rglob(pattern)
            else:
                scanner = path_obj.glob(pattern)
            
            for item in scanner:
                try:
                    # Skip if we can't access the item
                    if not item.exists():
                        continue
                    
                    # Apply type filters
                    if item.is_file() and not include_files:
                        continue
                    if item.is_dir() and not include_dirs:
                        continue
                    
                    # Get basic item info
                    item_info = self._get_basic_item_info(item)
                    if item_info:
                        items.append(item_info)
                        
                except (PermissionError, OSError) as e:
                    logger.debug(f"Could not access {item}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Scanning error: {e}")
        
        return items
    
    def _get_basic_item_info(self, item: Path) -> Optional[Dict[str, Any]]:
        """Get basic information about a directory item."""
        try:
            stat = item.stat()
            
            info = {
                'name': item.name,
                'path': str(item),
                'absolute_path': str(item.absolute()),
                'is_file': item.is_file(),
                'is_dir': item.is_dir(),
                'is_symlink': item.is_symlink(),
                'size': stat.st_size if item.is_file() else 0,
                'modified': stat.st_mtime,
                'modified_iso': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': stat.st_ctime,
                'created_iso': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'extension': item.suffix.lower() if item.is_file() else '',
                'parent': str(item.parent),
            }
            
            # Add permissions
            info.update(self._get_permissions_info(stat.st_mode))
            
            return info
            
        except (OSError, PermissionError) as e:
            logger.debug(f"Could not get info for {item}: {e}")
            return None
    
    def _get_permissions_info(self, mode: int) -> Dict[str, Any]:
        """Get file permission information."""
        return {
            'permissions': oct(mode)[-3:],
            'permissions_string': self._mode_to_string(mode),
            'is_readable': bool(mode & 0o444),  # Read permission
            'is_writable': bool(mode & 0o222),  # Write permission
            'is_executable': bool(mode & 0o111),  # Execute permission
        }
    
    def _mode_to_string(self, mode: int) -> str:
        """Convert file mode to string representation."""
        perm_chars = ['r', 'w', 'x']
        mode_str = bin(mode)[-9:]  # Last 9 bits for permissions
        result = []
        
        for i, bit in enumerate(mode_str):
            result.append(perm_chars[i % 3] if bit == '1' else '-')
        
        return ''.join(result)
    
    def _apply_filters(self, 
                      items: List[Dict[str, Any]],
                      file_types: Optional[List[str]],
                      min_size: Optional[int],
                      max_size: Optional[int],
                      modified_after: Optional[str],
                      modified_before: Optional[str]) -> List[Dict[str, Any]]:
        """Apply filters to scanned items."""
        filtered = items.copy()
        
        # Filter by file type
        if file_types:
            file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in file_types]
            filtered = [item for item in filtered 
                       if not item['is_file'] or item['extension'] in file_types]
        
        # Filter by size
        if min_size is not None:
            filtered = [item for item in filtered 
                       if not item['is_file'] or item['size'] >= min_size]
        
        if max_size is not None:
            filtered = [item for item in filtered 
                       if not item['is_file'] or item['size'] <= max_size]
        
        # Filter by modification date
        if modified_after:
            try:
                after_date = datetime.fromisoformat(modified_after).timestamp()
                filtered = [item for item in filtered if item['modified'] >= after_date]
            except ValueError:
                logger.warning(f"Invalid modified_after date: {modified_after}")
        
        if modified_before:
            try:
                before_date = datetime.fromisoformat(modified_before).timestamp()
                filtered = [item for item in filtered if item['modified'] <= before_date]
            except ValueError:
                logger.warning(f"Invalid modified_before date: {modified_before}")
        
        return filtered
    
    def _sort_items(self, 
                   items: List[Dict[str, Any]], 
                   sort_by: str, 
                   sort_order: str) -> List[Dict[str, Any]]:
        """Sort items based on specified criteria."""
        if not items:
            return items
        
        reverse = (sort_order.lower() == 'desc')
        
        sort_functions = {
            'name': lambda x: x['name'].lower(),
            'size': lambda x: x['size'],
            'modified': lambda x: x['modified'],
            'type': lambda x: (x['extension'], x['name'].lower()),
            'path': lambda x: x['path'].lower(),
        }
        
        sort_key = sort_functions.get(sort_by, sort_functions['name'])
        
        try:
            return sorted(items, key=sort_key, reverse=reverse)
        except Exception as e:
            logger.warning(f"Sorting failed, using default: {e}")
            return sorted(items, key=sort_functions['name'])
    
    def _add_metadata(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add additional metadata to items."""
        for item in items:
            if item['is_file']:
                item.update(self._get_file_metadata(Path(item['path'])))
            elif item['is_dir']:
                item.update(self._get_directory_metadata(Path(item['path'])))
        
        return items
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get additional file metadata."""
        metadata = {}
        
        try:
            # File type categorization
            metadata['file_type'] = self._categorize_file_type(file_path)
            
            # MIME type detection
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                metadata['mime_type'] = mime_type
            
        except Exception as e:
            logger.debug(f"Could not get additional metadata for {file_path}: {e}")
        
        return metadata
    
    def _get_directory_metadata(self, dir_path: Path) -> Dict[str, Any]:
        """Get additional directory metadata."""
        metadata = {}
        
        try:
            # Count items in directory
            items = list(dir_path.iterdir())
            metadata['item_count'] = len(items)
            metadata['file_count'] = len([item for item in items if item.is_file()])
            metadata['dir_count'] = len([item for item in items if item.is_dir()])
            
        except (PermissionError, OSError):
            metadata['item_count'] = 0
            metadata['file_count'] = 0
            metadata['dir_count'] = 0
        
        return metadata
    
    def _categorize_file_type(self, file_path: Path) -> str:
        """Categorize file by extension."""
        extension = file_path.suffix.lower()
        
        categories = {
            'document': {'.txt', '.md', '.pdf', '.doc', '.docx', '.rtf'},
            'code': {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h'},
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'},
            'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg'},
            'video': {'.mp4', '.avi', '.mov', '.mkv', '.webm'},
            'archive': {'.zip', '.tar', '.gz', '.7z', '.rar'},
            'spreadsheet': {'.xls', '.xlsx', '.csv'},
            'presentation': {'.ppt', '.pptx'},
        }
        
        for category, extensions in categories.items():
            if extension in extensions:
                return category
        
        return 'other'
    
    def _generate_statistics(self, items: List[Dict[str, Any]], path_obj: Path) -> Dict[str, Any]:
        """Generate comprehensive scan statistics."""
        files = [item for item in items if item['is_file']]
        directories = [item for item in items if item['is_dir']]
        
        total_size = sum(item['size'] for item in files)
        
        # File type distribution
        file_types = {}
        for item in files:
            file_type = item.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        # Size distribution
        size_ranges = {
            'tiny': len([f for f in files if f['size'] < 1024]),  # < 1KB
            'small': len([f for f in files if 1024 <= f['size'] < 1024*1024]),  # 1KB - 1MB
            'medium': len([f for f in files if 1024*1024 <= f['size'] < 1024*1024*10]),  # 1MB - 10MB
            'large': len([f for f in files if f['size'] >= 1024*1024*10]),  # >= 10MB
        }
        
        return {
            'total_items': len(items),
            'file_count': len(files),
            'directory_count': len(directories),
            'total_size': total_size,
            'total_size_human': self._format_size(total_size),
            'average_file_size': total_size / len(files) if files else 0,
            'file_type_distribution': file_types,
            'size_distribution': size_ranges,
            'oldest_item': min(items, key=lambda x: x['modified']) if items else None,
            'newest_item': max(items, key=lambda x: x['modified']) if items else None,
            'largest_file': max(files, key=lambda x: x['size']) if files else None,
        }
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        size = float(size_bytes)
        
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        
        return f"{size:.2f} {units[i]}"
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy directory scanner (maintains original interface)."""
    
    name = 'scan_directory'
    description = 'Scan directory contents'
    
    def __init__(self):
        self.enhanced_tool = DirectoryScannerTool()
    
    def run(self, path: str, pattern: str = '*', recursive: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Scan directory with enhanced capabilities.
        
        Args:
            path: Directory path to scan
            pattern: File pattern to match
            recursive: Whether to scan subdirectories
            **kwargs: Additional filtering options
            
        Returns:
            Directory scanning results
        """
        return self.enhanced_tool.run(
            path=path,
            pattern=pattern,
            recursive=recursive,
            **kwargs
        )


# Utility functions
def compare_directories(dir1: str, dir2: str, **kwargs) -> Dict[str, Any]:
    """Compare two directories and find differences."""
    scanner = DirectoryScannerTool()
    
    scan1 = scanner.run(dir1, **kwargs)
    scan2 = scanner.run(dir2, **kwargs)
    
    if not scan1['success'] or not scan2['success']:
        return {'error': 'Could not scan one or both directories'}
    
    items1 = {item['name']: item for item in scan1['items']}
    items2 = {item['name']: item for item in scan2['items']}
    
    common = set(items1.keys()) & set(items2.keys())
    only_in_dir1 = set(items1.keys()) - set(items2.keys())
    only_in_dir2 = set(items2.keys()) - set(items1.keys())
    
    return {
        'directory1': dir1,
        'directory2': dir2,
        'common_files': list(common),
        'only_in_dir1': list(only_in_dir1),
        'only_in_dir2': list(only_in_dir2),
        'comparison_timestamp': datetime.now().isoformat()
    }


# Example usage
if __name__ == "__main__":
    scanner = DirectoryScannerTool()
    
    # Basic scan
    result = scanner.run(".", pattern="*.py", recursive=True, include_metadata=True)
    print(f"Found {result['statistics']['file_count']} Python files")
    
    # Advanced scan with filtering
    result = scanner.run(
        ".",
        file_types=['.py', '.txt'],
        min_size=100,  # At least 100 bytes
        sort_by='size',
        sort_order='desc',
        limit=10,
        include_metadata=True
    )
    
    print("\nTop 10 largest Python/TXT files:")
    for item in result['items']:
        print(f"  {item['name']} - {scanner._format_size(item['size'])}")