"""
File Tools Package for MCP
Provides comprehensive file system operations, analysis, and management tools.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import core file tools
from .file_detector import Tool as FileTypeDetectionTool
from .file_info import Tool as FileInfoTool
from .directory_scanner import Tool as DirectoryScannerTool

# Optional tools with fallback
try:
    from .file_search import Tool as FileSearchTool
    FILE_SEARCH_AVAILABLE = True
except ImportError:
    FILE_SEARCH_AVAILABLE = False
    logger.warning("File search tool not available")

try:
    from .file_organizer import Tool as FileOrganizerTool
    FILE_ORGANIZER_AVAILABLE = True
except ImportError:
    FILE_ORGANIZER_AVAILABLE = False
    logger.warning("File organizer tool not available")

try:
    from .duplicate_finder import Tool as DuplicateFinderTool
    DUPLICATE_FINDER_AVAILABLE = True
except ImportError:
    DUPLICATE_FINDER_AVAILABLE = False
    logger.warning("Duplicate finder tool not available")

__all__ = [
    "FileTypeDetectionTool",
    "FileInfoTool", 
    "DirectoryScannerTool",
    "get_all_tools",
    "get_tool_by_name",
    "get_file_tools_statistics",
    "analyze_file_system",
    "batch_file_operations"
]

class FileToolsManager:
    """Manager for file tools with enhanced coordination and utilities."""
    
    def __init__(self):
        self.tools = self._discover_tools()
        self.tool_categories = self._categorize_tools()
    
    def _discover_tools(self) -> Dict[str, Any]:
        """Discover and initialize all available file tools."""
        tools = {
            'detect_file_type': FileTypeDetectionTool(),
            'get_file_info': FileInfoTool(),
            'scan_directory': DirectoryScannerTool(),
        }
        
        # Add optional tools if available
        if FILE_SEARCH_AVAILABLE:
            tools['search_files'] = FileSearchTool()
        
        if FILE_ORGANIZER_AVAILABLE:
            tools['organize_files'] = FileOrganizerTool()
            
        if DUPLICATE_FINDER_AVAILABLE:
            tools['find_duplicates'] = DuplicateFinderTool()
        
        return tools
    
    def _categorize_tools(self) -> Dict[str, List[str]]:
        """Categorize tools by functionality."""
        return {
            'analysis': [
                'detect_file_type',
                'get_file_info',
                'scan_directory',
            ],
            'search': [
                'search_files' if FILE_SEARCH_AVAILABLE else None,
            ],
            'organization': [
                'organize_files' if FILE_ORGANIZER_AVAILABLE else None,
                'find_duplicates' if DUPLICATE_FINDER_AVAILABLE else None,
            ]
        }
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool."""
        if tool_name not in self.tools:
            return None
        
        tool = self.tools[tool_name]
        return {
            'name': tool.name,
            'description': tool.description,
            'category': self._get_tool_category(tool_name),
            'available': True
        }
    
    def _get_tool_category(self, tool_name: str) -> str:
        """Get the category of a tool."""
        for category, tools in self.tool_categories.items():
            if tool_name in tools:
                return category
        return 'uncategorized'

# Global manager instance
_manager = FileToolsManager()

def get_all_tools() -> List[Any]:
    """
    Get instances of all available file tools.
    
    Returns:
        List of initialized tool instances
    """
    return list(_manager.tools.values())

def get_tool_by_name(name: str) -> Optional[Any]:
    """
    Get a specific file tool by name.
    
    Args:
        name: Tool name (e.g., 'get_file_info', 'scan_directory')
        
    Returns:
        Tool instance or None if not found
    """
    return _manager.tools.get(name)

def get_file_tools_statistics() -> Dict[str, Any]:
    """
    Get statistics and information about available file tools.
    
    Returns:
        File tools statistics and metadata
    """
    tools = get_all_tools()
    
    return {
        'total_tools': len(tools),
        'available_tools': list(_manager.tools.keys()),
        'tool_categories': _manager.tool_categories,
        'optional_tools': {
            'file_search': FILE_SEARCH_AVAILABLE,
            'file_organizer': FILE_ORGANIZER_AVAILABLE,
            'duplicate_finder': DUPLICATE_FINDER_AVAILABLE
        },
        'tools_info': {
            name: _manager.get_tool_info(name) 
            for name in _manager.tools.keys()
        }
    }

def analyze_file_system(path: str, depth: int = 2) -> Dict[str, Any]:
    """
    Perform comprehensive file system analysis using multiple tools.
    
    Args:
        path: Path to analyze
        depth: Analysis depth (1=basic, 2=detailed, 3=comprehensive)
        
    Returns:
        Comprehensive file system analysis
    """
    from datetime import datetime
    
    analysis = {
        'path': path,
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_depth': depth,
        'tools_used': []
    }
    
    # Basic analysis (always performed)
    try:
        # Get file/directory info
        info_tool = get_tool_by_name('get_file_info')
        if info_tool:
            analysis['basic_info'] = info_tool.run(path)
            analysis['tools_used'].append('get_file_info')
        
        # Scan directory if it's a directory
        path_obj = Path(path)
        if path_obj.is_dir():
            scanner_tool = get_tool_by_name('scan_directory')
            if scanner_tool:
                scan_result = scanner_tool.run(
                    path, 
                    recursive=(depth >= 2),
                    include_metadata=(depth >= 2)
                )
                analysis['directory_scan'] = scan_result
                analysis['tools_used'].append('scan_directory')
    
    except Exception as e:
        analysis['basic_analysis_error'] = str(e)
    
    # Detailed analysis (depth 2+)
    if depth >= 2:
        try:
            # File type analysis for directories
            if path_obj.is_dir():
                detector_tool = get_tool_by_name('detect_file_type')
                if detector_tool:
                    # Sample file type detection on first few files
                    if 'directory_scan' in analysis:
                        files = analysis['directory_scan']['items'][:10]  # Sample first 10
                        file_types = []
                        for file_info in files:
                            if file_info['is_file']:
                                type_result = detector_tool.run(file_info['path'])
                                file_types.append({
                                    'file': file_info['name'],
                                    'type_info': type_result
                                })
                        analysis['file_type_samples'] = file_types
                        analysis['tools_used'].append('detect_file_type')
                        
        except Exception as e:
            analysis['detailed_analysis_error'] = str(e)
    
    # Comprehensive analysis (depth 3)
    if depth >= 3:
        try:
            # Optional tools for deep analysis
            if DUPLICATE_FINDER_AVAILABLE and path_obj.is_dir():
                duplicate_tool = get_tool_by_name('find_duplicates')
                if duplicate_tool:
                    duplicates = duplicate_tool.run(path, recursive=True)
                    analysis['duplicate_analysis'] = duplicates
                    analysis['tools_used'].append('find_duplicates')
                    
        except Exception as e:
            analysis['comprehensive_analysis_error'] = str(e)
    
    analysis['tools_used'] = list(set(analysis['tools_used']))  # Remove duplicates
    return analysis

def batch_file_operations(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute multiple file operations in batch.
    
    Args:
        operations: List of operation specifications
            Example: [
                {
                    'tool': 'get_file_info',
                    'path': '/some/file.txt',
                    'parameters': {}  # Tool-specific parameters
                },
                {
                    'tool': 'scan_directory', 
                    'path': '/some/directory',
                    'parameters': {'recursive': True}
                }
            ]
        
    Returns:
        Batch operation results
    """
    results = {
        'total_operations': len(operations),
        'successful_operations': 0,
        'failed_operations': 0,
        'operations': []
    }
    
    for i, operation in enumerate(operations):
        try:
            tool_name = operation.get('tool')
            path = operation.get('path')
            parameters = operation.get('parameters', {})
            
            if not tool_name or not path:
                results['operations'].append({
                    'index': i,
                    'status': 'failed',
                    'error': 'Missing tool name or path'
                })
                results['failed_operations'] += 1
                continue
            
            tool = get_tool_by_name(tool_name)
            if not tool:
                results['operations'].append({
                    'index': i,
                    'status': 'failed', 
                    'error': f'Tool not found: {tool_name}'
                })
                results['failed_operations'] += 1
                continue
            
            # Execute operation
            operation_result = tool.run(path, **parameters)
            
            results['operations'].append({
                'index': i,
                'status': 'success',
                'tool': tool_name,
                'path': path,
                'result': operation_result
            })
            results['successful_operations'] += 1
            
        except Exception as e:
            results['operations'].append({
                'index': i,
                'status': 'failed',
                'error': str(e),
                'tool': operation.get('tool'),
                'path': operation.get('path')
            })
            results['failed_operations'] += 1
    
    return results

def create_file_system_report(path: str, report_type: str = 'summary') -> Dict[str, Any]:
    """
    Create a comprehensive file system report.
    
    Args:
        path: Path to analyze
        report_type: Type of report ('summary', 'detailed', 'security')
        
    Returns:
        File system report
    """
    analysis = analyze_file_system(path, depth=3 if report_type == 'detailed' else 2)
    
    report = {
        'report_type': report_type,
        'generated_at': analysis['analysis_timestamp'],
        'target_path': path,
    }
    
    if report_type == 'summary':
        report['summary'] = _create_summary_report(analysis)
    elif report_type == 'detailed':
        report['detailed_analysis'] = analysis
    elif report_type == 'security':
        report['security_analysis'] = _create_security_report(analysis)
    
    return report

def _create_summary_report(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary report from analysis data."""
    summary = {}
    
    if 'basic_info' in analysis:
        basic = analysis['basic_info']
        summary['type'] = basic.get('type', 'unknown')
        summary['size'] = basic.get('size_human', '0 B')
        summary['permissions'] = basic.get('permissions', 'unknown')
    
    if 'directory_scan' in analysis:
        scan = analysis['directory_scan']
        stats = scan.get('statistics', {})
        summary.update({
            'total_items': stats.get('total_items', 0),
            'files': stats.get('file_count', 0),
            'directories': stats.get('directory_count', 0),
            'total_size': stats.get('total_size_human', '0 B'),
        })
    
    return summary

def _create_security_report(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create a security-focused report."""
    security = {
        'permissions_analysis': {},
        'suspicious_files': [],
        'recommendations': []
    }
    
    # Analyze permissions
    if 'basic_info' in analysis:
        basic = analysis['basic_info']
        if basic.get('is_writable') and basic.get('type') == 'file':
            security['permissions_analysis']['world_writable'] = True
            security['recommendations'].append(
                "Consider restricting write permissions on critical files"
            )
    
    # Check for suspicious file types in directories
    if 'file_type_samples' in analysis:
        for sample in analysis['file_type_samples']:
            type_info = sample.get('type_info', {})
            mime_type = type_info.get('mime_type', '')
            if 'executable' in mime_type and not sample['file'].endswith(('.exe', '.bin')):
                security['suspicious_files'].append({
                    'file': sample['file'],
                    'reason': 'Unexpected executable file type'
                })
    
    return security

# Example usage and testing
if __name__ == "__main__":
    # Print available tools and statistics
    stats = get_file_tools_statistics()
    print("File Tools Package Statistics:")
    print(f"Total tools: {stats['total_tools']}")
    print("Available tools:")
    for tool_name in stats['available_tools']:
        tool_info = stats['tools_info'][tool_name]
        print(f"  - {tool_name}: {tool_info['description']} ({tool_info['category']})")
    
    # Test file system analysis
    print("\nTesting file system analysis:")
    analysis = analyze_file_system(".", depth=2)
    print(f"Analysis completed using {len(analysis['tools_used'])} tools")
    
    if 'basic_info' in analysis:
        basic = analysis['basic_info']
        print(f"Path: {basic.get('path', 'N/A')}")
        print(f"Type: {basic.get('type', 'N/A')}")
        print(f"Size: {basic.get('size_human', 'N/A')}")
    
    # Test batch operations
    print("\nTesting batch operations:")
    batch_ops = [
        {
            'tool': 'get_file_info',
            'path': __file__,
            'parameters': {'include_content_preview': True}
        },
        {
            'tool': 'scan_directory',
            'path': '.',
            'parameters': {'pattern': '*.py', 'limit': 5}
        }
    ]
    
    batch_results = batch_file_operations(batch_ops)
    print(f"Batch results: {batch_results['successful_operations']}/{batch_results['total_operations']} successful")