from .system_info import Tool as SystemInfoTool
from .disk_usage import Tool as DiskUsageTool
from .process_checker import Tool as ProcessCheckerTool
from .network_tools import Tool as NetworkTools
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ToolManager:
    """Enhanced tool manager with additional capabilities."""
    
    def __init__(self):
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize all available tools."""
        return [
            SystemInfoTool(),
            DiskUsageTool(), 
            ProcessCheckerTool(),
            NetworkTools()
        ]
    
    def get_all_tools(self) -> List[Any]:
        """Get all available tools."""
        return self.tools.copy()
    
    def get_tool_by_name(self, name: str) -> Optional[Any]:
        """Get a specific tool by name."""
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == name:
                return tool
        return None
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of all available tools with names and descriptions."""
        tool_info = []
        for tool in self.tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                tool_info.append({
                    'name': tool.name,
                    'description': tool.description
                })
        return tool_info
    
    def run_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Run a specific tool by name."""
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            return {'error': f"Tool '{tool_name}' not found"}
        
        try:
            return tool.run(**kwargs)
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return {'error': f"Tool execution failed: {str(e)}"}
    
    def run_comprehensive_diagnostics(self, 
                                    target: Optional[str] = None,
                                    include_network: bool = True,
                                    include_system: bool = True,
                                    include_storage: bool = True,
                                    include_processes: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive system diagnostics using all tools.
        
        Args:
            target: Specific target for network diagnostics
            include_network: Include network diagnostics
            include_system: Include system information
            include_storage: Include disk usage analysis
            include_processes: Include process checking
            
        Returns:
            Comprehensive system diagnostics
        """
        diagnostics = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'summary': {},
            'details': {}
        }
        
        try:
            # System information
            if include_system:
                system_tool = self.get_tool_by_name('system_info')
                if system_tool:
                    diagnostics['details']['system_info'] = system_tool.run()
            
            # Disk usage
            if include_storage:
                disk_tool = self.get_tool_by_name('disk_usage')
                if disk_tool:
                    diagnostics['details']['disk_usage'] = disk_tool.run(detailed=True)
            
            # Process checking
            if include_processes:
                process_tool = self.get_tool_by_name('process_checker')
                if process_tool:
                    diagnostics['details']['processes'] = process_tool.run()
            
            # Network diagnostics
            if include_network:
                network_tool = self.get_tool_by_name('network_check')
                if network_tool and target:
                    diagnostics['details']['network'] = network_tool.run(
                        target=target, 
                        detailed=True
                    )
                elif network_tool:
                    diagnostics['details']['network'] = network_tool.run(detailed=True)
            
            # Generate summary
            diagnostics['summary'] = self._generate_diagnostics_summary(diagnostics['details'])
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Comprehensive diagnostics failed: {e}")
            return {'error': f"Diagnostics failed: {str(e)}"}
    
    def _generate_diagnostics_summary(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary from diagnostics details."""
        summary = {
            'overall_status': 'healthy',
            'issues_found': [],
            'recommendations': [],
            'components_checked': list(details.keys())
        }
        
        # Analyze system info
        if 'system_info' in details:
            system_info = details['system_info']
            # Add system-specific checks
            pass
        
        # Analyze disk usage
        if 'disk_usage' in details:
            disk_info = details['disk_usage']
            analysis = disk_info.get('analysis', {})
            if analysis.get('status') in ['warning', 'critical']:
                summary['issues_found'].append(f"Disk usage: {analysis.get('message')}")
                summary['recommendations'].extend(analysis.get('recommendations', []))
        
        # Analyze processes
        if 'processes' in details:
            processes = details['processes']
            # Add process-specific checks
            pass
        
        # Analyze network
        if 'network' in details:
            network = details['network']
            assessment = network.get('connectivity_assessment', {})
            if assessment.get('overall') != 'healthy':
                summary['issues_found'].extend(assessment.get('issues', []))
                summary['recommendations'].extend(assessment.get('recommendations', []))
        
        # Update overall status
        if any('critical' in issue.lower() for issue in summary['issues_found']):
            summary['overall_status'] = 'critical'
        elif summary['issues_found']:
            summary['overall_status'] = 'warning'
        
        return summary
    
    def validate_tools(self) -> Dict[str, Any]:
        """Validate that all tools are working correctly."""
        validation_results = {}
        
        for tool in self.tools:
            tool_name = getattr(tool, 'name', 'unknown')
            try:
                # Test each tool with a basic run
                result = tool.run()
                validation_results[tool_name] = {
                    'status': 'healthy',
                    'result_keys': list(result.keys()) if isinstance(result, dict) else 'non_dict_output'
                }
            except Exception as e:
                validation_results[tool_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return validation_results


# Maintain original function for backward compatibility
def get_all_tools():
    """Original function to get all tools."""
    return [SystemInfoTool(), DiskUsageTool(), ProcessCheckerTool(), NetworkTools()]


# Additional utility functions
def get_tool_names() -> List[str]:
    """Get names of all available tools."""
    tools = get_all_tools()
    return [tool.name for tool in tools if hasattr(tool, 'name')]


def get_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all available tools."""
    tools = get_all_tools()
    descriptions = {}
    for tool in tools:
        if hasattr(tool, 'name') and hasattr(tool, 'description'):
            descriptions[tool.name] = tool.description
    return descriptions


# Example usage
if __name__ == "__main__":
    # Original usage
    tools = get_all_tools()
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    # Enhanced usage
    manager = ToolManager()
    
    # Get tool information
    available_tools = manager.get_available_tools()
    print("Available tools:")
    for tool in available_tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Run comprehensive diagnostics
    diagnostics = manager.run_comprehensive_diagnostics(target='google.com')
    print(f"Overall status: {diagnostics['summary']['overall_status']}")
    
    # Validate tools
    validation = manager.validate_tools()
    print("Tool validation:", validation)