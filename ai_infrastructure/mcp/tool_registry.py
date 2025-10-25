"""
Enhanced Tool Registry for MCP server with async support and better tool management.
"""
import inspect
from typing import Any, Dict, List, Optional, Callable
import importlib
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters_schema = parameters
        self.required_parameters = [
            param for param, schema in parameters.items() 
            if schema.get('required', False)
        ]
        self.is_async = False
        self.timeout = 30  # Default timeout in seconds
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool."""
        pass
    
    def validate_arguments(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against the parameter schema."""
        # Check required parameters
        for param in self.required_parameters:
            if param not in args:
                return f"Missing required parameter: {param}"
        
        # Check parameter types (basic validation)
        for param, value in args.items():
            if param in self.parameters_schema:
                expected_type = self.parameters_schema[param].get('type')
                if expected_type and not isinstance(value, eval(expected_type)):
                    return f"Parameter {param} should be of type {expected_type}"
        
        return None

class AsyncTool(BaseTool):
    """Base class for async tools."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        super().__init__(name, description, parameters)
        self.is_async = True
    
    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Execute the tool asynchronously."""
        pass

class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]
    
    def list_tools(self) -> Dict[str, BaseTool]:
        """List all registered tools."""
        return self._tools.copy()
    
    async def auto_discover_tools(self, package_path: str = "package.tools"):
        """Auto-discover and register tools from a package."""
        try:
            module = importlib.import_module(package_path)
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (inspect.isclass(attr) and 
                    issubclass(attr, BaseTool) and 
                    attr not in [BaseTool, AsyncTool]):
                    
                    try:
                        tool_instance = attr()
                        self.register_tool(tool_instance)
                    except Exception as e:
                        print(f"Failed to instantiate tool {attr_name}: {e}")
                        
        except ImportError as e:
            print(f"Could not import tools package {package_path}: {e}")
    
    async def cleanup(self):
        """Clean up tool resources."""
        for tool in self._tools.values():
            if hasattr(tool, 'close'):
                if inspect.iscoroutinefunction(tool.close):
                    await tool.close()
                else:
                    tool.close()