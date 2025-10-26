"""
Enhanced Tool Registry for MCP Server
FIXED VERSION - Fully functional with proper imports and error handling
"""

import os
import sys
import asyncio
import inspect
import functools
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable, get_type_hints
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# Import utility functions with fallbacks
def validate_tool_definition(tool_def: Dict[str, Any]) -> bool:
    """Validate tool definition - basic implementation"""
    required_fields = ['name', 'description']
    for field in required_fields:
        if field not in tool_def or not tool_def[field]:
            logger.warning(f"Tool definition missing required field: {field}")
            return False
    return True

def safe_execute_tool(func: Callable, *args, **kwargs) -> Any:
    """Safely execute a tool function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise

def format_tool_result(result: Any) -> Dict[str, Any]:
    """Format tool result for consistent response"""
    if isinstance(result, dict):
        return result
    elif isinstance(result, (str, int, float, bool)) or result is None:
        return {"result": result}
    else:
        # Try to convert to dict or string representation
        try:
            if hasattr(result, '__dict__'):
                return result.__dict__
            else:
                return {"result": str(result)}
        except:
            return {"result": "Execution completed"}

def create_error_response(error: str, details: Any = None) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {"error": error}
    if details:
        response["details"] = details
    return response


class Tool:
    """
    Enhanced Tool representation with metadata and execution capabilities
    """
    
    def __init__(
        self,
        name: str,
        function: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        category: str = "general",
        version: str = "1.0.0",
        deprecated: bool = False,
        timeout: Optional[int] = None,
        max_workers: int = 5
    ):
        self.name = name
        self.function = function
        self.description = description or (function.__doc__ or "No description provided").strip()
        self.parameters = parameters or self._infer_parameters()
        self.returns = returns or self._infer_return_type()
        self.category = category
        self.version = version
        self.deprecated = deprecated
        self.timeout = timeout
        self.max_workers = max_workers
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._execution_count = 0
        self._success_count = 0
        self._error_count = 0
        
        # Validate tool definition
        self._validate()
        
        logger.debug(f"Created tool: {name} (category: {category})")

    def _infer_parameters(self) -> Dict[str, Any]:
        """Infer parameters from function signature"""
        try:
            sig = inspect.signature(self.function)
            type_hints = get_type_hints(self.function)
            
            parameters = {}
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                param_type = type_hints.get(param_name, Any)
                param_info = {
                    "type": getattr(param_type, '__name__', str(param_type)),
                    "required": param.default == param.empty,
                    "description": f"Parameter {param_name}"
                }
                
                if param.default != param.empty:
                    param_info["default"] = param.default
                
                parameters[param_name] = param_info
            
            required_params = [name for name, param in sig.parameters.items() 
                             if param.default == param.empty and name != 'self']
            
            return {
                "type": "object",
                "properties": parameters,
                "required": required_params
            }
        except Exception as e:
            logger.warning(f"Could not infer parameters for {self.name}: {e}")
            return {"type": "object", "properties": {}, "required": []}

    def _infer_return_type(self) -> Dict[str, Any]:
        """Infer return type from function type hints"""
        try:
            return_type = get_type_hints(self.function).get('return', Any)
            return {
                "type": getattr(return_type, '__name__', str(return_type)),
                "description": f"Result from {self.name}"
            }
        except Exception as e:
            logger.warning(f"Could not infer return type for {self.name}: {e}")
            return {"type": "any", "description": "Execution result"}

    def _validate(self):
        """Validate tool configuration"""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool name must be a non-empty string")
        
        if not callable(self.function):
            raise ValueError("Tool function must be callable")
        
        if not validate_tool_definition({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns
        }):
            logger.warning(f"Tool {self.name} validation had warnings")

    async def execute(self, parameters: Dict[str, Any], timeout: Optional[int] = None) -> Any:
        """
        Execute the tool with given parameters
        
        Args:
            parameters: Tool parameters
            timeout: Execution timeout in seconds
            
        Returns:
            Tool execution result
        """
        self._execution_count += 1
        start_time = datetime.now()
        
        try:
            # Use provided timeout or tool default
            execution_timeout = timeout or self.timeout
            
            # Check if function is async
            if inspect.iscoroutinefunction(self.function):
                # Async function execution
                if execution_timeout:
                    result = await asyncio.wait_for(
                        self.function(**parameters),
                        timeout=execution_timeout
                    )
                else:
                    result = await self.function(**parameters)
            else:
                # Sync function execution with thread pool
                loop = asyncio.get_event_loop()
                if execution_timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._thread_pool, 
                            functools.partial(safe_execute_tool, self.function, **parameters)
                        ),
                        timeout=execution_timeout
                    )
                else:
                    result = await loop.run_in_executor(
                        self._thread_pool,
                        functools.partial(safe_execute_tool, self.function, **parameters)
                    )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._success_count += 1
            
            logger.info(f"Tool {self.name} executed successfully in {execution_time:.3f}s")
            
            return format_tool_result(result)
            
        except asyncio.TimeoutError:
            self._error_count += 1
            error_msg = f"Tool {self.name} execution timed out after {execution_timeout}s"
            logger.error(error_msg)
            return create_error_response(error_msg)
            
        except Exception as e:
            self._error_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool {self.name} execution failed after {execution_time:.3f}s: {str(e)}"
            logger.error(error_msg)
            return create_error_response(error_msg)

    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        success_rate = (self._success_count / self._execution_count * 100 
                       if self._execution_count > 0 else 0)
        
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": round(success_rate, 2)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "category": self.category,
            "version": self.version,
            "deprecated": self.deprecated,
            "timeout": self.timeout,
            "stats": self.get_stats()
        }

    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)


class ToolRegistry:
    """
    Enhanced registry for managing tools with advanced features
    """
    
    def __init__(self, max_workers: int = 10):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._max_workers = max_workers
        self._lock = threading.RLock()
        self._registry_metrics = {
            "total_registrations": 0,
            "total_executions": 0,
            "total_errors": 0,
            "start_time": datetime.now()
        }
        
        logger.info("Initialized Tool Registry")

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        category: str = "general",
        version: str = "1.0.0",
        deprecated: bool = False,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Register a tool with the registry
        
        Args:
            name: Unique tool name
            function: Tool function (sync or async)
            description: Tool description
            parameters: Tool parameters schema
            returns: Return value schema
            category: Tool category
            version: Tool version
            deprecated: Whether tool is deprecated
            timeout: Default execution timeout
            
        Returns:
            True if registration successful
        """
        with self._lock:
            if name in self._tools:
                logger.warning(f"Tool '{name}' already registered, overwriting")
            
            try:
                tool = Tool(
                    name=name,
                    function=function,
                    description=description,
                    parameters=parameters,
                    returns=returns,
                    category=category,
                    version=version,
                    deprecated=deprecated,
                    timeout=timeout,
                    max_workers=min(3, self._max_workers)  # Limit per tool
                )
                
                self._tools[name] = tool
                
                # Update category index
                if category not in self._categories:
                    self._categories[category] = []
                if name not in self._categories[category]:
                    self._categories[category].append(name)
                
                self._registry_metrics["total_registrations"] += 1
                logger.info(f"Registered tool: {name} (category: {category})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register tool {name}: {e}")
                return False

    def register_tools_from_module(self, module) -> int:
        """
        Register all tools from a module that are decorated with @tool
        
        Args:
            module: Python module to scan for tools
            
        Returns:
            Number of tools registered
        """
        count = 0
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, '_is_mcp_tool') and obj._is_mcp_tool:
                tool_info = getattr(obj, '_mcp_tool_info', {})
                if self.register_tool(
                    name=tool_info.get('name', name),
                    function=obj,
                    description=tool_info.get('description', ''),
                    parameters=tool_info.get('parameters'),
                    returns=tool_info.get('returns'),
                    category=tool_info.get('category', 'general'),
                    version=tool_info.get('version', '1.0.0'),
                    deprecated=tool_info.get('deprecated', False),
                    timeout=tool_info.get('timeout')
                ):
                    count += 1
        
        logger.info(f"Registered {count} tools from module {module.__name__}")
        return count

    def register_simple_tool(self, name: str, func: Callable, description: str = "") -> bool:
        """
        Register a simple tool without complex configuration
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            
        Returns:
            True if successful
        """
        return self.register_tool(
            name=name,
            function=func,
            description=description,
            category="utility"
        )

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool from the registry"""
        with self._lock:
            if name in self._tools:
                tool = self._tools.pop(name)
                # Remove from category index
                category = tool.category
                if category in self._categories and name in self._categories[category]:
                    self._categories[category].remove(name)
                logger.info(f"Unregistered tool: {name}")
                return True
            return False

    async def execute_tool(
        self, 
        name: str, 
        parameters: Dict[str, Any], 
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a registered tool
        
        Args:
            name: Tool name
            parameters: Tool parameters
            timeout: Execution timeout
            
        Returns:
            Tool execution result
        """
        with self._lock:
            if name not in self._tools:
                error_msg = f"Tool '{name}' not found in registry"
                logger.error(error_msg)
                return create_error_response(error_msg)
            
            tool = self._tools[name]
            
            if tool.deprecated:
                logger.warning(f"Using deprecated tool: {name}")
        
        self._registry_metrics["total_executions"] += 1
        
        try:
            result = await tool.execute(parameters, timeout)
            return result
        except Exception as e:
            self._registry_metrics["total_errors"] += 1
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)
            return create_error_response(error_msg)

    def tool_exists(self, name: str) -> bool:
        """Check if a tool exists in the registry"""
        return name in self._tools

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition"""
        with self._lock:
            tool = self._tools.get(name)
            return tool.to_dict() if tool else None

    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tools, optionally filtered by category"""
        with self._lock:
            if category:
                tool_names = self._categories.get(category, [])
                tools = [self._tools[name] for name in tool_names if name in self._tools]
            else:
                tools = list(self._tools.values())
            
            return [tool.to_dict() for tool in tools]

    def get_categories(self) -> List[str]:
        """Get all available tool categories"""
        with self._lock:
            return list(self._categories.keys())

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tools in a specific category"""
        return self.list_tools(category=category)

    def get_registry_metrics(self) -> Dict[str, Any]:
        """Get registry-wide metrics"""
        with self._lock:
            uptime = (datetime.now() - self._registry_metrics["start_time"]).total_seconds()
            
            return {
                **self._registry_metrics,
                "uptime_seconds": round(uptime, 2),
                "total_tools": len(self._tools),
                "total_categories": len(self._categories),
                "tools_by_category": {
                    category: len(tools) 
                    for category, tools in self._categories.items()
                }
            }

    def clear_registry(self):
        """Clear all tools from registry"""
        with self._lock:
            self._tools.clear()
            self._categories.clear()
            logger.info("Cleared all tools from registry")

    def __contains__(self, name: str) -> bool:
        """Check if tool exists in registry"""
        return name in self._tools

    def __len__(self) -> int:
        """Get number of registered tools"""
        return len(self._tools)

    def __iter__(self):
        """Iterate over tool names"""
        return iter(self._tools.keys())


# Tool decorator for easy tool registration
def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    returns: Optional[Dict[str, Any]] = None,
    category: str = "general",
    version: str = "1.0.0",
    deprecated: bool = False,
    timeout: Optional[int] = None
):
    """
    Decorator for registering functions as MCP tools
    
    Usage:
        @tool(name="my_tool", description="My awesome tool")
        def my_tool(param1: str, param2: int) -> str:
            return f"Result: {param1} {param2}"
    """
    def decorator(func):
        # Store tool info in function attributes
        func._is_mcp_tool = True
        func._mcp_tool_info = {
            'name': name or func.__name__,
            'description': description or (func.__doc__ or "").strip(),
            'parameters': parameters,
            'returns': returns,
            'category': category,
            'version': version,
            'deprecated': deprecated,
            'timeout': timeout
        }
        return func
    return decorator


# Global tool registry instance
_tool_registry_instance = None

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = ToolRegistry()
    return _tool_registry_instance