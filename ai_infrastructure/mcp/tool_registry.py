"""
Enhanced Tool Registry for MCP Server
Provides dynamic tool registration, validation, and execution
"""

import asyncio
import inspect
import functools
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable, get_type_hints
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

from .utils import (
    validate_tool_definition,
    safe_execute_tool,
    format_tool_result,
    create_error_response,
    logger
)


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
        self.description = description or function.__doc__ or "No description provided"
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
                    
                param_info = {
                    "type": type_hints.get(param_name, Any).__name__,
                    "required": param.default == param.empty,
                    "description": f"Parameter {param_name}"
                }
                
                if param.default != param.empty:
                    param_info["default"] = param.default
                
                parameters[param_name] = param_info
            
            return {
                "type": "object",
                "properties": parameters,
                "required": [name for name, param in sig.parameters.items() 
                           if param.default == param.empty and name != 'self']
            }
        except Exception as e:
            logger.warning(f"Could not infer parameters for {self.name}: {e}")
            return {"type": "object", "properties": {}}

    def _infer_return_type(self) -> Dict[str, Any]:
        """Infer return type from function type hints"""
        try:
            return_type = get_type_hints(self.function).get('return', Any)
            return {
                "type": return_type.__name__,
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
        
        validate_tool_definition({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns
        })

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
                            functools.partial(self.function, **parameters)
                        ),
                        timeout=execution_timeout
                    )
                else:
                    result = await loop.run_in_executor(
                        self._thread_pool,
                        functools.partial(self.function, **parameters)
                    )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._success_count += 1
            
            logger.info(f"Tool {self.name} executed successfully in {execution_time:.3f}s")
            
            return format_tool_result(result)
            
        except asyncio.TimeoutError:
            self._error_count += 1
            error_msg = f"Tool {self.name} execution timed out after {execution_timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
            
        except Exception as e:
            self._error_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool {self.name} execution failed after {execution_time:.3f}s: {str(e)}"
            logger.error(error_msg)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": (self._success_count / self._execution_count * 100 
                           if self._execution_count > 0 else 0)
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
    
    def __init__(self, max_workers: int = 20):
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
                    max_workers=self._max_workers
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
    ) -> Any:
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
                raise ValueError(f"Tool '{name}' not found in registry")
            
            tool = self._tools[name]
            
            if tool.deprecated:
                logger.warning(f"Using deprecated tool: {name}")
        
        self._registry_metrics["total_executions"] += 1
        
        try:
            result = await tool.execute(parameters, timeout)
            return result
        except Exception as e:
            self._registry_metrics["total_errors"] += 1
            raise

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
                "uptime_seconds": uptime,
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
            'description': description or func.__doc__ or "",
            'parameters': parameters,
            'returns': returns,
            'category': category,
            'version': version,
            'deprecated': deprecated,
            'timeout': timeout
        }
        return func
    return decorator