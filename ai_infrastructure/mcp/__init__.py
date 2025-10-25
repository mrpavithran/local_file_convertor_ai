"""
MCP (Model Context Protocol) Package

A comprehensive toolkit for building MCP servers with tool management,
execution, and protocol compliance.

Key Components:
- MCPServer: FastAPI-based server with tool execution endpoints
- ToolRegistry: Tool discovery, registration, and management
- BaseTool: Abstract base class for creating tools
- AsyncTool: Base class for asynchronous tools

Example usage:
    >>> from mcp_package import MCPServer, ToolRegistry, BaseTool
    >>> 
    >>> # Create a custom tool
    >>> class CalculatorTool(BaseTool):
    >>>     def __init__(self):
    >>>         super().__init__(
    >>>             name="calculator",
    >>>             description="Perform basic arithmetic operations",
    >>>             parameters={
    >>>                 "operation": {"type": "str", "required": True, "enum": ["add", "subtract", "multiply", "divide"]},
    >>>                 "a": {"type": "float", "required": True},
    >>>                 "b": {"type": "float", "required": True}
    >>>             }
    >>>         )
    >>>     
    >>>     def run(self, operation: str, a: float, b: float) -> float:
    >>>         if operation == "add":
    >>>             return a + b
    >>>         elif operation == "subtract":
    >>>             return a - b
    >>>         elif operation == "multiply":
    >>>             return a * b
    >>>         elif operation == "divide":
    >>>             if b == 0:
    >>>                 raise ValueError("Cannot divide by zero")
    >>>             return a / b
    >>> 
    >>> # Create and run server
    >>> server = MCPServer()
    >>> server.registry.register_tool(CalculatorTool())
    >>> server.run()
"""

__version__ = "1.0.0"
__author__ = "MCP Package Maintainers"
__email__ = "mcp@example.com"

# Core components
from .mcp_server import MCPServer, app
from .tool_registry import ToolRegistry, BaseTool, AsyncTool

# Protocol types and models
try:
    from .models import (
        RunRequest,
        RunResponse,
        ToolInfo,
        HealthResponse,
        ListToolsResponse,
        ErrorResponse
    )
    from .protocol import MCPProtocol, MCPMessage, MCPMessageType
except ImportError:
    # Fallback for basic installation
    pass

# Utility functions and helpers
from .utils import (
    create_tool_from_function,
    validate_tool_parameters,
    format_error_response,
    generate_request_id
)

# Client implementation (optional)
try:
    from .client import MCPClient
except ImportError:
    pass

# Exports for public API
__all__ = [
    # Core classes
    "MCPServer",
    "ToolRegistry", 
    "BaseTool",
    "AsyncTool",
    
    # FastAPI app for customization
    "app",
    
    # Models (if available)
    "RunRequest",
    "RunResponse", 
    "ToolInfo",
    "HealthResponse",
    "ListToolsResponse",
    "ErrorResponse",
    "MCPProtocol",
    "MCPMessage",
    "MCPMessageType",
    
    # Utilities
    "create_tool_from_function",
    "validate_tool_parameters", 
    "format_error_response",
    "generate_request_id",
    
    # Client (if available)
    "MCPClient",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Package configuration
class MCPConfig:
    """Package-level configuration defaults."""
    
    # Server defaults
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8765
    DEFAULT_TIMEOUT = 30
    
    # Tool defaults
    DEFAULT_TOOL_TIMEOUT = 30
    MAX_TOOL_TIMEOUT = 300
    
    # Security defaults
    ENABLE_AUTH = False
    ALLOWED_ORIGINS = ["*"]
    
    # Protocol version
    PROTOCOL_VERSION = "2024-11-05"

# Convenience functions for common operations
def create_server(
    tools_package: str = "tools",
    host: str = None,
    port: int = None,
    enable_auth: bool = None,
    **kwargs
) -> MCPServer:
    """
    Create a pre-configured MCP server with auto-discovered tools.
    
    Args:
        tools_package: Python package path to discover tools
        host: Server host address
        port: Server port
        enable_auth: Whether to enable authentication
        **kwargs: Additional server configuration
    
    Returns:
        Configured MCPServer instance
    """
    server = MCPServer(**kwargs)
    
    # Auto-discover tools
    server.registry.auto_discover_tools(tools_package)
    
    # Apply configuration
    if host is not None:
        server.host = host
    if port is not None:
        server.port = port
    if enable_auth is not None:
        server.enable_auth = enable_auth
    
    return server

def quick_start(
    tools: list = None,
    host: str = "0.0.0.0", 
    port: int = 8765,
    **kwargs
):
    """
    Quick start function to create and run an MCP server.
    
    Args:
        tools: List of tool instances or tool classes to register
        host: Server host address
        port: Server port
        **kwargs: Additional server configuration
    """
    server = create_server(host=host, port=port, **kwargs)
    
    # Register provided tools
    if tools:
        for tool in tools:
            if isinstance(tool, type):
                # Tool class, instantiate it
                server.registry.register_tool(tool())
            else:
                # Tool instance
                server.registry.register_tool(tool)
    
    print(f"Starting MCP server on {host}:{port}")
    print(f"Registered tools: {list(server.registry.list_tools().keys())}")
    
    # Start server
    server.run()

# Export configuration and convenience functions
__all__.extend(["MCPConfig", "create_server", "quick_start"])

# Runtime configuration
import os

# Environment variable configuration
def load_from_env():
    """Load configuration from environment variables."""
    config = {}
    
    # Server configuration
    config['host'] = os.getenv('MCP_HOST', MCPConfig.DEFAULT_HOST)
    config['port'] = int(os.getenv('MCP_PORT', MCPConfig.DEFAULT_PORT))
    config['enable_auth'] = os.getenv('MCP_ENABLE_AUTH', str(MCPConfig.ENABLE_AUTH)).lower() == 'true'
    
    # Tool configuration
    config['default_timeout'] = int(os.getenv('MCP_DEFAULT_TIMEOUT', MCPConfig.DEFAULT_TIMEOUT))
    config['max_timeout'] = int(os.getenv('MCP_MAX_TIMEOUT', MCPConfig.MAX_TOOL_TIMEOUT))
    
    return config

# Package initialization
def initialize_package():
    """Initialize package-wide settings and checks."""
    # Check for required dependencies
    try:
        import fastapi
        import pydantic
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "MCP package requires fastapi, pydantic, and uvicorn. "
            "Install with: pip install fastapi pydantic uvicorn"
        ) from e
    
    # Set up logging
    import logging
    logging.getLogger("mcp_package").addHandler(logging.NullHandler())
    
    print(f"MCP Package v{__version__} initialized successfully")

# Auto-initialize on import
initialize_package()