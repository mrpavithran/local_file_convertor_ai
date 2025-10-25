"""
MCP (Model Context Protocol) Package
Complete implementation for AI file system
"""

from .mcp_server import MCPServer, mcp_app, create_mcp_server, get_mcp_server
from .tool_registry import ToolRegistry, Tool
from .utils import (
    validate_tool_definition,
    safe_execute_tool,
    format_tool_result,
    create_error_response,
    logger,
    performance_monitor,
    cache,
    rate_limiter
)

# Create a global registry instance for backward compatibility
_registry_instance = ToolRegistry()

# Backward compatibility functions (direct implementation, no wrapper needed)
def get_all_tools():
    """Backward-compatible function - returns a flat list of all tools"""
    return _registry_instance.get_all_tools()

def get_tool(tool_name: str):
    """Get a specific tool by name"""
    return _registry_instance.get_tool(tool_name)

def get_tools_by_category(category: str):
    """Get all tools from a specific category"""
    return _registry_instance.get_tools_by_category(category)

def get_available_categories():
    """Get information about all available categories"""
    return _registry_instance.get_available_categories()

def get_tool_catalog():
    """Get a complete catalog of all tools with metadata"""
    return _registry_instance.get_tool_catalog()

def search_tools(query: str):
    """Search tools by name, description, or category"""
    return _registry_instance.search_tools(query)

def validate_tools():
    """Validate that all tools are working correctly"""
    return _registry_instance.validate_tools()

def refresh_tools():
    """Clear cache and reload all tools"""
    return _registry_instance.refresh_tools()

__version__ = "1.0.0"
__author__ = "AI File System Team"
__description__ = "Enhanced MCP Server for AI-powered file operations"

__all__ = [
    # Server
    "MCPServer",
    "mcp_app", 
    "create_mcp_server",
    "get_mcp_server",
    
    # Registry
    "ToolRegistry",
    "Tool",
    
    # Backward compatibility functions
    "get_all_tools",
    "get_tool",
    "get_tools_by_category", 
    "get_available_categories",
    "get_tool_catalog",
    "search_tools",
    "validate_tools",
    "refresh_tools",
    
    # Utilities
    "validate_tool_definition",
    "safe_execute_tool",
    "format_tool_result", 
    "create_error_response",
    "logger",
    "performance_monitor",
    "cache",
    "rate_limiter"
]