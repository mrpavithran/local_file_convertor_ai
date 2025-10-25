# tool_registry_wrapper.py

from .tool_registry import ToolRegistry

# Create a singleton registry instance to reuse
_registry_instance = ToolRegistry()

def get_all_tools():
    """
    Backward-compatible function.
    Returns a flat list of all tools using the ToolRegistry.
    """
    return _registry_instance.get_all_tools()

def get_tool(tool_name: str):
    """
    Get a specific tool by name.
    """
    return _registry_instance.get_tool(tool_name)

def get_tools_by_category(category: str):
    """
    Get all tools from a specific category.
    """
    return _registry_instance.get_tools_by_category(category)

def get_available_categories():
    """
    Get information about all available categories.
    """
    return _registry_instance.get_available_categories()

def get_tool_catalog():
    """
    Get a complete catalog of all tools with metadata.
    """
    return _registry_instance.get_tool_catalog()

def search_tools(query: str):
    """
    Search tools by name, description, or category.
    """
    return _registry_instance.search_tools(query)

def validate_tools():
    """
    Validate that all tools are working correctly.
    """
    return _registry_instance.validate_tools()

def refresh_tools():
    """
    Clear cache and reload all tools.
    """
    return _registry_instance.refresh_tools()
