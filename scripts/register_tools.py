#!/usr/bin/env python3
"""
MCP (Model Context Protocol) tool registration script.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


def register_mcp_tools(discover: bool = True, 
                      tool_categories: Optional[List[str]] = None,
                      register_all: bool = False) -> bool:
    """
    Register MCP tools for AI File System.
    
    Args:
        discover: Whether to discover tools automatically
        tool_categories: Specific tool categories to register
        register_all: Register all discovered tools
        
    Returns:
        True if registration successful
    """
    console = Console()
    
    console.print(Panel.fit(
        "🛠️ MCP Tool Registration for AI File System",
        style="bold blue"
    ))
    
    try:
        # Discover available tools
        if discover:
            console.print("🔍 Discovering available tools...")
            tools = discover_tools()
        else:
            tools = {}
        
        # Filter tools by category if specified
        if tool_categories:
            filtered_tools = {}
            for category in tool_categories:
                if category in tools:
                    filtered_tools[category] = tools[category]
            tools = filtered_tools
        
        if not tools:
            console.print("❌ No tools discovered")
            return False
        
        # Display discovered tools
        display_tool_discovery(tools, console)
        
        # Register tools
        console.print("📝 Registering tools...")
        registration_results = register_tools(tools, register_all)
        
        # Display registration results
        display_registration_results(registration_results, console)
        
        # Create tool registry
        console.print("💾 Creating tool registry...")
        if create_tool_registry(registration_results):
            console.print("✅ Tool registry created")
        else:
            console.print("❌ Failed to create tool registry")
        
        # Test registered tools
        console.print("🧪 Testing registered tools...")
        test_results = test_registered_tools(registration_results)
        display_test_results(test_results, console)
        
        console.print("✅ MCP tool registration completed!")
        return True
        
    except Exception as e:
        console.print(f"❌ Tool registration failed: {e}")
        return False


def discover_tools() -> Dict[str, List[Dict[str, Any]]]:
    """Discover available MCP tools in the tools directory."""
    tools_dir = Path(__file__).parent.parent / "ai_infrastructure" / "mcp" / "tools"
    tools = {}
    
    if not tools_dir.exists():
        return tools
    
    # Look for tool categories (subdirectories)
    for category_dir in tools_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('__'):
            category = category_dir.name
            tools[category] = []
            
            # Look for Python files in the category directory
            for tool_file in category_dir.glob("*.py"):
                if not tool_file.name.startswith('__'):
                    tool_info = analyze_tool_file(tool_file, category)
                    if tool_info:
                        tools[category].append(tool_info)
    
    return tools


def analyze_tool_file(tool_file: Path, category: str) -> Optional[Dict[str, Any]]:
    """Analyze a tool file to extract tool information."""
    try:
        # Read the file content
        with open(tool_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tool_name = tool_file.stem
        tool_module = f"ai_infrastructure.mcp.tools.{category}.{tool_name}"
        
        # Basic tool info
        tool_info = {
            'name': tool_name,
            'category': category,
            'file_path': str(tool_file),
            'module': tool_module,
            'functions': [],
            'class_name': None,
            'description': extract_tool_description(content)
        }
        
        # Try to import the module dynamically
        spec = importlib.util.spec_from_file_location(tool_module, tool_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find tool classes and functions
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and not name.startswith('_'):
                    # Check if it's a tool class
                    if has_tool_methods(obj):
                        tool_info['class_name'] = name
                        tool_info['functions'] = extract_class_methods(obj)
                
                elif inspect.isfunction(obj) and not name.startswith('_'):
                    # Check if it's a tool function
                    if is_tool_function(obj):
                        tool_info['functions'].append({
                            'name': name,
                            'type': 'function',
                            'signature': str(inspect.signature(obj)),
                            'docstring': inspect.getdoc(obj) or ''
                        })
        
        return tool_info if tool_info['functions'] else None
        
    except Exception as e:
        print(f"Warning: Could not analyze {tool_file}: {e}")
        return None


def has_tool_methods(cls) -> bool:
    """Check if a class has tool-like methods."""
    methods = [name for name, _ in inspect.getmembers(cls, predicate=inspect.ismethod)]
    tool_indicators = ['execute', 'process', 'convert', 'analyze', 'enhance']
    return any(indicator in name.lower() for name in methods for indicator in tool_indicators)


def extract_class_methods(cls) -> List[Dict[str, str]]:
    """Extract public methods from a tool class."""
    methods = []
    for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
        if not name.startswith('_'):
            methods.append({
                'name': name,
                'type': 'method',
                'signature': str(inspect.signature(method)),
                'docstring': inspect.getdoc(method) or ''
            })
    return methods


def is_tool_function(func) -> bool:
    """Check if a function looks like a tool function."""
    func_name = func.__name__.lower()
    tool_indicators = ['process', 'convert', 'analyze', 'enhance', 'detect', 'scan']
    return any(indicator in func_name for indicator in tool_indicators)


def extract_tool_description(content: str) -> str:
    """Extract tool description from docstring."""
    lines = content.split('\n')
    description = []
    
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                break
            in_docstring = True
            continue
        
        if in_docstring:
            if stripped.endswith('"""') or stripped.endswith("'''"):
                break
            if stripped:
                description.append(stripped)
    
    return ' '.join(description) if description else "No description available"


def display_tool_discovery(tools: Dict[str, List[Dict[str, Any]]], console: Console):
    """Display discovered tools in a formatted table."""
    table = Table(title="Discovered MCP Tools")
    table.add_column("Category", style="bold cyan")
    table.add_column("Tool", style="bold green")
    table.add_column("Type", style="dim")
    table.add_column("Functions", style="white")
    table.add_column("Description", style="yellow")
    
    for category, tool_list in tools.items():
        for tool in tool_list:
            func_count = len(tool['functions'])
            func_types = {f['type'] for f in tool['functions']}
            type_str = ', '.join(func_types)
            
            table.add_row(
                category,
                tool['name'],
                type_str,
                str(func_count),
                tool['description'][:100] + "..." if len(tool['description']) > 100 else tool['description']
            )
    
    console.print(table)


def register_tools(tools: Dict[str, List[Dict[str, Any]]], register_all: bool) -> Dict[str, Any]:
    """Register discovered tools in the MCP system."""
    registration_results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    for category, tool_list in tools.items():
        for tool in tool_list:
            tool_key = f"{category}.{tool['name']}"
            
            try:
                # Check if tool should be registered
                if not register_all and not should_register_tool(tool):
                    registration_results['skipped'].append(tool_key)
                    continue
                
                # Register the tool
                if register_tool_in_system(tool):
                    registration_results['successful'].append({
                        'key': tool_key,
                        'tool': tool
                    })
                else:
                    registration_results['failed'].append({
                        'key': tool_key,
                        'error': 'Registration function failed'
                    })
                    
            except Exception as e:
                registration_results['failed'].append({
                    'key': tool_key,
                    'error': str(e)
                })
    
    return registration_results


def should_register_tool(tool: Dict[str, Any]) -> bool:
    """Determine if a tool should be registered."""
    # Check for essential tool indicators
    essential_indicators = ['file', 'convert', 'process', 'analyze']
    tool_name = tool['name'].lower()
    
    return any(indicator in tool_name for indicator in essential_indicators)


def register_tool_in_system(tool: Dict[str, Any]) -> bool:
    """Register a single tool in the MCP system."""
    try:
        # This would integrate with the actual MCP server registration
        # For now, we'll create registration metadata
        
        tool_registry_path = Path(__file__).parent.parent / "ai_infrastructure" / "mcp" / "tool_registry.json"
        
        # Load existing registry
        if tool_registry_path.exists():
            with open(tool_registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        # Add tool to registry
        tool_key = f"{tool['category']}.{tool['name']}"
        registry[tool_key] = {
            'module': tool['module'],
            'category': tool['category'],
            'class_name': tool['class_name'],
            'functions': tool['functions'],
            'description': tool['description'],
            'registered_at': str(Path(tool['file_path']).stat().st_mtime)
        }
        
        # Save registry
        with open(tool_registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Registration failed for {tool['name']}: {e}")
        return False


def create_tool_registry(registration_results: Dict[str, Any]) -> bool:
    """Create a comprehensive tool registry file."""
    try:
        registry_path = Path(__file__).parent.parent / "config" / "tool_registry.yaml"
        
        registry = {
            'mcp_tools': {
                'registration_summary': {
                    'successful': len(registration_results['successful']),
                    'failed': len(registration_results['failed']),
                    'skipped': len(registration_results['skipped'])
                },
                'tools': {}
            }
        }
        
        # Add successful tools
        for result in registration_results['successful']:
            tool = result['tool']
            tool_key = result['key']
            
            registry['mcp_tools']['tools'][tool_key] = {
                'enabled': True,
                'category': tool['category'],
                'module': tool['module'],
                'functions': [f['name'] for f in tool['functions']],
                'description': tool['description']
            }
        
        # Save registry
        with open(registry_path, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False)
        
        return True
        
    except Exception as e:
        print(f"Failed to create tool registry: {e}")
        return False


def test_registered_tools(registration_results: Dict[str, Any]) -> Dict[str, Any]:
    """Test the registered tools."""
    test_results = {
        'passed': [],
        'failed': [],
        'errors': []
    }
    
    for result in registration_results['successful']:
        tool_key = result['key']
        tool = result['tool']
        
        try:
            # Basic import test
            spec = importlib.util.spec_from_file_location(tool['module'], tool['file_path'])
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # If there's a class, try to instantiate it
                if tool['class_name']:
                    tool_class = getattr(module, tool['class_name'])
                    instance = tool_class()
                    test_results['passed'].append(tool_key)
                else:
                    # Test functions
                    has_functions = any(
                        hasattr(module, func['name']) 
                        for func in tool['functions'] 
                        if func['type'] == 'function'
                    )
                    if has_functions:
                        test_results['passed'].append(tool_key)
                    else:
                        test_results['failed'].append({
                            'tool': tool_key,
                            'error': 'No functions found'
                        })
            else:
                test_results['failed'].append({
                    'tool': tool_key,
                    'error': 'Could not import module'
                })
                
        except Exception as e:
            test_results['errors'].append({
                'tool': tool_key,
                'error': str(e)
            })
    
    return test_results


def display_registration_results(results: Dict[str, Any], console: Console):
    """Display tool registration results."""
    table = Table(title="Tool Registration Results")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Details", style="dim")
    
    table.add_row(
        "✅ Successful",
        str(len(results['successful'])),
        ", ".join([r['key'] for r in results['successful'][:3]]) + 
        ("..." if len(results['successful']) > 3 else "")
    )
    
    table.add_row(
        "❌ Failed", 
        str(len(results['failed'])),
        ", ".join([r['key'] for r in results['failed'][:3]]) +
        ("..." if len(results['failed']) > 3 else "")
    )
    
    table.add_row(
        "⏭️ Skipped",
        str(len(results['skipped'])),
        ", ".join(results['skipped'][:3]) +
        ("..." if len(results['skipped']) > 3 else "")
    )
    
    console.print(table)


def display_test_results(results: Dict[str, Any], console: Console):
    """Display tool test results."""
    table = Table(title="Tool Test Results")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Details", style="dim")
    
    table.add_row(
        "✅ Passed",
        str(len(results['passed'])),
        ", ".join(results['passed'][:3]) +
        ("..." if len(results['passed']) > 3 else "")
    )
    
    table.add_row(
        "❌ Failed",
        str(len(results['failed'])),
        ", ".join([r['tool'] for r in results['failed'][:3]]) +
        ("..." if len(results['failed']) > 3 else "")
    )
    
    if results['errors']:
        table.add_row(
            "💥 Errors",
            str(len(results['errors'])),
            ", ".join([r['tool'] for r in results['errors'][:3]]) +
            ("..." if len(results['errors']) > 3 else "")
        )
    
    console.print(table)


def main():
    """Main function for standalone execution."""
    console = Console()
    
    parser = argparse.ArgumentParser(description="Register MCP tools for AI File System")
    parser.add_argument("--categories", nargs="+", 
                       help="Specific tool categories to register")
    parser.add_argument("--all", action="store_true",
                       help="Register all discovered tools")
    parser.add_argument("--no-discover", action="store_true",
                       help="Skip tool discovery")
    parser.add_argument("--list", action="store_true",
                       help="List available tools without registering")
    
    args = parser.parse_args()
    
    if args.list:
        tools = discover_tools()
        display_tool_discovery(tools, console)
        return
    
    success = register_mcp_tools(
        discover=not args.no_discover,
        tool_categories=args.categories,
        register_all=args.all
    )
    
    if success:
        console.print("🎉 MCP tool registration completed successfully!")
        sys.exit(0)
    else:
        console.print("💥 MCP tool registration failed!")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()