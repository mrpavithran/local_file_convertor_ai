#!/usr/bin/env python3
"""
Health check and system verification script for AI File System.
"""

import os
import sys
import importlib
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax


def run_health_check(verbose: bool = False, fix_issues: bool = False) -> bool:
    """
    Run comprehensive health check for AI File System.
    
    Args:
        verbose: Show detailed information
        fix_issues: Attempt to fix detected issues
        
    Returns:
        True if system is healthy
    """
    console = Console()
    
    console.print(Panel.fit(
        "🏥 AI File System Health Check",
        style="bold blue"
    ))
    
    all_checks_passed = True
    check_results = {}
    
    # Run all health checks
    checks = [
        ("System Requirements", check_system_requirements),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_configuration_files),
        ("Python Dependencies", check_python_dependencies),
        ("AI Infrastructure", check_ai_infrastructure),
        ("File Operations", check_file_operations),
        ("CLI Functionality", check_cli_functionality),
        ("RAG System", check_rag_system),
        ("MCP Tools", check_mcp_tools)
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Running health checks...", total=len(checks))
        
        for check_name, check_function in checks:
            progress.update(task, description=f"Checking {check_name}...")
            
            try:
                result = check_function(verbose)
                check_results[check_name] = result
                
                if not result['healthy']:
                    all_checks_passed = False
                    
                    if fix_issues:
                        progress.update(task, description=f"Fixing {check_name}...")
                        fix_result = attempt_fix(check_name, result, console)
                        if fix_result:
                            check_results[check_name] = fix_result
                            if fix_result['healthy']:
                                all_checks_passed = all_checks_passed and True
                
            except Exception as e:
                check_results[check_name] = {
                    'healthy': False,
                    'issues': [f"Check failed: {e}"],
                    'details': {}
                }
                all_checks_passed = False
            
            progress.update(task, advance=1)
    
    # Display results
    display_health_results(check_results, console, verbose)
    
    # Generate report
    generate_health_report(check_results)
    
    if all_checks_passed:
        console.print("✅ All health checks passed! System is healthy. 🎉", style="bold green")
    else:
        console.print("❌ Some health checks failed. See details above.", style="bold red")
    
    return all_checks_passed


def check_system_requirements(verbose: bool = False) -> Dict[str, any]:
    """Check system requirements and environment."""
    issues = []
    details = {}
    
    # Python version check
    python_version = platform.python_version()
    details['python_version'] = python_version
    
    if sys.version_info < (3, 8):
        issues.append(f"Python version {python_version} is below minimum required 3.8")
    
    # Platform information
    details['platform'] = platform.platform()
    details['system'] = platform.system()
    
    # Available memory (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        details['total_memory_gb'] = round(memory.total / (1024**3), 1)
        details['available_memory_gb'] = round(memory.available / (1024**3), 1)
        
        if memory.available < 2 * 1024**3:  # 2GB
            issues.append("Low available memory (less than 2GB)")
    except ImportError:
        details['memory_info'] = "psutil not available"
        issues.append("psutil package not installed for memory monitoring")
    
    # Disk space
    try:
        disk = psutil.disk_usage('/')
        details['disk_free_gb'] = round(disk.free / (1024**3), 1)
        details['disk_total_gb'] = round(disk.total / (1024**3), 1)
        
        if disk.free < 5 * 1024**3:  # 5GB
            issues.append("Low disk space (less than 5GB free)")
    except:
        details['disk_info'] = "Disk info unavailable"
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def check_directory_structure(verbose: bool = False) -> Dict[str, any]:
    """Check if all required directories exist."""
    issues = []
    details = {}
    
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        base_dir / "ai_infrastructure",
        base_dir / "ai_infrastructure" / "ollama",
        base_dir / "ai_infrastructure" / "rag",
        base_dir / "ai_infrastructure" / "mcp",
        base_dir / "ai_infrastructure" / "mcp" / "tools",
        base_dir / "prompt_system",
        base_dir / "file_operations",
        base_dir / "core",
        base_dir / "core" / "utils",
        base_dir / "cli",
        base_dir / "cli" / "commands",
        base_dir / "data",
        base_dir / "data" / "processed_files",
        base_dir / "data" / "logs",
        base_dir / "data" / "chroma_db",
        base_dir / "config",
        base_dir / "scripts",
        base_dir / "tests",
        base_dir / "docs",
        base_dir / "examples"
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for directory in required_dirs:
        if directory.exists():
            existing_dirs.append(str(directory.relative_to(base_dir)))
        else:
            missing_dirs.append(str(directory.relative_to(base_dir)))
    
    details['existing_dirs'] = existing_dirs
    details['missing_dirs'] = missing_dirs
    
    if missing_dirs:
        issues.append(f"Missing directories: {', '.join(missing_dirs[:3])}" + 
                     ("..." if len(missing_dirs) > 3 else ""))
    
    return {
        'healthy': len(missing_dirs) == 0,
        'issues': issues,
        'details': details
    }


def check_configuration_files(verbose: bool = False) -> Dict[str, any]:
    """Check if configuration files exist and are valid."""
    issues = []
    details = {}
    
    config_dir = Path(__file__).parent.parent / "config"
    required_configs = [
        "system_config.yaml",
        "prompt_config.yaml", 
        "tool_config.yaml",
        "paths_config.yaml"
    ]
    
    missing_configs = []
    invalid_configs = []
    valid_configs = []
    
    for config_file in required_configs:
        config_path = config_dir / config_file
        
        if not config_path.exists():
            missing_configs.append(config_file)
            continue
        
        # Validate YAML syntax
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                valid_configs.append(config_file)
                details[config_file] = "Valid"
            else:
                invalid_configs.append(config_file)
                details[config_file] = "Empty or invalid"
                
        except yaml.YAMLError as e:
            invalid_configs.append(config_file)
            details[config_file] = f"YAML error: {e}"
        except Exception as e:
            invalid_configs.append(config_file)
            details[config_file] = f"Error: {e}"
    
    if missing_configs:
        issues.append(f"Missing config files: {', '.join(missing_configs)}")
    
    if invalid_configs:
        issues.append(f"Invalid config files: {', '.join(invalid_configs)}")
    
    return {
        'healthy': len(missing_configs) == 0 and len(invalid_configs) == 0,
        'issues': issues,
        'details': details
    }


def check_python_dependencies(verbose: bool = False) -> Dict[str, any]:
    """Check if required Python dependencies are installed."""
    issues = []
    details = {}
    
    # Core dependencies
    core_dependencies = [
        "rich",          # Console output
        "yaml",          # Configuration
        "pathlib",       # File paths
        "typing",        # Type hints
        "requests",      # HTTP requests
        "chromadb",      # Vector database
        "sentence_transformers",  # Embeddings
        "llama_index",   # RAG framework
        "pillow",        # Image processing
        "pypdf2",        # PDF processing
        "python-docx",   # Word documents
        "beautifulsoup4" # HTML parsing
    ]
    
    missing_deps = []
    available_deps = []
    
    for dep in core_dependencies:
        try:
            # Handle different import names
            if dep == "yaml":
                import yaml
            elif dep == "pathlib":
                from pathlib import Path
            elif dep == "typing":
                from typing import Dict, List
            else:
                importlib.import_module(dep)
            
            available_deps.append(dep)
            details[dep] = "Available"
            
        except ImportError:
            missing_deps.append(dep)
            details[dep] = "Missing"
    
    if missing_deps:
        issues.append(f"Missing dependencies: {', '.join(missing_deps[:5])}" +
                     ("..." if len(missing_deps) > 5 else ""))
    
    return {
        'healthy': len(missing_deps) == 0,
        'issues': issues,
        'details': details
    }


def check_ai_infrastructure(verbose: bool = False) -> Dict[str, any]:
    """Check AI infrastructure components."""
    issues = []
    details = {}
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            details['ollama'] = "Running"
            models = response.json().get('models', [])
            details['ollama_models'] = [m['name'] for m in models]
            
            if not models:
                issues.append("Ollama running but no models installed")
        else:
            details['ollama'] = f"HTTP {response.status_code}"
            issues.append("Ollama not responding properly")
    except:
        details['ollama'] = "Not running"
        issues.append("Ollama service not available")
    
    # Check ChromaDB
    try:
        db_path = Path(__file__).parent.parent / "data" / "chroma_db"
        if db_path.exists():
            import chromadb
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            details['chromadb'] = f"Available ({len(collections)} collections)"
        else:
            details['chromadb'] = "Database directory missing"
            issues.append("ChromaDB not initialized")
    except Exception as e:
        details['chromadb'] = f"Error: {e}"
        issues.append("ChromaDB not working")
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def check_file_operations(verbose: bool = False) -> Dict[str, any]:
    """Check file operations functionality."""
    issues = []
    details = {}
    
    try:
        # Test basic file operations
        test_dir = Path(__file__).parent.parent / "data" / "test"
        test_dir.mkdir(exist_ok=True)
        
        # Test file creation
        test_file = test_dir / "health_check_test.txt"
        test_file.write_text("Health check test file")
        details['file_creation'] = "Working"
        
        # Test file reading
        content = test_file.read_text()
        if content == "Health check test file":
            details['file_reading'] = "Working"
        else:
            issues.append("File reading test failed")
        
        # Cleanup
        test_file.unlink()
        test_dir.rmdir()
        
    except Exception as e:
        issues.append(f"File operations test failed: {e}")
        details['file_operations'] = f"Error: {e}"
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def check_cli_functionality(verbose: bool = False) -> Dict[str, any]:
    """Check CLI functionality."""
    issues = []
    details = {}
    
    try:
        cli_dir = Path(__file__).parent.parent / "cli"
        
        # Check if main CLI files exist
        cli_files = ["main.py", "command_parser.py", "interactive_mode.py"]
        missing_cli_files = []
        
        for cli_file in cli_files:
            if (cli_dir / cli_file).exists():
                details[cli_file] = "Exists"
            else:
                missing_cli_files.append(cli_file)
                details[cli_file] = "Missing"
        
        if missing_cli_files:
            issues.append(f"Missing CLI files: {', '.join(missing_cli_files)}")
        
        # Test basic CLI import
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from cli.main import main
            details['cli_import'] = "Working"
        except ImportError as e:
            issues.append(f"CLI import failed: {e}")
            details['cli_import'] = f"Error: {e}"
        
    except Exception as e:
        issues.append(f"CLI check failed: {e}")
        details['cli_check'] = f"Error: {e}"
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def check_rag_system(verbose: bool = False) -> Dict[str, any]:
    """Check RAG system functionality."""
    issues = []
    details = {}
    
    try:
        # Check if RAG components are available
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        # Test embedding model
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            test_embedding = model.encode(["test document"])
            details['embedding_model'] = "Working"
        except Exception as e:
            issues.append(f"Embedding model failed: {e}")
            details['embedding_model'] = f"Error: {e}"
        
        # Test ChromaDB
        try:
            db_path = Path(__file__).parent.parent / "data" / "chroma_db"
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            details['chromadb_collections'] = len(collections)
            details['chromadb'] = "Working"
        except Exception as e:
            issues.append(f"ChromaDB failed: {e}")
            details['chromadb'] = f"Error: {e}"
        
    except ImportError as e:
        issues.append(f"RAG dependencies missing: {e}")
        details['rag_system'] = "Dependencies missing"
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def check_mcp_tools(verbose: bool = False) -> Dict[str, any]:
    """Check MCP tools functionality."""
    issues = []
    details = {}
    
    try:
        tools_dir = Path(__file__).parent.parent / "ai_infrastructure" / "mcp" / "tools"
        
        if not tools_dir.exists():
            issues.append("MCP tools directory missing")
            details['mcp_tools'] = "Directory missing"
            return {
                'healthy': False,
                'issues': issues,
                'details': details
            }
        
        # Count available tool categories
        categories = [d.name for d in tools_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
        details['tool_categories'] = len(categories)
        details['categories'] = categories
        
        # Count tool files
        tool_files = []
        for category in categories:
            category_dir = tools_dir / category
            tools_in_category = [f.name for f in category_dir.glob("*.py") if not f.name.startswith('__')]
            tool_files.extend([f"{category}.{tool}" for tool in tools_in_category])
        
        details['total_tools'] = len(tool_files)
        details['tools'] = tool_files[:5]  # Show first 5
        
        if len(tool_files) == 0:
            issues.append("No MCP tools found")
        
        # Check tool registry
        registry_path = Path(__file__).parent.parent / "ai_infrastructure" / "mcp" / "tool_registry.json"
        if registry_path.exists():
            details['tool_registry'] = "Exists"
        else:
            issues.append("Tool registry not found")
            details['tool_registry'] = "Missing"
        
    except Exception as e:
        issues.append(f"MCP tools check failed: {e}")
        details['mcp_tools'] = f"Error: {e}"
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def attempt_fix(check_name: str, result: Dict[str, any], console: Console) -> Dict[str, any]:
    """Attempt to fix detected issues."""
    console.print(f"  Attempting to fix {check_name}...")
    
    if check_name == "Directory Structure":
        return fix_directory_structure(result)
    elif check_name == "Python Dependencies":
        return fix_python_dependencies(result)
    elif check_name == "AI Infrastructure":
        return fix_ai_infrastructure(result, console)
    else:
        console.print(f"  No auto-fix available for {check_name}")
        return result


def fix_directory_structure(result: Dict[str, any]) -> Dict[str, any]:
    """Fix missing directories."""
    fixed_issues = []
    
    base_dir = Path(__file__).parent.parent
    missing_dirs = result.get('details', {}).get('missing_dirs', [])
    
    for dir_path in missing_dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        fixed_issues.append(f"Created directory: {dir_path}")
    
    return {
        'healthy': len(missing_dirs) == 0,
        'issues': fixed_issues if fixed_issues else ["Directories created successfully"],
        'details': {'fixed_directories': fixed_issues}
    }


def fix_python_dependencies(result: Dict[str, any]) -> Dict[str, any]:
    """Attempt to install missing dependencies."""
    import subprocess
    
    missing_deps = []
    for dep, status in result.get('details', {}).items():
        if status == "Missing":
            missing_deps.append(dep)
    
    if not missing_deps:
        return result
    
    fixed_deps = []
    failed_deps = []
    
    for dep in missing_deps:
        try:
            # Map import names to package names
            package_map = {
                'yaml': 'pyyaml',
                'pathlib': 'pathlib',
                'typing': 'typing',
                'chromadb': 'chromadb',
                'sentence_transformers': 'sentence-transformers',
                'llama_index': 'llama-index',
                'pillow': 'pillow',
                'pypdf2': 'pypdf2',
                'python-docx': 'python-docx',
                'beautifulsoup4': 'beautifulsoup4'
            }
            
            package_name = package_map.get(dep, dep)
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                fixed_deps.append(dep)
            else:
                failed_deps.append(dep)
                
        except Exception as e:
            failed_deps.append(dep)
    
    issues = []
    if failed_deps:
        issues.append(f"Failed to install: {', '.join(failed_deps)}")
    
    return {
        'healthy': len(failed_deps) == 0,
        'issues': issues,
        'details': {'fixed_dependencies': fixed_deps, 'failed_dependencies': failed_deps}
    }


def fix_ai_infrastructure(result: Dict[str, any], console: Console) -> Dict[str, any]:
    """Attempt to fix AI infrastructure issues."""
    from scripts.deploy_ollama import deploy_ollama
    from scripts.setup_rag import setup_rag_system
    
    fixed_issues = []
    
    # Fix Ollama if needed
    if "Ollama" in str(result.get('details', {}).get('ollama', '')):
        console.print("  Deploying Ollama...")
        if deploy_ollama():
            fixed_issues.append("Ollama deployed successfully")
        else:
            fixed_issues.append("Failed to deploy Ollama")
    
    # Fix RAG system if needed
    if "ChromaDB" in str(result.get('details', {}).get('chromadb', '')):
        console.print("  Setting up RAG system...")
        if setup_rag_system():
            fixed_issues.append("RAG system setup completed")
        else:
            fixed_issues.append("Failed to setup RAG system")
    
    return {
        'healthy': len(fixed_issues) > 0 and "Failed" not in str(fixed_issues),
        'issues': fixed_issues,
        'details': {'fix_attempts': fixed_issues}
    }


def display_health_results(check_results: Dict[str, any], console: Console, verbose: bool):
    """Display health check results in a formatted table."""
    table = Table(title="Health Check Results")
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Issues", style="red")
    table.add_column("Details", style="dim")
    
    for check_name, result in check_results.items():
        status = "✅" if result['healthy'] else "❌"
        style = "green" if result['healthy'] else "red"
        
        issues = "\n".join(result['issues'][:2])  # Show first 2 issues
        if len(result['issues']) > 2:
            issues += f"\n... and {len(result['issues']) - 2} more"
        
        details = str(result['details'])[:100] + "..." if verbose and len(str(result['details'])) > 100 else ""
        
        table.add_row(
            check_name,
            status,
            issues,
            details
        )
    
    console.print(table)
    
    # Show detailed information if verbose
    if verbose:
        for check_name, result in check_results.items():
            if not result['healthy'] or verbose:
                console.print(f"\n[bold]{check_name} Details:[/bold]")
                console.print(Syntax(
                    json.dumps(result['details'], indent=2), 
                    "json", 
                    theme="monokai"
                ))


def generate_health_report(check_results: Dict[str, any]):
    """Generate a health check report file."""
    report_path = Path(__file__).parent.parent / "data" / "logs" / "health_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'system_info': {
            'python_version': platform.python_version(),
            'platform': platform.platform()
        },
        'checks': check_results,
        'summary': {
            'total_checks': len(check_results),
            'passed_checks': sum(1 for r in check_results.values() if r['healthy']),
            'failed_checks': sum(1 for r in check_results.values() if not r['healthy']),
            'all_passed': all(r['healthy'] for r in check_results.values())
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    """Main function for standalone execution."""
    console = Console()
    
    parser = argparse.ArgumentParser(description="Run AI File System health check")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed information")
    parser.add_argument("--fix", action="store_true",
                       help="Attempt to fix detected issues")
    parser.add_argument("--report", action="store_true",
                       help="Generate health report only")
    
    args = parser.parse_args()
    
    if args.report:
        # Run checks and generate report without display
        check_results = {}
        checks = [
            ("System Requirements", check_system_requirements),
            ("Directory Structure", check_directory_structure),
            ("Configuration Files", check_configuration_files),
            ("Python Dependencies", check_python_dependencies),
            ("AI Infrastructure", check_ai_infrastructure),
        ]
        
        for check_name, check_function in checks:
            check_results[check_name] = check_function(False)
        
        generate_health_report(check_results)
        console.print("📊 Health report generated")
        return
    
    healthy = run_health_check(
        verbose=args.verbose,
        fix_issues=args.fix
    )
    
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    import argparse
    main()