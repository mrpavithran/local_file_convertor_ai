#!/usr/bin/env python3
"""
Health check and system verification script for AI File System.
FIXED VERSION - Complete and functional
"""

import os
import sys
import importlib
import platform
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    Console = SimpleConsole


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
    
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "🏥 AI File System Health Check",
            style="bold blue"
        ))
    else:
        console.print("🏥 AI File System Health Check")
        console.print("=" * 50)
    
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
    
    if RICH_AVAILABLE:
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
    else:
        # Simple progress without rich
        for i, (check_name, check_function) in enumerate(checks, 1):
            console.print(f"Checking {check_name}... ({i}/{len(checks)})")
            
            try:
                result = check_function(verbose)
                check_results[check_name] = result
                
                if not result['healthy']:
                    all_checks_passed = False
                    
                    if fix_issues:
                        console.print(f"  Fixing {check_name}...")
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
    
    # Display results
    display_health_results(check_results, console, verbose)
    
    # Generate report
    generate_health_report(check_results)
    
    if all_checks_passed:
        console.print("✅ All health checks passed! System is healthy. 🎉")
    else:
        console.print("❌ Some health checks failed. See details above.")
    
    return all_checks_passed


def check_system_requirements(verbose: bool = False) -> Dict[str, Any]:
    """Check system requirements and environment."""
    issues = []
    details = {}
    
    # Python version check
    python_version = platform.python_version()
    details['python_version'] = python_version
    
    if sys.version_info < (3, 8):
        issues.append(f"Python version {python_version} is below minimum required 3.8")
    else:
        details['python_version_status'] = "OK"
    
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
        else:
            details['memory_status'] = "Sufficient"
    except ImportError:
        details['memory_info'] = "psutil not available"
        issues.append("psutil package not installed for memory monitoring")
    
    # Disk space
    try:
        disk = psutil.disk_usage('.')
        details['disk_free_gb'] = round(disk.free / (1024**3), 1)
        details['disk_total_gb'] = round(disk.total / (1024**3), 1)
        
        if disk.free < 1 * 1024**3:  # 1GB
            issues.append("Low disk space (less than 1GB free)")
        else:
            details['disk_status'] = "Sufficient"
    except:
        details['disk_info'] = "Disk info unavailable"
        # Don't add this as an issue since it's not critical
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def check_directory_structure(verbose: bool = False) -> Dict[str, Any]:
    """Check if all required directories exist."""
    issues = []
    details = {}
    
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "ai_infrastructure",
        "ai_infrastructure/ollama",
        "ai_infrastructure/rag", 
        "ai_infrastructure/mcp",
        "ai_infrastructure/mcp/tools",
        "ai_infrastructure/mcp/tools/conversion_tools",
        "ai_infrastructure/mcp/tools/file_tools",
        "prompt_system",
        "file_operations",
        "core",
        "core/utils",
        "cli", 
        "cli/commands",
        "data",
        "data/processed_files",
        "data/processed_files/maintained_structure",
        "data/processed_files/flattened_structure",
        "data/logs",
        "data/chroma_db",
        "config",
        "scripts",
        "tests",
        "tests/test_data",
        "docs",
        "examples"
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for directory in required_dirs:
        dir_path = base_dir / directory
        if dir_path.exists():
            existing_dirs.append(directory)
        else:
            missing_dirs.append(directory)
    
    details['existing_dirs'] = existing_dirs
    details['missing_dirs'] = missing_dirs
    
    if missing_dirs:
        issues.append(f"Missing {len(missing_dirs)} directories")
        if verbose:
            details['missing_dirs_full'] = missing_dirs
    
    return {
        'healthy': len(missing_dirs) == 0,
        'issues': issues,
        'details': details
    }


def check_configuration_files(verbose: bool = False) -> Dict[str, Any]:
    """Check if configuration files exist and are valid."""
    issues = []
    details = {}
    
    config_dir = Path(__file__).parent.parent / "config"
    required_configs = [
        "system_config.yaml",
        "tool_config.yaml"
    ]
    
    optional_configs = [
        "prompt_config.yaml", 
        "paths_config.yaml"
    ]
    
    missing_configs = []
    invalid_configs = []
    valid_configs = []
    
    # Check required configs
    for config_file in required_configs:
        config_path = config_dir / config_file
        
        if not config_path.exists():
            missing_configs.append(config_file)
            details[config_file] = "Missing"
            continue
        
        # Validate YAML syntax if yaml is available
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                valid_configs.append(config_file)
                details[config_file] = "Valid"
            else:
                invalid_configs.append(config_file)
                details[config_file] = "Empty or invalid"
                
        except ImportError:
            # YAML not available, just check file existence
            valid_configs.append(config_file)
            details[config_file] = "Exists (YAML not available)"
        except yaml.YAMLError as e:
            invalid_configs.append(config_file)
            details[config_file] = f"YAML error: {e}"
        except Exception as e:
            invalid_configs.append(config_file)
            details[config_file] = f"Error: {e}"
    
    # Check optional configs
    for config_file in optional_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            details[config_file] = "Exists (optional)"
        else:
            details[config_file] = "Missing (optional)"
    
    if missing_configs:
        issues.append(f"Missing required config files: {', '.join(missing_configs)}")
    
    if invalid_configs:
        issues.append(f"Invalid config files: {', '.join(invalid_configs)}")
    
    return {
        'healthy': len(missing_configs) == 0 and len(invalid_configs) == 0,
        'issues': issues,
        'details': details
    }


def check_python_dependencies(verbose: bool = False) -> Dict[str, Any]:
    """Check if required Python dependencies are installed."""
    issues = []
    details = {}
    
    # Core dependencies for basic functionality
    core_dependencies = [
        "pandas",        # Data processing
        "openpyxl",      # Excel files
        "python_docx",   # Word documents (note: actual package is python-docx)
        "pypdf",         # PDF processing
        "reportlab",     # PDF generation
        "psutil",        # System monitoring
    ]
    
    # Optional dependencies
    optional_dependencies = [
        "pdfminer",      # Advanced PDF processing
        "pillow",        # Image processing
        "requests",      # HTTP requests
        "chromadb",      # Vector database
        "sentence_transformers",  # Embeddings
        "fastapi",       # Web API
        "uvicorn",       # ASGI server
        "rich",          # Console formatting
    ]
    
    missing_core_deps = []
    missing_optional_deps = []
    available_deps = []
    
    # Check core dependencies
    for dep in core_dependencies:
        try:
            # Handle different import names
            if dep == "python_docx":
                import docx
            elif dep == "pypdf":
                import pypdf
            else:
                importlib.import_module(dep)
            
            available_deps.append(dep)
            details[dep] = "Available"
            
        except ImportError:
            missing_core_deps.append(dep)
            details[dep] = "Missing"
    
    # Check optional dependencies
    for dep in optional_dependencies:
        try:
            if dep == "sentence_transformers":
                import sentence_transformers
            else:
                importlib.import_module(dep)
            details[dep] = "Available (optional)"
        except ImportError:
            missing_optional_deps.append(dep)
            details[dep] = "Missing (optional)"
    
    if missing_core_deps:
        issues.append(f"Missing core dependencies: {', '.join(missing_core_deps)}")
    
    if verbose and missing_optional_deps:
        details['missing_optional'] = missing_optional_deps
    
    return {
        'healthy': len(missing_core_deps) == 0,
        'issues': issues,
        'details': details
    }


def check_ai_infrastructure(verbose: bool = False) -> Dict[str, Any]:
    """Check AI infrastructure components."""
    issues = []
    details = {}
    
    # Check Ollama (optional)
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
            # Don't add as issue since Ollama is optional
    except:
        details['ollama'] = "Not running (optional)"
        # Not an issue since Ollama is optional
    
    # Check ChromaDB (optional)
    try:
        import chromadb
        db_path = Path(__file__).parent.parent / "data" / "chroma_db"
        if db_path.exists():
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            details['chromadb'] = f"Available ({len(collections)} collections)"
        else:
            details['chromadb'] = "Database directory missing (optional)"
            # Not an issue since ChromaDB is optional
    except ImportError:
        details['chromadb'] = "ChromaDB not installed (optional)"
    except Exception as e:
        details['chromadb'] = f"Error: {e} (optional)"
    
    return {
        'healthy': True,  # AI infrastructure is optional
        'issues': issues,
        'details': details
    }


def check_file_operations(verbose: bool = False) -> Dict[str, Any]:
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
        
        details['file_operations'] = "All tests passed"
        
    except Exception as e:
        issues.append(f"File operations test failed: {e}")
        details['file_operations'] = f"Error: {e}"
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def check_cli_functionality(verbose: bool = False) -> Dict[str, Any]:
    """Check CLI functionality."""
    issues = []
    details = {}
    
    try:
        cli_dir = Path(__file__).parent.parent / "cli"
        
        # Check if main CLI files exist
        cli_files = ["main.py", "interactive_mode.py"]
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


def check_rag_system(verbose: bool = False) -> Dict[str, Any]:
    """Check RAG system functionality."""
    issues = []
    details = {}
    
    # RAG is optional, so don't fail if not available
    try:
        # Check if RAG components are available
        import chromadb
        details['chromadb'] = "Available"
        
        # Test embedding model
        try:
            from sentence_transformers import SentenceTransformer
            # Don't actually load the model to save time
            details['sentence_transformers'] = "Available"
        except ImportError:
            details['sentence_transformers'] = "Not available (optional)"
        
        details['rag_system'] = "Components available"
        
    except ImportError as e:
        details['rag_system'] = "Optional components not installed"
        # Not an issue since RAG is optional
    
    return {
        'healthy': True,  # RAG is optional
        'issues': issues,
        'details': details
    }


def check_mcp_tools(verbose: bool = False) -> Dict[str, Any]:
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
        
        # Count tool files in conversion_tools (most important)
        conversion_dir = tools_dir / "conversion_tools"
        if conversion_dir.exists():
            conversion_tools = [f.name for f in conversion_dir.glob("*.py") if not f.name.startswith('__')]
            details['conversion_tools'] = conversion_tools
        else:
            issues.append("Conversion tools directory missing")
        
        # Check if we can import conversion tools
        try:
            from ai_infrastructure.mcp.tools.conversion_tools.convert_csv_to_xlsx import Tool as CSVTool
            details['csv_tool'] = "Importable"
        except ImportError as e:
            issues.append(f"CSV tool import failed: {e}")
            details['csv_tool'] = f"Error: {e}"
        
        try:
            from ai_infrastructure.mcp.tools.conversion_tools.convert_docx_to_pdf import Tool as DOCXTool
            details['docx_tool'] = "Importable"
        except ImportError as e:
            issues.append(f"DOCX tool import failed: {e}")
            details['docx_tool'] = f"Error: {e}"
        
        try:
            from ai_infrastructure.mcp.tools.conversion_tools.convert_pdf_to_docx import Tool as PDFTool
            details['pdf_tool'] = "Importable"
        except ImportError as e:
            issues.append(f"PDF tool import failed: {e}")
            details['pdf_tool'] = f"Error: {e}"
        
    except Exception as e:
        issues.append(f"MCP tools check failed: {e}")
        details['mcp_tools'] = f"Error: {e}"
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'details': details
    }


def attempt_fix(check_name: str, result: Dict[str, Any], console: Console) -> Dict[str, Any]:
    """Attempt to fix detected issues."""
    console.print(f"  Attempting to fix {check_name}...")
    
    if check_name == "Directory Structure":
        return fix_directory_structure(result)
    elif check_name == "Python Dependencies":
        return fix_python_dependencies(result, console)
    elif check_name == "Configuration Files":
        return fix_configuration_files(result, console)
    else:
        console.print(f"  No auto-fix available for {check_name}")
        return result


def fix_directory_structure(result: Dict[str, Any]) -> Dict[str, Any]:
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
        'issues': [],
        'details': {'fixed_directories': fixed_issues}
    }


def fix_python_dependencies(result: Dict[str, Any], console: Console) -> Dict[str, Any]:
    """Attempt to install missing dependencies."""
    missing_deps = []
    for dep, status in result.get('details', {}).items():
        if status == "Missing" and dep in ["pandas", "openpyxl", "pypdf", "reportlab", "psutil"]:
            missing_deps.append(dep)
    
    if not missing_deps:
        return result
    
    fixed_deps = []
    failed_deps = []
    
    # Map import names to package names
    package_map = {
        'pandas': 'pandas',
        'openpyxl': 'openpyxl', 
        'python_docx': 'python-docx',
        'pypdf': 'pypdf',
        'reportlab': 'reportlab',
        'psutil': 'psutil'
    }
    
    for dep in missing_deps:
        package_name = package_map.get(dep, dep)
        console.print(f"    Installing {package_name}...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                fixed_deps.append(dep)
                console.print(f"    ✅ Installed {package_name}")
            else:
                failed_deps.append(dep)
                console.print(f"    ❌ Failed to install {package_name}")
                
        except Exception as e:
            failed_deps.append(dep)
            console.print(f"    ❌ Error installing {package_name}: {e}")
    
    # Re-check the dependencies
    new_result = check_python_dependencies()
    new_result['details']['fix_attempt'] = {
        'fixed': fixed_deps,
        'failed': failed_deps
    }
    
    return new_result


def fix_configuration_files(result: Dict[str, Any], console: Console) -> Dict[str, Any]:
    """Create missing configuration files."""
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create basic system config
    system_config = config_dir / "system_config.yaml"
    if not system_config.exists():
        system_config.write_text("""system:
  name: "AI File Converter"
  version: "1.0.0"
  debug: false
  log_level: "INFO"

file_operations:
  default_input_dir: "./input"
  default_output_dir: "./output"
  backup_files: true
  max_file_size_mb: 100

conversion:
  preserve_structure: true
  overwrite_existing: false
  create_backups: true

paths:
  data_dir: "./data"
  logs_dir: "./data/logs"
  temp_dir: "./temp"
""")
        console.print("    ✅ Created system_config.yaml")
    
    # Create basic tool config
    tool_config = config_dir / "tool_config.yaml"
    if not tool_config.exists():
        tool_config.write_text("""tools:
  conversion:
    csv_to_xlsx:
      enabled: true
      timeout: 30
    docx_to_pdf:
      enabled: true  
      timeout: 60
    pdf_to_docx:
      enabled: true
      timeout: 60
""")
        console.print("    ✅ Created tool_config.yaml")
    
    return check_configuration_files()


def display_health_results(check_results: Dict[str, Any], console: Console, verbose: bool):
    """Display health check results in a formatted table."""
    if RICH_AVAILABLE:
        table = Table(title="Health Check Results")
        table.add_column("Check", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Issues", style="red")
        table.add_column("Details", style="dim")
        
        for check_name, result in check_results.items():
            status = "✅" if result['healthy'] else "❌"
            
            issues = "\n".join(result['issues'][:2])  # Show first 2 issues
            if len(result['issues']) > 2:
                issues += f"\n... and {len(result['issues']) - 2} more"
            
            details = ""
            if verbose:
                details = str(result['details'])[:100] + "..." if len(str(result['details'])) > 100 else str(result['details'])
            
            table.add_row(
                check_name,
                status,
                issues,
                details
            )
        
        console.print(table)
    else:
        # Simple table without rich
        console.print("\nHealth Check Results:")
        console.print("-" * 80)
        console.print(f"{'Check':<25} {'Status':<8} {'Issues'}")
        console.print("-" * 80)
        
        for check_name, result in check_results.items():
            status = "PASS" if result['healthy'] else "FAIL"
            issues = "; ".join(result['issues'][:2])
            if len(result['issues']) > 2:
                issues += f" ... (+{len(result['issues'])-2})"
            
            console.print(f"{check_name:<25} {status:<8} {issues}")
        
        console.print("-" * 80)
    
    # Show detailed information if verbose
    if verbose:
        for check_name, result in check_results.items():
            if not result['healthy']:
                console.print(f"\n{check_name} Details:")
                console.print(f"  Issues: {result['issues']}")
                if result['details']:
                    console.print(f"  Details: {result['details']}")


def generate_health_report(check_results: Dict[str, Any]):
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
    main()