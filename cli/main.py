"""
Main CLI entry point - Fixed version
Fully functional with proper error handling and system integration
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the CLI"""
    try:
        # Try to import the command parser first
        try:
            from cli.command_parser import CommandParser
            parser = CommandParser()
            args = parser.parse_args()
            
            if parser.validate_args(args):
                result = parser.execute(args)
                
                # Print result summary
                if result.get('success'):
                    print("🎉 Operation completed successfully!")
                    if result.get('message'):
                        print(f"   {result['message']}")
                else:
                    print("💥 Operation failed!")
                    if result.get('error'):
                        print(f"   Error: {result['error']}")
            else:
                # Show help if validation fails
                parser.parser.print_help()
                
        except ImportError as e:
            print(f"⚠️  Command parser not available: {e}")
            print("🔄 Using built-in argument parser...")
            _fallback_main()
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def _fallback_main():
    """Fallback main function when CommandParser is not available."""
    parser = argparse.ArgumentParser(
        description='AI File Converter - Convert files with AI assistance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.main --interactive          # Start interactive mode
  python -m cli.main --convert input.csv output.xlsx csv2xlsx
  python -m cli.main --scan ./documents     # Scan directory for files
  python -m cli.main --status               # Show system status
        """
    )
    
    # Main operation modes
    operation_group = parser.add_argument_group('Operation Modes')
    operation_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    operation_group.add_argument(
        '--convert',
        nargs=3,
        metavar=('INPUT', 'OUTPUT', 'TYPE'),
        help='Convert a file: --convert input.pdf output.docx pdf2docx'
    )
    operation_group.add_argument(
        '--scan',
        metavar='DIRECTORY',
        help='Scan directory for convertible files'
    )
    
    # System commands
    system_group = parser.add_argument_group('System Commands')
    system_group.add_argument(
        '--list-tools', '--tools',
        action='store_true',
        help='List available conversion tools'
    )
    system_group.add_argument(
        '--status',
        action='store_true', 
        help='Show system status'
    )
    system_group.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration'
    )
    
    args = parser.parse_args()
    
    # Execute based on arguments
    try:
        from core.system_orchestrator import get_system_orchestrator
        system = get_system_orchestrator()
        
        if args.interactive:
            _start_interactive_mode()
            
        elif args.convert:
            input_file, output_file, conversion_type = args.convert
            _convert_file(system, input_file, output_file, conversion_type)
            
        elif args.scan:
            _scan_directory(system, args.scan)
            
        elif args.list_tools:
            _list_tools(system)
            
        elif args.status:
            _show_status(system)
            
        elif args.config:
            _show_config()
            
        else:
            # No arguments provided
            parser.print_help()
            print("\n💡 Try '--interactive' for the full experience!")
            
    except ImportError as e:
        print(f"❌ System initialization failed: {e}")
        print("💡 Make sure all dependencies are installed and system is properly configured")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        sys.exit(1)


def _start_interactive_mode():
    """Start interactive mode."""
    try:
        from cli.interactive_mode import start_interactive_mode
        print("🚀 Starting AI File Converter Interactive Mode...")
        result = start_interactive_mode()
        if not result.get('success'):
            print(f"❌ Interactive mode error: {result.get('error')}")
    except ImportError as e:
        print(f"❌ Interactive mode not available: {e}")
        print("💡 Check that cli/interactive_mode.py exists and is functional")


def _convert_file(system, input_file: str, output_file: str, conversion_type: str):
    """Convert a single file."""
    print(f"🔄 Converting: {input_file} -> {output_file} ({conversion_type})")
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Validate conversion type
    supported_types = ['csv2xlsx', 'docx2pdf', 'pdf2docx']
    if conversion_type not in supported_types:
        print(f"❌ Unsupported conversion type: {conversion_type}")
        print(f"   Supported types: {', '.join(supported_types)}")
        return
    
    # Perform conversion
    result = system.convert_file(input_file, output_file, conversion_type)
    
    if result.get('success'):
        print(f"✅ Conversion successful!")
        print(f"   Output: {result.get('output_path')}")
        if result.get('message'):
            print(f"   {result['message']}")
    else:
        print(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")


def _scan_directory(system, directory: str):
    """Scan directory for convertible files."""
    print(f"🔍 Scanning: {directory}")
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return
    
    result = system.scan_directory(directory)
    
    if result.get('success'):
        files = result.get('convertible_files', [])
        total_files = result.get('total_files', 0)
        total_convertible = result.get('total_convertible', 0)
        
        print(f"📊 Found {total_convertible} convertible files out of {total_files} total files:")
        
        for i, file_info in enumerate(files[:10], 1):  # Show first 10
            file_path = file_info.get('path', 'Unknown')
            filename = os.path.basename(file_path)
            print(f"  {i:2d}. {filename}")
            
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
            
    else:
        print(f"❌ Scan failed: {result.get('error', 'Unknown error')}")


def _list_tools(system):
    """List available conversion tools."""
    print("🛠️ Available Conversion Tools:")
    print("=" * 50)
    
    tools = system.get_available_tools()
    
    for tool_name, tool_info in tools.items():
        status = "✅" if tool_info.get('available') else "❌"
        description = tool_info.get('description', 'No description')
        print(f"  {status} {tool_name:15} - {description}")


def _show_status(system):
    """Show system status."""
    print("📊 System Status:")
    print("=" * 40)
    
    status = system.get_system_status()
    
    print(f"  Status: {status.get('status', 'unknown')}")
    print(f"  Uptime: {status.get('uptime_seconds', 0):.1f} seconds")
    print(f"  Active Operations: {status.get('active_operations', 0)}")
    print(f"  Available Tools: {status.get('available_tools', 0)}/{status.get('total_tools', 0)}")
    
    metrics = status.get('metrics', {})
    print(f"  Total Operations: {metrics.get('total_operations', 0)}")
    print(f"  Successful: {metrics.get('successful_operations', 0)}")
    print(f"  Failed: {metrics.get('failed_operations', 0)}")
    
    # Memory usage
    memory = status.get('memory_usage', {})
    if memory.get('rss_mb', 0) > 0:
        print(f"  Memory Usage: {memory.get('rss_mb', 0):.1f} MB ({memory.get('percent', 0):.1f}%)")


def _show_config():
    """Show current configuration."""
    try:
        from core.config_manager import get_config_manager
        config = get_config_manager()
        config.show_config()
    except ImportError:
        print("❌ Config manager not available")
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")


if __name__ == "__main__":
    main()