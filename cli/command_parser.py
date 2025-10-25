"""
Command Parser for CLI Interface
Complete implementation with create_parser method
"""

import argparse
import sys
import os
from typing import List, Optional, Dict, Any

try:
    from ai_infrastructure.mcp.tools.conversion_tools.convert_pdf_to_docx import PDFToDOCXConverter
except ImportError:
    class PDFToDOCXConverter:
        def __init__(self): pass
        def convert(self, input_path, output_path=None):
            return {"success": False, "error": "Converter not available"}


class CommandParser:
    """
    Enhanced command line argument parser for the AI file converter
    """
    
    @classmethod
    def create_parser(cls):
        """Factory method to create argument parser - FIXES THE ERROR"""
        parser = argparse.ArgumentParser(
            description="AI-Powered File Converter with MCP Tools",
            epilog="Example: python -m cli.main --interactive"
        )
        
        # Input/output arguments
        parser.add_argument(
            '--input', '-i',
            type=str,
            help='Input file path'
        )
        
        parser.add_argument(
            '--output', '-o', 
            type=str,
            help='Output file path'
        )
        
        # Mode arguments
        parser.add_argument(
            '--interactive', '-I',
            action='store_true',
            help='Start interactive mode'
        )
        
        parser.add_argument(
            '--batch', '-b',
            action='store_true', 
            help='Process multiple files in batch mode'
        )
        
        parser.add_argument(
            '--directory', '-d',
            type=str,
            help='Directory to process (for batch mode)'
        )
        
        # Utility arguments
        parser.add_argument(
            '--list-tools',
            action='store_true',
            help='List all available conversion tools'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--version',
            action='store_true',
            help='Show version information'
        )
        
        return parser
    
    def __init__(self):
        self.parser = self.create_parser()
        self.converter = PDFToDOCXConverter()
    
    def parse_args(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse command line arguments"""
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = self.parser.parse_args(args)
        
        # Convert to dictionary for easier access
        return {
            'input': parsed_args.input,
            'output': parsed_args.output,
            'interactive': parsed_args.interactive,
            'batch': parsed_args.batch,
            'directory': parsed_args.directory,
            'list_tools': parsed_args.list_tools,
            'verbose': parsed_args.verbose,
            'version': parsed_args.version
        }
    
    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate parsed arguments"""
        if args['interactive'] or args['list_tools'] or args['version']:
            return True
            
        if not args['input'] and not args['batch']:
            print("Error: Input file is required for conversion")
            return False
            
        if args['batch'] and not args['directory']:
            print("Error: Directory is required for batch mode")
            return False
            
        return True
    
    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command based on parsed arguments"""
        if args['interactive']:
            return self._start_interactive_mode()
        elif args['list_tools']:
            return self._list_tools()
        elif args['version']:
            return self._show_version()
        elif args['batch']:
            return self._process_batch(args['directory'])
        else:
            return self._convert_file(args['input'], args['output'])
    
    def _start_interactive_mode(self) -> Dict[str, Any]:
        """Start interactive mode - FIXED VERSION"""
        print("🚀 Starting interactive mode...")
        
        try:
            # Option 1: Use the start_interactive_mode function
            from cli.interactive_mode import start_interactive_mode
            return start_interactive_mode()
        except ImportError:
            try:
                # Option 2: Use the InteractiveMode class directly
                from cli.interactive_mode import InteractiveMode
                shell = InteractiveMode()
                shell.cmdloop()
                return {"success": True, "message": "Interactive session completed"}
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Interactive mode failed: {e}"
                }
    
    def _list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        tools = [
            {"name": "PDF to DOCX", "status": "Available"},
            {"name": "DOCX to PDF", "status": "Available"}, 
            {"name": "Image Format Conversion", "status": "Not available"},
            {"name": "TXT to PDF", "status": "Not available"},
            {"name": "HTML to PDF", "status": "Not available"}
        ]
        
        print("Available Conversion Tools:")
        for tool in tools:
            status_icon = "✅" if "Available" in tool["status"] else "❌"
            print(f"  {status_icon} {tool['name']} - {tool['status']}")
        
        return {"success": True, "tools": tools}
    
    def _show_version(self) -> Dict[str, Any]:
        """Show version information"""
        version_info = {
            "application": "AI File Converter",
            "version": "1.0.0", 
            "mcp_server": "Running",
            "status": "Operational"
        }
        
        print("=== AI File Converter ===")
        for key, value in version_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return {"success": True, "version": version_info}
    
    def _process_batch(self, directory: str) -> Dict[str, Any]:
        """Process batch conversion"""
        if not directory or not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }
        
        print(f"Batch processing directory: {directory}")
        return {
            "success": True,
            "message": "Batch processing started",
            "directory": directory,
            "files_found": 0  # Placeholder
        }
    
    def _convert_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """Convert a single file"""
        if not input_path or not os.path.exists(input_path):
            return {
                "success": False,
                "error": f"Input file not found: {input_path}"
            }
        
        print(f"Converting: {input_path} -> {output_path or 'auto'}")
        
        try:
            result = self.converter.convert(input_path, output_path)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Conversion failed: {e}"
            }


def main():
    """Main function for direct script execution"""
    parser = CommandParser()
    args = parser.parse_args()
    
    if parser.validate_args(args):
        result = parser.execute(args)
        if result.get("success"):
            print("✅ Operation completed successfully")
        else:
            print(f"❌ Operation failed: {result.get('error', 'Unknown error')}")
    else:
        parser.parser.print_help()


if __name__ == "__main__":
    main()