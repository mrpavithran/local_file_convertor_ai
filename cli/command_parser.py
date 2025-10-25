"""
Command line argument parser for AI File System.
"""

import argparse
from pathlib import Path


class CommandParser:
    """Handles command line argument parsing."""
    
    @staticmethod
    def create_parser():
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="AI File System - Intelligent file operations with AI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python -m cli.main convert --input ./project --output ./converted
  python -m cli.main enhance --file main.py --language python
  python -m cli.main batch --config batch_config.yaml
  python -m cli.main --interactive
            """
        )
        
        # Global arguments
        parser.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Launch interactive mode'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        # Subparsers for different commands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Convert command
        convert_parser = subparsers.add_parser('convert', help='Convert codebase files')
        convert_parser.add_argument('--input', '-i', required=True, help='Input directory')
        convert_parser.add_argument('--output', '-o', required=True, help='Output directory')
        convert_parser.add_argument('--source-lang', '-s', required=True, help='Source language')
        convert_parser.add_argument('--target-lang', '-t', required=True, help='Target language')
        convert_parser.add_argument('--recursive', '-r', action='store_true', help='Process recursively')
        
        # Enhance command
        enhance_parser = subparsers.add_parser('enhance', help='Enhance existing code')
        enhance_parser.add_argument('--file', '-f', required=True, help='File to enhance')
        enhance_parser.add_argument('--language', '-l', required=True, help='Programming language')
        enhance_parser.add_argument('--task', '-t', help='Specific enhancement task')
        
        # Batch command
        batch_parser = subparsers.add_parser('batch', help='Batch process multiple files')
        batch_parser.add_argument('--config', '-c', required=True, help='Batch configuration file')
        batch_parser.add_argument('--workers', '-w', type=int, default=4, help='Number of worker processes')
        
        return parser
    
    @staticmethod
    def validate_args(args):
        """Validate command line arguments."""
        if hasattr(args, 'input') and args.input:
            input_path = Path(args.input)
            if not input_path.exists():
                raise ValueError(f"Input path does not exist: {args.input}")
        
        if hasattr(args, 'output') and args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
        
        return True