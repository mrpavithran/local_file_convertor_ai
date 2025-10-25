#!/usr/bin/env python3
"""
Main CLI entry point for AI File System operations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.command_parser import CommandParser
from cli.interactive_mode import InteractiveMode
from core.config_manager import ConfigManager


def main():
    """Main CLI entry point."""
    try:
        # Initialize configuration
        ConfigManager.initialize()
        
        parser = CommandParser.create_parser()
        args = parser.parse_args()
        
        if hasattr(args, 'func'):
            args.func(args)
        elif hasattr(args, 'interactive') and args.interactive:
            InteractiveMode().run()
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()