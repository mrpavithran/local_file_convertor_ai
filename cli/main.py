"""
Main CLI entry point - Fixed version
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    """Main entry point for the CLI"""
    try:
        from cli.command_parser import CommandParser
        
        # Create parser and process arguments
        parser = CommandParser()
        args = parser.parse_args()
        
        # Validate and execute
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
        print(f"❌ Import error: {e}")
        print("Please check that all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()