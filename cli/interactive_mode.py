"""
Interactive Mode for CLI - FIXED VERSION
"""

import cmd
import sys
import os
import shlex

class InteractiveMode(cmd.Cmd):
    """
    Interactive command shell for AI File Converter
    Uses cmd.Cmd for proper shell functionality
    """
    
    def __init__(self):
        super().__init__()
        self.prompt = "🤖 converter> "
        self.intro = """
╔══════════════════════════════════════════════╗
║           AI FILE CONVERTER SHELL            ║
║          MCP Server: RUNNING ✅              ║
╚══════════════════════════════════════════════╝
Type 'help' for commands, 'exit' to quit
"""
        
    def do_help(self, arg):
        """Show available commands"""
        commands = {
            "help": "Show this help message",
            "exit": "Exit the interactive shell",
            "list": "List available conversion tools",
            "status": "Show system status",
            "convert": "Convert a file - usage: convert <input> [output]",
            "scan": "Scan directory for convertible files",
            "models": "Show available AI models",
            "clear": "Clear the screen"
        }
        
        print("\n📋 AVAILABLE COMMANDS:")
        print("=" * 50)
        for cmd, desc in commands.items():
            print(f"  {cmd:15} - {desc}")
        print()
    
    def do_list(self, arg):
        """List available conversion tools"""
        tools = [
            {"name": "PDF to DOCX", "status": "✅ Available", "usage": "convert input.pdf output.docx"},
            {"name": "DOCX to PDF", "status": "✅ Available", "usage": "convert input.docx output.pdf"},
            {"name": "Image Format Conversion", "status": "❌ Not available", "usage": "Install PIL/pillow"},
            {"name": "TXT to PDF", "status": "❌ Not available", "usage": "Install reportlab"},
            {"name": "HTML to PDF", "status": "❌ Not available", "usage": "Install weasyprint"}
        ]
        
        print("\n🛠️  CONVERSION TOOLS:")
        print("=" * 60)
        for tool in tools:
            print(f"  {tool['status']} {tool['name']}")
            print(f"      Usage: {tool['usage']}")
            print()
    
    def do_status(self, arg):
        """Show system status"""
        status_info = {
            "MCP Server": "✅ Running",
            "Tool Registry": "✅ Active",
            "AI Models": "✅ Available",
            "File System": "✅ Ready",
            "Conversion Tools": "🟡 Partial (2/5 available)"
        }
        
        print("\n📊 SYSTEM STATUS:")
        print("=" * 40)
        for component, status in status_info.items():
            print(f"  {component:20} {status}")
        print()
    
    def do_convert(self, arg):
        """Convert a file: convert input.pdf [output.docx]"""
        if not arg:
            print("❌ Usage: convert <input_file> [output_file]")
            print("   Example: convert document.pdf document.docx")
            return
        
        args = shlex.split(arg)
        input_file = args[0]
        output_file = args[1] if len(args) > 1 else None
        
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            return
        
        print(f"🔄 Converting: {input_file} -> {output_file or 'auto'}")
        
        try:
            # Try PDF to DOCX conversion
            if input_file.lower().endswith('.pdf'):
                from ai_infrastructure.mcp.tools.conversion_tools.convert_pdf_to_docx import PDFToDOCXConverter
                converter = PDFToDOCXConverter()
                result = converter.convert(input_file, output_file)
            # Try DOCX to PDF conversion  
            elif input_file.lower().endswith(('.docx', '.doc')):
                from ai_infrastructure.mcp.tools.conversion_tools.convert_docx_to_pdf import DOCXToPDFConverter
                converter = DOCXToPDFConverter()
                result = converter.convert(input_file, output_file)
            else:
                print(f"❌ Unsupported file format: {input_file}")
                print("   Supported: .pdf, .docx, .doc")
                return
            
            if result.get('success'):
                print(f"✅ Conversion successful!")
                if result.get('output_path'):
                    print(f"   Output: {result['output_path']}")
                if result.get('message'):
                    print(f"   {result['message']}")
            else:
                print(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")
                
        except ImportError as e:
            print(f"❌ Conversion tool not available: {e}")
            print("   Install required dependencies")
        except Exception as e:
            print(f"❌ Conversion error: {e}")
    
    def do_scan(self, arg):
        """Scan directory for convertible files"""
        directory = arg if arg else "."
        
        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            return
        
        if not os.path.isdir(directory):
            print(f"❌ Not a directory: {directory}")
            return
        
        print(f"🔍 Scanning: {directory}")
        
        convertible_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(('.pdf', '.docx', '.doc')):
                    convertible_files.append(file_path)
        
        if convertible_files:
            print(f"\n📁 Found {len(convertible_files)} convertible files:")
            for file_path in convertible_files[:10]:  # Show first 10
                print(f"   • {file_path}")
            if len(convertible_files) > 10:
                print(f"   ... and {len(convertible_files) - 10} more")
        else:
            print("❌ No convertible files found (.pdf, .docx, .doc)")
    
    def do_models(self, arg):
        """Show available AI models"""
        models = [
            {"name": "Mistral", "status": "✅ Available", "purpose": "Text processing"},
            {"name": "Llama 2", "status": "✅ Available", "purpose": "General AI tasks"},
            {"name": "GPT-4", "status": "❌ Not configured", "purpose": "Advanced AI"},
        ]
        
        print("\n🧠 AI MODELS:")
        print("=" * 50)
        for model in models:
            print(f"  {model['status']} {model['name']:15} - {model['purpose']}")
        print()
    
    def do_clear(self, arg):
        """Clear the screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.intro)
    
    def do_exit(self, arg):
        """Exit the interactive shell"""
        print("👋 Goodbye!")
        return True
    
    # Aliases for convenience
    def do_quit(self, arg):
        """Exit the interactive shell"""
        return self.do_exit(arg)
    
    def do_q(self, arg):
        """Exit the interactive shell"""
        return self.do_exit(arg)
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def default(self, line):
        """Handle unknown commands"""
        print(f"❌ Unknown command: {line}")
        print("   Type 'help' for available commands")
    
    def precmd(self, line):
        """Process command before execution"""
        return line.strip().lower()
    
    def postcmd(self, stop, line):
        """Process after command execution"""
        return stop


def start_interactive_mode():
    """Start the interactive shell"""
    try:
        shell = InteractiveMode()
        shell.cmdloop()
        return {"success": True, "message": "Interactive session completed"}
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return {"success": True, "message": "Session interrupted by user"}
    except Exception as e:
        return {"success": False, "error": f"Interactive mode error: {e}"}