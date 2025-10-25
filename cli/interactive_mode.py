"""
Interactive mode for AI File System.
"""

import os
import sys
from pathlib import Path
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from cli.commands.convert_command import ConvertCommand
from cli.commands.enhance_command import EnhanceCommand
from cli.commands.batch_command import BatchCommand


class InteractiveMode:
    """Interactive command-line interface."""
    
    def __init__(self):
        self.console = Console()
        self.running = True
        
    def display_banner(self):
        """Display welcome banner."""
        banner = """
# 🚀 AI File System

**Convert, enhance, and transform your files with AI assistance**
        """
        self.console.print(Panel(Markdown(banner), style="bold blue"))
    
    def main_menu(self):
        """Display main menu and get user choice."""
        choices = [
            "📁 Convert Files",
            "✨ Enhance Files", 
            "⚡ Batch Process",
            "📊 View Logs",
            "⚙️  Settings",
            "❌ Exit"
        ]
        
        return questionary.select(
            "What would you like to do?",
            choices=choices
        ).ask()
    
    def run_convert_flow(self):
        """Interactive flow for file conversion."""
        self.console.print("\n[bold]📁 File Conversion[/bold]")
        
        input_dir = questionary.path("Input directory:").ask()
        output_dir = questionary.path("Output directory:").ask()
        source_lang = questionary.text("Source format:").ask()
        target_lang = questionary.text("Target format:").ask()
        recursive = questionary.confirm("Process recursively?").ask()
        
        if all([input_dir, output_dir, source_lang, target_lang]):
            command = ConvertCommand()
            command.execute({
                'input': input_dir,
                'output': output_dir,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'recursive': recursive
            })
        else:
            self.console.print("[red]Missing required information![/red]")
    
    def run_enhance_flow(self):
        """Interactive flow for file enhancement."""
        self.console.print("\n[bold]✨ File Enhancement[/bold]")
        
        file_path = questionary.path("File to enhance:").ask()
        language = questionary.text("File type:").ask()
        
        if file_path and language:
            command = EnhanceCommand()
            command.execute({
                'file': file_path,
                'language': language
            })
        else:
            self.console.print("[red]Missing required information![/red]")
    
    def run_batch_flow(self):
        """Interactive flow for batch processing."""
        self.console.print("\n[bold]⚡ Batch Processing[/bold]")
        
        config_file = questionary.path("Configuration file:").ask()
        workers = questionary.text("Number of workers (default: 4):", default="4").ask()
        
        if config_file:
            command = BatchCommand()
            command.execute({
                'config': config_file,
                'workers': int(workers)
            })
        else:
            self.console.print("[red]Configuration file is required![/red]")
    
    def run(self):
        """Run the interactive mode."""
        self.display_banner()
        
        while self.running:
            try:
                choice = self.main_menu()
                
                if choice == "📁 Convert Files":
                    self.run_convert_flow()
                elif choice == "✨ Enhance Files":
                    self.run_enhance_flow()
                elif choice == "⚡ Batch Process":
                    self.run_batch_flow()
                elif choice == "📊 View Logs":
                    self.console.print("[yellow]Log viewing not yet implemented[/yellow]")
                elif choice == "⚙️  Settings":
                    self.console.print("[yellow]Settings not yet implemented[/yellow]")
                elif choice == "❌ Exit":
                    self.console.print("[green]Goodbye! 👋[/green]")
                    self.running = False
                    
            except KeyboardInterrupt:
                self.console.print("\n[green]Goodbye! 👋[/green]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")