"""
Batch command implementation.
"""

import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from cli.commands.convert_command import ConvertCommand
from cli.commands.enhance_command import EnhanceCommand


class BatchCommand:
    """Handle batch processing commands."""
    
    def __init__(self):
        self.console = Console()
    
    def execute(self, args):
        """Execute the batch command."""
        try:
            config_path = Path(args['config'])
            workers = args.get('workers', 4)
            
            if not config_path.exists():
                self.console.print(f"[red]Error: Config file does not exist: {config_path}[/red]")
                return
            
            # Load batch configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.console.print(f"[bold]Batch Processing[/bold]")
            self.console.print(f"Config: {config_path}")
            self.console.print(f"Workers: {workers}")
            
            # Process batch tasks
            tasks = config.get('tasks', [])
            
            if not tasks:
                self.console.print("[yellow]No tasks found in configuration.[/yellow]")
                return
            
            total_tasks = len(tasks)
            completed = 0
            failed = 0
            
            with Progress() as progress:
                task_progress = progress.add_task("Processing batch...", total=total_tasks)
                
                for task in tasks:
                    try:
                        task_type = task.get('type')
                        task_config = task.get('config', {})
                        
                        progress.update(task_progress, description=f"Processing {task_type} task")
                        
                        if task_type == 'convert':
                            self.process_convert_task(task_config)
                        elif task_type == 'enhance':
                            self.process_enhance_task(task_config)
                        else:
                            self.console.print(f"[yellow]Unknown task type: {task_type}[/yellow]")
                            continue
                        
                        completed += 1
                        
                    except Exception as e:
                        self.console.print(f"[red]Task failed: {e}[/red]")
                        failed += 1
                    
                    progress.update(task_progress, advance=1)
            
            # Display batch results
            self.display_batch_results(completed, failed, total_tasks)
            
        except Exception as e:
            self.console.print(f"[red]Batch processing failed: {e}[/red]")
    
    def process_convert_task(self, config):
        """Process a conversion task."""
        convert_cmd = ConvertCommand()
        convert_cmd.execute(config)
    
    def process_enhance_task(self, config):
        """Process an enhancement task."""
        enhance_cmd = EnhanceCommand()
        enhance_cmd.execute(config)
    
    def display_batch_results(self, completed, failed, total):
        """Display batch processing results."""
        table = Table(title="Batch Processing Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Tasks", str(total))
        table.add_row("✅ Completed", str(completed))
        table.add_row("❌ Failed", str(failed))
        table.add_row("🎯 Success Rate", f"{(completed/total*100):.1f}%" if total > 0 else "0%")
        
        self.console.print(table)