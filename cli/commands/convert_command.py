"""
Convert command implementation.
"""

import os
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from file_operations.file_manager import FileManager
from file_operations.type_detector import TypeDetector


class ConvertCommand:
    """Handle file conversion commands."""

    def __init__(self):
        self.console = Console()
        self.file_processor = FileManager()
        self.type_detector = TypeDetector()

    def execute(self, args):
        """Execute the convert command."""
        try:
            input_path = Path(args['input'])
            output_path = Path(args['output'])

            if not input_path.exists():
                self.console.print(f"[red]Error: Input path does not exist: {input_path}[/red]")
                return

            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)

            self.console.print(f"[bold]Converting from {args['source_lang']} to {args['target_lang']}[/bold]")
            self.console.print(f"Input: {input_path}")
            self.console.print(f"Output: {output_path}")

            # Find files to process
            if args.get('recursive'):
                files = list(input_path.rglob('*'))
            else:
                files = list(input_path.glob('*'))

            # Filter files
            files = [f for f in files if f.is_file()]

            if not files:
                self.console.print("[yellow]No files found to process.[/yellow]")
                return

            # Process files with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:

                task = progress.add_task("Converting files...", total=len(files))

                successful = 0
                failed = 0

                for file_path in files:
                    try:
                        progress.update(task, advance=1, description=f"Processing {file_path.name}")

                        # Determine file type and process
                        relative_path = file_path.relative_to(input_path)
                        output_file = output_path / relative_path

                        # Create output directory structure
                        output_file.parent.mkdir(parents=True, exist_ok=True)

                        # For now, just copy the file (replace with actual conversion)
                        with open(file_path, 'r', encoding='utf-8') as src:
                            content = src.read()

                        with open(output_file, 'w', encoding='utf-8') as dst:
                            dst.write(f"# Converted from {args['source_lang']} to {args['target_lang']}\n")
                            dst.write(content)

                        successful += 1

                    except Exception as e:
                        self.console.print(f"[red]Error processing {file_path}: {e}[/red]")
                        failed += 1

            # Display results
            self.display_results(successful, failed, output_path)

        except Exception as e:
            self.console.print(f"[red]Conversion failed: {e}[/red]")

    def display_results(self, successful, failed, output_path):
        """Display conversion results."""
        table = Table(title="Conversion Results", show_header=True, header_style="bold magenta")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")

        total = successful + failed
        success_pct = (successful / total * 100) if total > 0 else 0
        fail_pct = (failed / total * 100) if total > 0 else 0

        table.add_row("✅ Successful", str(successful), f"{success_pct:.1f}%")
        table.add_row("❌ Failed", str(failed), f"{fail_pct:.1f}%")
        table.add_row("📊 Total", str(total), "100%")

        self.console.print(table)
        self.console.print(f"[green]Output directory: {output_path}[/green]")
