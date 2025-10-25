"""
Enhance command implementation with AI-powered file enhancement.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.prompt import Confirm, Prompt
import tempfile
import shutil

from file_operations.file_manager import FileEnhancer


class EnhanceCommand:
    """Handle file enhancement commands with AI-powered improvements."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.console = Console()
        self.file_enhancer = FileEnhancer()
        self.config = config or {}
        
        # Enhancement types and their descriptions
        self.enhancement_types = {
            'grammar': 'Fix grammar and spelling errors',
            'style': 'Improve writing style and clarity',
            'refactor': 'Refactor code for better structure',
            'document': 'Add documentation and comments',
            'optimize': 'Optimize performance and efficiency',
            'security': 'Improve security practices',
            'accessibility': 'Enhance accessibility features',
            'summarize': 'Create summary of content',
            'translate': 'Translate to another language',
            'format': 'Improve formatting and layout'
        }
    
    def execute(self, args):
        """Execute the enhance command."""
        try:
            file_path = Path(args['file'])
            enhancement_type = args.get('enhancement_type', 'style')
            language = args.get('language', 'auto')
            output_path = args.get('output')
            preview = args.get('preview', True)
            backup = args.get('backup', True)
            
            if not file_path.exists():
                self.console.print(f"[red]Error: File does not exist: {file_path}[/red]")
                return False
            
            # Auto-detect language if not specified
            if language == 'auto':
                language = self._detect_language(file_path)
            
            # Validate enhancement type
            if enhancement_type not in self.enhancement_types:
                self.console.print(f"[yellow]Warning: Unknown enhancement type '{enhancement_type}'. Using 'style'.[/yellow]")
                enhancement_type = 'style'
            
            self.console.print(Panel.fit(
                f"[bold]AI File Enhancement[/bold]\n"
                f"File: {file_path}\n"
                f"Type: {enhancement_type} - {self.enhancement_types[enhancement_type]}\n"
                f"Language: {language}",
                title="🚀 Enhancement Started"
            ))
            
            # Read original file
            original_content = self._read_file(file_path)
            if original_content is None:
                return False
            
            # Display original content if preview mode
            if preview:
                self._display_content(original_content, language, "Original Content", "red")
            
            # Apply enhancement
            enhanced_content = self._apply_enhancement(
                original_content, enhancement_type, language, file_path
            )
            
            if enhanced_content is None:
                self.console.print("[red]Enhancement failed to generate content.[/red]")
                return False
            
            # Display enhanced content if preview mode
            if preview:
                self._display_content(enhanced_content, language, "Enhanced Content", "green")
                
                # Show differences
                self._show_differences(original_content, enhanced_content, file_path.name)
            
            # Handle output
            success = self._handle_output(
                file_path, original_content, enhanced_content, 
                output_path, backup, enhancement_type
            )
            
            return success
            
        except Exception as e:
            self.console.print(f"[red]Enhancement failed: {e}[/red]")
            return False
    
    def batch_enhance(self, args):
        """Enhance multiple files in batch mode."""
        try:
            input_path = Path(args['input'])
            output_path = Path(args.get('output', input_path))
            enhancement_type = args.get('enhancement_type', 'style')
            recursive = args.get('recursive', False)
            file_pattern = args.get('pattern', '*')
            
            if not input_path.exists():
                self.console.print(f"[red]Error: Input path does not exist: {input_path}[/red]")
                return False
            
            # Find files
            if input_path.is_file():
                files = [input_path]
            else:
                pattern = f"**/{file_pattern}" if recursive else file_pattern
                files = [f for f in input_path.glob(pattern) if f.is_file()]
            
            if not files:
                self.console.print("[yellow]No files found to enhance.[/yellow]")
                return False
            
            self.console.print(Panel.fit(
                f"[bold]Batch AI Enhancement[/bold]\n"
                f"Input: {input_path}\n"
                f"Output: {output_path}\n"
                f"Enhancement: {enhancement_type}\n"
                f"Files: {len(files)} found",
                title="🚀 Batch Enhancement Started"
            ))
            
            results = self._process_batch_enhancement(
                files, input_path, output_path, enhancement_type
            )
            
            self._display_batch_results(results)
            return results['successful'] > 0
            
        except Exception as e:
            self.console.print(f"[red]Batch enhancement failed: {e}[/red]")
            return False
    
    def _apply_enhancement(self, content: str, enhancement_type: str, 
                          language: str, file_path: Path) -> Optional[str]:
        """Apply enhancement using AI service."""
        try:
            self.console.print(f"[blue]Applying {enhancement_type} enhancement...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Processing with AI...", total=None)
                
                # Use the FileEnhancer from file_manager
                enhanced_content = self.file_enhancer.enhance_content(
                    content=content,
                    enhancement_type=enhancement_type,
                    language=language,
                    file_path=file_path
                )
                
                progress.update(task, completed=1)
                
                if enhanced_content and enhanced_content.strip():
                    return enhanced_content
                else:
                    self.console.print("[red]AI service returned empty content.[/red]")
                    return None
                    
        except Exception as e:
            self.console.print(f"[red]AI enhancement error: {e}[/red]")
            return None
    
    def _process_batch_enhancement(self, files: List[Path], input_path: Path,
                                 output_path: Path, enhancement_type: str) -> Dict:
        """Process multiple files for enhancement."""
        results = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Enhancing files...", total=len(files))
            
            for file_path in files:
                try:
                    progress.update(task, advance=1, description=f"Processing {file_path.name}")
                    
                    # Determine output file path
                    if input_path.is_file():
                        relative_path = Path(file_path.name)
                    else:
                        relative_path = file_path.relative_to(input_path)
                    
                    output_file = output_path / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Read original content
                    original_content = self._read_file(file_path)
                    if original_content is None:
                        results['skipped'] += 1
                        continue
                    
                    # Auto-detect language
                    language = self._detect_language(file_path)
                    
                    # Apply enhancement
                    enhanced_content = self._apply_enhancement(
                        original_content, enhancement_type, language, file_path
                    )
                    
                    if enhanced_content:
                        # Write enhanced content
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(enhanced_content)
                        
                        results['successful'] += 1
                        results['details'].append({
                            'file': str(file_path),
                            'status': 'enhanced',
                            'output': str(output_file)
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'file': str(file_path),
                            'status': 'failed',
                            'output': str(output_file)
                        })
                        
                except Exception as e:
                    results['failed'] += 1
                    results['details'].append({
                        'file': str(file_path),
                        'status': f'error: {e}',
                        'output': str(output_file)
                    })
        
        return results
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content with proper encoding handling."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # If text decoding fails, read as binary
            with open(file_path, 'rb') as f:
                content = f.read()
                # Try to decode as text, replace problematic characters
                return content.decode('utf-8', errors='replace')
                
        except Exception as e:
            self.console.print(f"[red]Error reading file {file_path}: {e}[/red]")
            return None
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.html': 'html',
            '.css': 'css',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.ts': 'typescript',
            '.sql': 'sql',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text'
        }
        
        return extension_languages.get(file_path.suffix.lower(), 'text')
    
    def _display_content(self, content: str, language: str, title: str, style: str):
        """Display file content in a formatted panel."""
        try:
            # Limit content for display to avoid overwhelming output
            display_content = content
            if len(display_content) > 10000:
                display_content = content[:10000] + "\n\n... (content truncated for display)"
            
            syntax = Syntax(display_content, language, theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title=title, style=style))
        except Exception as e:
            self.console.print(f"[yellow]Could not display syntax: {e}[/yellow]")
            # Fallback to simple display
            self.console.print(Panel(display_content[:2000], title=title, style=style))
    
    def _show_differences(self, original: str, enhanced: str, filename: str):
        """Show basic differences between original and enhanced content."""
        if original == enhanced:
            self.console.print("[yellow]No changes detected in content.[/yellow]")
            return
        
        original_lines = original.splitlines()
        enhanced_lines = enhanced.splitlines()
        
        table = Table(title=f"📊 Changes Summary - {filename}", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Original", justify="right")
        table.add_column("Enhanced", justify="right")
        table.add_column("Change", justify="right")
        
        # Line count
        line_change = len(enhanced_lines) - len(original_lines)
        table.add_row("Lines", str(len(original_lines)), str(len(enhanced_lines)), 
                     f"{line_change:+d}")
        
        # Character count
        char_change = len(enhanced) - len(original)
        table.add_row("Characters", str(len(original)), str(len(enhanced)),
                     f"{char_change:+d}")
        
        # Word count (approximate)
        orig_words = len(original.split())
        enh_words = len(enhanced.split())
        word_change = enh_words - orig_words
        table.add_row("Words", str(orig_words), str(enh_words),
                     f"{word_change:+d}")
        
        self.console.print(table)
    
    def _handle_output(self, file_path: Path, original_content: str,
                      enhanced_content: str, output_path: Optional[Path],
                      backup: bool, enhancement_type: str) -> bool:
        """Handle output of enhanced content."""
        try:
            if output_path:
                # Write to specified output path
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)
                
                self.console.print(f"[green]✓ Enhanced version saved to: {output_path}[/green]")
                return True
            else:
                # Interactive mode - ask for confirmation
                if not Confirm.ask("\nSave enhanced version?"):
                    self.console.print("[yellow]Enhancement not saved.[/yellow]")
                    return False
                
                # Create backup if requested
                if backup:
                    backup_path = file_path.with_suffix(file_path.suffix + f'.{enhancement_type}.backup')
                    shutil.copy2(file_path, backup_path)
                    self.console.print(f"[blue]✓ Backup created: {backup_path}[/blue]")
                
                # Overwrite original file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)
                
                self.console.print(f"[green]✓ File enhanced successfully: {file_path}[/green]")
                return True
                
        except Exception as e:
            self.console.print(f"[red]Error saving enhanced file: {e}[/red]")
            return False
    
    def _display_batch_results(self, results: Dict):
        """Display batch enhancement results."""
        total = results['successful'] + results['failed'] + results['skipped']
        
        table = Table(title="📊 Batch Enhancement Results", show_header=True)
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        success_pct = (results['successful'] / total * 100) if total > 0 else 0
        fail_pct = (results['failed'] / total * 100) if total > 0 else 0
        skip_pct = (results['skipped'] / total * 100) if total > 0 else 0
        
        table.add_row("✅ Successful", str(results['successful']), f"{success_pct:.1f}%")
        table.add_row("❌ Failed", str(results['failed']), f"{fail_pct:.1f}%")
        table.add_row("⏭️ Skipped", str(results['skipped']), f"{skip_pct:.1f}%")
        table.add_row("📊 Total", str(total), "100%")
        
        self.console.print(table)
        
        # Show some successful files
        successful_files = [d for d in results['details'] if d['status'] == 'enhanced']
        if successful_files:
            self.console.print("\n[green]Successfully Enhanced:[/green]")
            for detail in successful_files[:3]:
                self.console.print(f"  • {detail['file']}")
            if len(successful_files) > 3:
                self.console.print(f"  ... and {len(successful_files) - 3} more")
    
    def list_enhancement_types(self):
        """Display available enhancement types."""
        table = Table(title="🎨 Available Enhancement Types", show_header=True)
        table.add_column("Type", style="bold cyan")
        table.add_column("Description", style="white")
        
        for enh_type, description in self.enhancement_types.items():
            table.add_row(enh_type, description)
        
        self.console.print(table)