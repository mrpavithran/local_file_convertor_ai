"""
Manage output directory structures and file organization.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from rich.console import Console
from rich.tree import Tree


class OutputManager:
    """Manage output directories with flexible structure options."""
    
    def __init__(self, base_output_dir: Union[str, Path]):
        self.base_output_dir = Path(base_output_dir)
        self.console = Console()
        
        # Create base directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_output_structure(self, 
                             structure_type: str = 'maintained',
                             categories: Optional[List[str]] = None) -> Path:
        """
        Setup output directory structure.
        
        Args:
            structure_type: 'maintained', 'flattened', or 'categorized'
            categories: List of categories for categorized structure
            
        Returns:
            Path to the output directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if structure_type == 'maintained':
            output_dir = self.base_output_dir / 'maintained_structure' / timestamp
        elif structure_type == 'flattened':
            output_dir = self.base_output_dir / 'flattened_structure' / timestamp
        elif structure_type == 'categorized':
            output_dir = self.base_output_dir / 'categorized_structure' / timestamp
        else:
            output_dir = self.base_output_dir / 'output' / timestamp
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category subdirectories if needed
        if structure_type == 'categorized' and categories:
            for category in categories:
                (output_dir / category).mkdir(parents=True, exist_ok=True)
        
        self.console.print(f"[green]Output directory created: {output_dir}[/green]")
        return output_dir
    
    def maintain_structure(self, 
                          source_file: Path, 
                          source_root: Path, 
                          output_dir: Path) -> Path:
        """
        Maintain original directory structure in output.
        
        Args:
            source_file: Source file path
            source_root: Root of source directory
            output_dir: Output directory
            
        Returns:
            Path for output file
        """
        try:
            relative_path = source_file.relative_to(source_root)
            output_path = output_dir / relative_path
            
            # Create parent directories
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return output_path
            
        except ValueError:
            # If file is not relative to source_root, place in root of output
            return output_dir / source_file.name
    
    def flatten_structure(self, source_file: Path, output_dir: Path) -> Path:
        """
        Flatten directory structure (all files in single directory).
        
        Args:
            source_file: Source file path
            output_dir: Output directory
            
        Returns:
            Path for output file
        """
        # Handle duplicate names
        base_name = source_file.stem
        extension = source_file.suffix
        counter = 1
        
        output_path = output_dir / source_file.name
        
        while output_path.exists():
            output_path = output_dir / f"{base_name}_{counter}{extension}"
            counter += 1
        
        return output_path
    
    def categorize_structure(self, 
                           source_file: Path, 
                           output_dir: Path,
                           category: str) -> Path:
        """
        Organize files by category.
        
        Args:
            source_file: Source file path
            output_dir: Output directory
            category: File category
            
        Returns:
            Path for output file
        """
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle duplicate names
        base_name = source_file.stem
        extension = source_file.suffix
        counter = 1
        
        output_path = category_dir / source_file.name
        
        while output_path.exists():
            output_path = category_dir / f"{base_name}_{counter}{extension}"
            counter += 1
        
        return output_path
    
    def copy_with_structure(self, 
                          source_file: Path, 
                          target_file: Path,
                          overwrite: bool = False) -> bool:
        """
        Copy file to target location with structure creation.
        
        Args:
            source_file: Source file path
            target_file: Target file path
            overwrite: Whether to overwrite existing files
            
        Returns:
            True if successful
        """
        try:
            if target_file.exists() and not overwrite:
                self.console.print(f"[yellow]File exists, skipping: {target_file}[/yellow]")
                return False
            
            # Ensure target directory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_file, target_file)
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error copying {source_file}: {e}[/red]")
            return False
    
    def display_output_tree(self, output_dir: Path, max_depth: int = 3):
        """
        Display directory tree structure.
        
        Args:
            output_dir: Directory to display
            max_depth: Maximum depth to display
        """
        def build_tree(directory: Path, tree: Tree, depth: int = 0):
            if depth > max_depth:
                return
                
            try:
                items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                
                for item in items:
                    if item.is_dir():
                        branch = tree.add(f"[blue]{item.name}/[/blue]")
                        build_tree(item, branch, depth + 1)
                    else:
                        # File size display
                        size = item.stat().st_size
                        size_str = self._format_size(size)
                        tree.add(f"{item.name} [dim]({size_str})[/dim]")
            except PermissionError:
                tree.add("[red]Permission denied[/red]")
        
        if output_dir.exists():
            tree = Tree(f"[bold green]{output_dir}[/bold green]")
            build_tree(output_dir, tree)
            self.console.print(tree)
        else:
            self.console.print("[red]Output directory does not exist[/red]")
    
    def cleanup_old_outputs(self, keep_count: int = 5):
        """
        Clean up old output directories, keeping only the most recent.
        
        Args:
            keep_count: Number of recent outputs to keep
        """
        try:
            # Get all output directories sorted by modification time
            outputs = []
            for structure_type in ['maintained_structure', 'flattened_structure', 'categorized_structure']:
                structure_dir = self.base_output_dir / structure_type
                if structure_dir.exists():
                    for output_dir in structure_dir.iterdir():
                        if output_dir.is_dir():
                            outputs.append((output_dir.stat().st_mtime, output_dir))
            
            # Sort by modification time (newest first)
            outputs.sort(reverse=True)
            
            # Remove old directories
            for i, (mtime, output_dir) in enumerate(outputs):
                if i >= keep_count:
                    try:
                        shutil.rmtree(output_dir)
                        self.console.print(f"[yellow]Removed old output: {output_dir}[/yellow]")
                    except Exception as e:
                        self.console.print(f"[red]Error removing {output_dir}: {e}[/red]")
                        
        except Exception as e:
            self.console.print(f"[red]Error during cleanup: {e}[/red]")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"