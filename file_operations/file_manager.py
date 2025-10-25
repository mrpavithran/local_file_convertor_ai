"""
File system operations manager with AI-enhanced capabilities.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import magic
from rich.console import Console
from rich.progress import Progress

from .type_detector import TypeDetector
from .error_handler import ErrorHandler


class FileManager:
    """Advanced file system operations with AI integration."""
    
    def __init__(self):
        self.console = Console()
        self.type_detector = TypeDetector()
        self.error_handler = ErrorHandler()
        self.supported_extensions = {
            'documents': ['.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt'],
            'spreadsheets': ['.xlsx', '.xls', '.csv', '.ods'],
            'presentations': ['.pptx', '.ppt', '.odp'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'code': ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.php', '.rb']
        }
    
    def scan_directory(self, directory: Union[str, Path], 
                      recursive: bool = True,
                      file_types: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """
        Scan directory and categorize files by type.
        
        Args:
            directory: Path to scan
            recursive: Whether to scan subdirectories
            file_types: Specific file types to include
            
        Returns:
            Dictionary of file categories with file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        results = {
            'documents': [],
            'spreadsheets': [], 
            'presentations': [],
            'images': [],
            'archives': [],
            'code': [],
            'other': []
        }
        
        pattern = "**/*" if recursive else "*"
        
        # Count total files for progress tracking
        file_count = sum(1 for _ in directory.glob(pattern) if _.is_file())
        
        with Progress() as progress:
            task = progress.add_task("Scanning directory...", total=file_count)
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_category = self.type_detector.categorize_file(file_path)
                    
                    if file_types and file_category not in file_types:
                        progress.update(task, advance=1)
                        continue
                    
                    if file_category in results:
                        results[file_category].append(file_path)
                    else:
                        results['other'].append(file_path)
                    
                    progress.update(task, advance=1)
        
        return results
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Basic file info
            stat = file_path.stat()
            file_type = self.type_detector.detect_file_type(file_path)
            
            info = {
                'path': str(file_path.absolute()),
                'name': file_path.name,
                'size_bytes': stat.st_size,
                'size_human': self._format_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'file_type': file_type,
                'category': self.type_detector.categorize_file(file_path),
                'mime_type': self.type_detector.get_mime_type(file_path),
                'extension': file_path.suffix.lower(),
                'permissions': oct(stat.st_mode)[-3:],
                'hash_md5': self._calculate_hash(file_path, 'md5'),
                'hash_sha256': self._calculate_hash(file_path, 'sha256')
            }
            
            # Additional info based on file type
            if file_type == 'image':
                info.update(self._get_image_info(file_path))
            elif file_type == 'text':
                info.update(self._get_text_info(file_path))
            elif file_type == 'code':
                info.update(self._get_code_info(file_path))
                
            return info
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Getting info for {file_path}")
            return {}
    
    def organize_files(self, source_dir: Union[str, Path], 
                      target_dir: Union[str, Path],
                      organization_strategy: str = 'category') -> Dict[str, int]:
        """
        Organize files based on specified strategy.
        
        Args:
            source_dir: Source directory to organize
            target_dir: Target directory for organized files
            organization_strategy: 'category', 'date', 'extension'
            
        Returns:
            Dictionary with organization results
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        files = self.scan_directory(source_dir, recursive=True)
        results = {'moved': 0, 'skipped': 0, 'errors': 0}
        
        # Calculate total files
        total_files = sum(len(file_list) for file_list in files.values())
        
        with Progress() as progress:
            task = progress.add_task("Organizing files...", total=total_files)
            
            for category, file_list in files.items():
                for file_path in file_list:
                    try:
                        if organization_strategy == 'category':
                            new_path = target_dir / category / file_path.name
                        elif organization_strategy == 'date':
                            file_info = self.get_file_info(file_path)
                            date_str = file_info['modified'].strftime('%Y-%m')
                            new_path = target_dir / date_str / file_path.name
                        elif organization_strategy == 'extension':
                            ext = file_path.suffix.lower()[1:] or 'no_extension'
                            new_path = target_dir / ext / file_path.name
                        else:
                            new_path = target_dir / file_path.name
                        
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Handle duplicate names
                        counter = 1
                        original_new_path = new_path
                        while new_path.exists():
                            stem = original_new_path.stem
                            suffix = original_new_path.suffix
                            new_path = original_new_path.parent / f"{stem}_{counter}{suffix}"
                            counter += 1
                        
                        # Only move if source and target are different
                        if file_path.resolve() != new_path.resolve():
                            shutil.move(str(file_path), str(new_path))
                            results['moved'] += 1
                        else:
                            results['skipped'] += 1
                        
                    except Exception as e:
                        self.error_handler.handle_error(e, f"Moving {file_path}")
                        results['errors'] += 1
                    
                    progress.update(task, advance=1)
        
        return results
    
    def batch_rename(self, directory: Union[str, Path], 
                    pattern: str,
                    replacement: str,
                    dry_run: bool = True) -> List[Dict[str, str]]:
        """
        Batch rename files using pattern matching.
        
        Args:
            directory: Directory containing files to rename
            pattern: Pattern to match (supports regex)
            replacement: Replacement pattern
            dry_run: If True, only show what would be renamed
            
        Returns:
            List of rename operations
        """
        import re
        
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        operations = []
        
        for file_path in directory.glob('*'):
            if file_path.is_file():
                new_name = re.sub(pattern, replacement, file_path.name)
                
                if new_name != file_path.name:
                    operation = {
                        'old_name': file_path.name,
                        'new_name': new_name,
                        'old_path': str(file_path),
                        'new_path': str(file_path.parent / new_name)
                    }
                    
                    if not dry_run:
                        try:
                            file_path.rename(file_path.parent / new_name)
                            operation['status'] = 'renamed'
                        except Exception as e:
                            operation['status'] = f'error: {e}'
                    else:
                        operation['status'] = 'would_rename'
                    
                    operations.append(operation)
        
        return operations
    
    def find_duplicates(self, directory: Union[str, Path], 
                       method: str = 'hash') -> Dict[str, List[Path]]:
        """
        Find duplicate files in directory.
        
        Args:
            directory: Directory to scan for duplicates
            method: Detection method ('hash', 'name_size')
            
        Returns:
            Dictionary of duplicate file groups
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        file_groups = {}
        duplicates = {}
        
        with Progress() as progress:
            files = list(directory.rglob('*'))
            task = progress.add_task("Finding duplicates...", total=len(files))
            
            for file_path in files:
                if file_path.is_file():
                    if method == 'hash':
                        key = self._calculate_hash(file_path, 'md5')
                    elif method == 'name_size':
                        key = f"{file_path.name}_{file_path.stat().st_size}"
                    else:
                        key = f"{file_path.name}_{file_path.stat().st_size}"
                    
                    if key in file_groups:
                        file_groups[key].append(file_path)
                    else:
                        file_groups[key] = [file_path]
                
                progress.update(task, advance=1)
        
        # Only return groups with duplicates
        for key, files in file_groups.items():
            if len(files) > 1:
                duplicates[key] = files
        
        return duplicates
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"
    
    def _calculate_hash(self, file_path: Path, algorithm: str) -> str:
        """Calculate file hash."""
        hash_func = getattr(hashlib, algorithm)()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            self.error_handler.handle_error(e, f"Calculating hash for {file_path}")
            return "unavailable"
    
    def _get_image_info(self, file_path: Path) -> Dict[str, Any]:
        """Get image-specific information."""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                return {
                    'dimensions': f"{img.width}x{img.height}",
                    'format': img.format,
                    'mode': img.mode,
                    'color_profile': getattr(img, 'info', {}).get('icc_profile', 'none')
                }
        except ImportError:
            self.console.print("[yellow]PIL not installed, limited image info available[/yellow]")
            return {'dimensions': 'unknown', 'format': 'unknown'}
        except Exception as e:
            self.error_handler.handle_error(e, f"Getting image info for {file_path}")
            return {'dimensions': 'unknown', 'format': 'unknown'}
    
    def _get_text_info(self, file_path: Path) -> Dict[str, Any]:
        """Get text file information."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        lines = content.splitlines()
                        
                        return {
                            'line_count': len(lines),
                            'word_count': len(content.split()),
                            'character_count': len(content),
                            'encoding': encoding
                        }
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, count as binary
            return {'line_count': 0, 'word_count': 0, 'character_count': 0, 'encoding': 'binary'}
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Getting text info for {file_path}")
            return {'line_count': 0, 'word_count': 0, 'character_count': 0, 'encoding': 'unknown'}
    
    def _get_code_info(self, file_path: Path) -> Dict[str, Any]:
        """Get code file information."""
        text_info = self._get_text_info(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Basic code analysis (language-agnostic patterns)
                functions = content.count('def ') + content.count('function ') + content.count('func ')
                classes = content.count('class ') + content.count('struct ') + content.count('interface ')
                imports = content.count('import ') + content.count('require') + content.count('#include')
                
                text_info.update({
                    'functions': functions,
                    'classes': classes,
                    'imports': imports
                })
        except Exception as e:
            self.error_handler.handle_error(e, f"Getting code info for {file_path}")
        
        return text_info


class FileEnhancer:
    """AI-powered file enhancement operations."""
    
    def __init__(self, ai_client=None):
        self.console = Console()
        self.ai_client = ai_client
    
    def enhance_content(self, content: str, enhancement_type: str = "grammar", 
                       language: str = "text", file_path: Optional[Path] = None) -> str:
        """
        Enhance file content using AI.
        
        Args:
            content: The content to enhance
            enhancement_type: Type of enhancement
            language: Programming language or text type
            file_path: Optional file path for context
            
        Returns:
            Enhanced content
        """
        try:
            # Placeholder for AI enhancement logic
            # This would integrate with your AI service (Ollama, OpenAI, etc.)
            
            enhancement_prompts = {
                "grammar": "Please correct grammar and spelling errors while preserving meaning:",
                "style": "Please improve writing style and clarity:",
                "refactor": "Please refactor this code for better structure and readability:",
                "document": "Please add appropriate documentation and comments:",
                "optimize": "Please optimize for performance and efficiency:"
            }
            
            prompt = enhancement_prompts.get(enhancement_type, "Please enhance this content:")
            
            # Simulate AI enhancement (replace with actual AI call)
            enhanced_content = self._simulate_ai_enhancement(content, enhancement_type, language)
            
            return enhanced_content
            
        except Exception as e:
            self.console.print(f"[red]Enhancement error: {e}[/red]")
            return content
    
    def _simulate_ai_enhancement(self, content: str, enhancement_type: str, language: str) -> str:
        """Simulate AI enhancement (replace with actual AI integration)."""
        header = f"# Enhanced with AI ({enhancement_type}) - Language: {language}\n\n"
        
        if enhancement_type == "grammar":
            return header + content  # In real implementation, this would be grammar-corrected
        
        elif enhancement_type == "document" and language in ["python", "java", "javascript"]:
            # Add basic documentation
            documented = header
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') or line.strip().startswith('function '):
                    documented += f'\"\"\"\nDocumentation for: {line.strip()}\n\"\"\"\n'
                documented += line + '\n'
            return documented
        
        else:
            return header + content