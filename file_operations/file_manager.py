"""
File system operations manager with AI-enhanced capabilities.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
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
        
        with Progress() as progress:
            task = progress.add_task("Scanning directory...", total=1)
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_category = self.type_detector.categorize_file(file_path)
                    
                    if file_types and file_category not in file_types:
                        continue
                    
                    if file_category in results:
                        results[file_category].append(file_path)
                    else:
                        results['other'].append(file_path)
            
            progress.update(task, advance=1)
        
        return results
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, any]:
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
        
        with Progress() as progress:
            total_files = sum(len(files) for files in files.values())
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
                            new_path = original_new_path.parent / f"{stem}_{counter}{original_new_path.suffix}"
                            counter += 1
                        
                        shutil.move(str(file_path), str(new_path))
                        results['moved'] += 1
                        
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
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def _calculate_hash(self, file_path: Path, algorithm: str) -> str:
        """Calculate file hash."""
        hash_func = getattr(hashlib, algorithm)()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except:
            return "unavailable"
    
    def _get_image_info(self, file_path: Path) -> Dict[str, any]:
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
        except:
            return {'dimensions': 'unknown', 'format': 'unknown'}
    
    def _get_text_info(self, file_path: Path) -> Dict[str, any]:
        """Get text file information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                
                return {
                    'line_count': len(lines),
                    'word_count': len(content.split()),
                    'character_count': len(content),
                    'encoding': 'utf-8'
                }
        except:
            return {'line_count': 0, 'word_count': 0, 'character_count': 0}
    
    def _get_code_info(self, file_path: Path) -> Dict[str, any]:
        """Get code file information."""
        text_info = self._get_text_info(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Basic code analysis
                functions = content.count('def ')  # Python specific
                classes = content.count('class ')
                imports = content.count('import ')
                
                text_info.update({
                    'functions': functions,
                    'classes': classes,
                    'imports': imports
                })
        except:
            pass
        
        return text_info