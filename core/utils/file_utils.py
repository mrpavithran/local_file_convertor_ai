"""
File Utility Functions
Common file operations and utilities.
"""

import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import hashlib
from datetime import datetime
import fnmatch

class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_directory(directory_path: str) -> bool:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            Success status
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Failed to create directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def safe_copy(source: str, destination: str, overwrite: bool = False) -> bool:
        """
        Safely copy a file with error handling.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            Success status
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source file does not exist: {source}")
            
            if dest_path.exists() and not overwrite:
                raise FileExistsError(f"Destination file already exists: {destination}")
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_path, dest_path)
            return True
            
        except Exception as e:
            print(f"File copy failed: {e}")
            return False
    
    @staticmethod
    def safe_move(source: str, destination: str, overwrite: bool = False) -> bool:
        """
        Safely move a file with error handling.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            Success status
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source file does not exist: {source}")
            
            if dest_path.exists() and not overwrite:
                raise FileExistsError(f"Destination file already exists: {destination}")
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(source_path, dest_path)
            return True
            
        except Exception as e:
            print(f"File move failed: {e}")
            return False
    
    @staticmethod
    def find_files(directory: str, patterns: List[str], recursive: bool = True) -> List[str]:
        """
        Find files matching patterns in directory.
        
        Args:
            directory: Directory to search
            patterns: List of file patterns (e.g., ['*.txt', '*.md'])
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        directory_path = Path(directory)
        matches = []
        
        if not directory_path.exists():
            return matches
        
        search_method = directory_path.rglob if recursive else directory_path.glob
        
        for pattern in patterns:
            for file_path in search_method(pattern):
                if file_path.is_file():
                    matches.append(str(file_path))
        
        return sorted(matches)
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            File information dictionary
        """
        path = Path(file_path)
        
        if not path.exists() or not path.is_file():
            return {}
        
        stat = path.stat()
        
        return {
            'path': str(path),
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'parent': str(path.parent),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK),
            'is_executable': os.access(file_path, os.X_OK)
        }
    
    @staticmethod
    def create_backup(file_path: str, backup_dir: str = None, suffix: str = '.bak') -> Optional[str]:
        """
        Create a backup of a file.
        
        Args:
            file_path: Path to file to backup
            backup_dir: Backup directory (uses file directory if None)
            suffix: Backup file suffix
            
        Returns:
            Path to backup file or None if failed
        """
        try:
            source_path = Path(file_path)
            
            if not source_path.exists():
                return None
            
            if backup_dir:
                backup_path = Path(backup_dir) / f"{source_path.name}{suffix}"
            else:
                backup_path = source_path.parent / f"{source_path.name}{suffix}"
            
            FileUtils.safe_copy(file_path, str(backup_path), overwrite=True)
            return str(backup_path)
            
        except Exception as e:
            print(f"Backup creation failed: {e}")
            return None
    
    @staticmethod
    def read_file_chunks(file_path: str, chunk_size: int = 8192) -> Generator[bytes, None, None]:
        """
        Read file in chunks for memory efficiency.
        
        Args:
            file_path: Path to file
            chunk_size: Size of each chunk in bytes
            
        Yields:
            File chunks
        """
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> Optional[str]:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm
            
        Returns:
            File hash or None if failed
        """
        try:
            hash_func = hashlib.new(algorithm)
            
            for chunk in FileUtils.read_file_chunks(file_path):
                hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            print(f"Hash calculation failed: {e}")
            return None
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """
        Calculate total size of directory contents.
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        directory_path = Path(directory)
        
        if not directory_path.exists():
            return 0
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    @staticmethod
    def clean_filename(filename: str, replace_with: str = '_') -> str:
        """
        Clean filename by removing invalid characters.
        
        Args:
            filename: Original filename
            replace_with: Character to replace invalid chars with
            
        Returns:
            Cleaned filename
        """
        # Windows reserved characters
        invalid_chars = '<>:"/\\|?*'
        
        cleaned = filename
        for char in invalid_chars:
            cleaned = cleaned.replace(char, replace_with)
        
        # Remove leading/trailing spaces and dots
        cleaned = cleaned.strip().strip('.')
        
        return cleaned
    
    @staticmethod
    def create_temp_file(content: str = '', suffix: str = '.tmp') -> str:
        """
        Create a temporary file with optional content.
        
        Args:
            content: File content
            suffix: File suffix
            
        Returns:
            Path to temporary file
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                if content:
                    f.write(content)
                return f.name
        except Exception as e:
            print(f"Temp file creation failed: {e}")
            return ''