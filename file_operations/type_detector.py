"""
Advanced file type detection using multiple methods.
"""

import os
import magic
from pathlib import Path
from typing import Dict, List, Optional
import filetype


class TypeDetector:
    """Detect file types using extension, magic numbers, and content analysis."""
    
    def __init__(self):
        self.mime_detector = magic.Magic(mime=True)
        
        # Extended file type mappings
        self.type_categories = {
            'documents': {
                'extensions': ['.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt', '.md'],
                'mime_prefixes': ['application/pdf', 'application/msword', 'application/vnd.openxmlformats', 'text/']
            },
            'spreadsheets': {
                'extensions': ['.xlsx', '.xls', '.csv', '.ods'],
                'mime_prefixes': ['application/vnd.ms-excel', 'application/vnd.openxmlformats', 'text/csv']
            },
            'presentations': {
                'extensions': ['.pptx', '.ppt', '.odp'],
                'mime_prefixes': ['application/vnd.ms-powerpoint', 'application/vnd.openxmlformats']
            },
            'images': {
                'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'],
                'mime_prefixes': ['image/']
            },
            'audio': {
                'extensions': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
                'mime_prefixes': ['audio/']
            },
            'video': {
                'extensions': ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
                'mime_prefixes': ['video/']
            },
            'archives': {
                'extensions': ['.zip', '.rar', '.7z', '.tar', '.gz'],
                'mime_prefixes': ['application/zip', 'application/x-rar', 'application/x-7z']
            },
            'code': {
                'extensions': ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.php', '.rb', '.go', '.rs'],
                'mime_prefixes': ['text/', 'application/javascript']
            },
            'executables': {
                'extensions': ['.exe', '.bin', '.sh', '.bat'],
                'mime_prefixes': ['application/x-executable', 'application/x-shellscript']
            }
        }
    
    def detect_file_type(self, file_path: Path) -> str:
        """
        Detect file type using multiple methods.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type
        """
        # Method 1: Extension-based detection
        extension_type = self._detect_by_extension(file_path)
        
        # Method 2: Magic number detection
        magic_type = self._detect_by_magic(file_path)
        
        # Method 3: Content-based detection
        content_type = self._detect_by_content(file_path)
        
        # Prioritize magic number detection as most accurate
        if magic_type and magic_type != 'unknown':
            return magic_type
        elif content_type and content_type != 'unknown':
            return content_type
        else:
            return extension_type or 'unknown'
    
    def categorize_file(self, file_path: Path) -> str:
        """
        Categorize file into broad categories.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File category
        """
        file_type = self.detect_file_type(file_path)
        extension = file_path.suffix.lower()
        
        for category, info in self.type_categories.items():
            if (extension in info['extensions'] or 
                any(file_type.startswith(prefix) for prefix in info['mime_prefixes'])):
                return category
        
        return 'other'
    
    def get_mime_type(self, file_path: Path) -> str:
        """
        Get MIME type of file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        try:
            return self.mime_detector.from_file(str(file_path))
        except:
            return 'application/octet-stream'
    
    def is_text_file(self, file_path: Path) -> bool:
        """
        Check if file is a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if text file
        """
        mime_type = self.get_mime_type(file_path)
        return mime_type.startswith('text/') or mime_type in [
            'application/javascript',
            'application/json',
            'application/xml'
        ]
    
    def is_binary_file(self, file_path: Path) -> bool:
        """
        Check if file is a binary file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if binary file
        """
        return not self.is_text_file(file_path)
    
    def _detect_by_extension(self, file_path: Path) -> Optional[str]:
        """Detect file type by extension."""
        extension_map = {
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.txt': 'text',
            '.py': 'code',
            '.js': 'code',
            '.java': 'code',
            '.html': 'code',
            '.css': 'code',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.gif': 'image',
            '.mp4': 'video',
            '.mp3': 'audio',
            '.zip': 'archive',
            '.xlsx': 'spreadsheet',
            '.csv': 'spreadsheet',
            '.pptx': 'presentation'
        }
        
        return extension_map.get(file_path.suffix.lower())
    
    def _detect_by_magic(self, file_path: Path) -> Optional[str]:
        """Detect file type using magic numbers."""
        try:
            mime_type = self.get_mime_type(file_path)
            
            # Map MIME types to our categories
            mime_map = {
                'application/pdf': 'document',
                'application/msword': 'document',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
                'text/plain': 'text',
                'text/html': 'code',
                'application/javascript': 'code',
                'text/x-python': 'code',
                'image/jpeg': 'image',
                'image/png': 'image',
                'image/gif': 'image',
                'video/mp4': 'video',
                'audio/mpeg': 'audio',
                'application/zip': 'archive',
                'application/vnd.ms-excel': 'spreadsheet',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'spreadsheet',
                'text/csv': 'spreadsheet'
            }
            
            return mime_map.get(mime_type, 'unknown')
            
        except:
            return 'unknown'
    
    def _detect_by_content(self, file_path: Path) -> Optional[str]:
        """Detect file type by analyzing content."""
        try:
            # For small files, read and analyze content
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return 'unknown'
            
            with open(file_path, 'rb') as f:
                header = f.read(512)  # Read first 512 bytes
            
            # Check for common file signatures
            signatures = {
                b'%PDF': 'document',
                b'PK\x03\x04': 'archive',  # ZIP
                b'\x89PNG': 'image',
                b'\xff\xd8\xff': 'image',  # JPEG
                b'GIF8': 'image',
                b'<!DOCTYPE': 'code',
                b'<?php': 'code',
                b'#!/': 'code'  # Script
            }
            
            for signature, file_type in signatures.items():
                if header.startswith(signature):
                    return file_type
            
            # Check if it's text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)
                return 'text'
            except:
                return 'binary'
                
        except:
            return 'unknown'