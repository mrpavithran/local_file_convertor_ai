"""
Enhanced File Type Detector with Conversion Support
"""

import os
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import magic  # python-magic-bin for Windows, python-magic for Linux

class FileTypeDetector:
    """
    Detect file types and determine conversion capabilities
    """
    
    def __init__(self):
        self.supported_conversions = {
            # Document conversions
            'pdf': {'docx', 'txt'},
            'docx': {'pdf', 'txt'}, 
            'doc': {'pdf', 'docx', 'txt'},
            'txt': {'pdf', 'docx'},
            'csv': {'xlsx', 'json'},
            'xlsx': {'csv', 'pdf'},
            
            # Image conversions
            'jpg': {'png', 'webp', 'pdf'},
            'jpeg': {'png', 'webp', 'pdf'},
            'png': {'jpg', 'webp', 'pdf'},
            'gif': {'png', 'jpg', 'pdf'},
            'webp': {'png', 'jpg', 'pdf'},
            
            # Text formats
            'json': {'csv', 'yaml', 'xml'},
            'yaml': {'json', 'xml'},
            'xml': {'json', 'yaml'}
        }
        
        # Initialize magic
        try:
            self.magic = magic.Magic(mime=True)
        except:
            self.magic = None
    
    def detect_file_type(self, file_path: str) -> Dict[str, str]:
        """
        Detect file type using multiple methods
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        result = {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'size': os.path.getsize(file_path)
        }
        
        # Method 1: File extension
        ext = Path(file_path).suffix.lower().lstrip('.')
        result['extension'] = ext
        
        # Method 2: MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        result['mime_type'] = mime_type or 'unknown'
        
        # Method 3: Magic (if available)
        if self.magic:
            try:
                magic_result = self.magic.from_file(file_path)
                result['magic_type'] = magic_result
            except:
                result['magic_type'] = 'unknown'
        
        # Determine category
        result['category'] = self._categorize_file(ext, mime_type)
        
        # Get available conversions
        result['convertible_to'] = self.get_available_conversions(ext)
        
        return result
    
    def _categorize_file(self, extension: str, mime_type: str) -> str:
        """Categorize file into broad categories"""
        document_extensions = {'pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'}
        spreadsheet_extensions = {'csv', 'xlsx', 'xls', 'ods'}
        image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'}
        presentation_extensions = {'pptx', 'ppt', 'odp'}
        
        if extension in document_extensions:
            return 'document'
        elif extension in spreadsheet_extensions:
            return 'spreadsheet'
        elif extension in image_extensions:
            return 'image'
        elif extension in presentation_extensions:
            return 'presentation'
        elif extension in {'json', 'yaml', 'xml'}:
            return 'data'
        else:
            return 'other'
    
    def get_available_conversions(self, from_extension: str) -> List[str]:
        """Get list of available conversion targets for a file type"""
        return list(self.supported_conversions.get(from_extension, set()))
    
    def can_convert(self, from_extension: str, to_extension: str) -> bool:
        """Check if conversion between two formats is supported"""
        return to_extension in self.supported_conversions.get(from_extension, set())
    
    def scan_directory(self, directory: str, recursive: bool = False) -> Dict[str, List]:
        """
        Scan directory for convertible files
        """
        if not os.path.exists(directory):
            return {"error": f"Directory not found: {directory}"}
        
        convertible_files = []
        other_files = []
        
        if recursive:
            walk_generator = os.walk(directory)
        else:
            # Simulate non-recursive walk
            try:
                items = os.listdir(directory)
                walk_generator = [(directory, [], items)]
            except PermissionError:
                return {"error": f"Permission denied: {directory}"}
        
        for root, dirs, files in walk_generator:
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    file_info = self.detect_file_type(file_path)
                    
                    if file_info.get('convertible_to'):
                        convertible_files.append(file_info)
                    else:
                        other_files.append(file_info)
                        
                except (OSError, PermissionError):
                    # Skip files that can't be accessed
                    continue
        
        return {
            'directory': directory,
            'convertible_files': convertible_files,
            'other_files': other_files,
            'total_convertible': len(convertible_files),
            'total_files': len(convertible_files) + len(other_files)
        }
    
    def get_conversion_suggestions(self, file_path: str) -> List[Dict]:
        """
        Get suggested conversions for a file
        """
        file_info = self.detect_file_type(file_path)
        suggestions = []
        
        for target_format in file_info.get('convertible_to', []):
            suggestion = {
                'from_format': file_info['extension'],
                'to_format': target_format,
                'description': self._get_conversion_description(file_info['extension'], target_format),
                'complexity': self._get_conversion_complexity(file_info['extension'], target_format)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _get_conversion_description(self, from_format: str, to_format: str) -> str:
        """Get human-readable conversion description"""
        descriptions = {
            ('pdf', 'docx'): "Extract text from PDF to editable DOCX",
            ('docx', 'pdf'): "Convert DOCX document to PDF format",
            ('csv', 'xlsx'): "Convert CSV data to Excel format",
            ('jpg', 'png'): "Convert JPEG image to PNG format",
            ('png', 'jpg'): "Convert PNG image to JPEG format"
        }
        
        return descriptions.get((from_format, to_format), f"Convert {from_format.upper()} to {to_format.upper()}")
    
    def _get_conversion_complexity(self, from_format: str, to_format: str) -> str:
        """Estimate conversion complexity"""
        simple_conversions = {('csv', 'xlsx'), ('txt', 'pdf')}
        complex_conversions = {('pdf', 'docx'), ('docx', 'pdf')}
        
        if (from_format, to_format) in simple_conversions:
            return "simple"
        elif (from_format, to_format) in complex_conversions:
            return "complex"
        else:
            return "medium"


# Utility functions
def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_file_category_icon(category: str) -> str:
    """Get emoji icon for file category"""
    icons = {
        'document': '📄',
        'spreadsheet': '📊', 
        'image': '🖼️',
        'presentation': '📽️',
        'data': '📁',
        'other': '📎'
    }
    return icons.get(category, '📎')