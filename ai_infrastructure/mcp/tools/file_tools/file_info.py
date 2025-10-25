"""
Enhanced File Information Tool
Provides comprehensive file metadata and system information.
"""
import os
import platform
import stat
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class FileInfoTool:
    """Enhanced file information tool with comprehensive metadata extraction."""
    
    name = 'get_file_info'
    description = 'Get comprehensive file information including metadata, permissions, and system details'
    
    def run(self, path: str, include_content_preview: bool = False, max_preview_size: int = 1024) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            path: File or directory path
            include_content_preview: Whether to include file content preview
            max_preview_size: Maximum bytes to read for content preview
            
        Returns:
            Comprehensive file information dictionary
        """
        try:
            if not os.path.exists(path):
                return self._error_result(f"Path not found: {path}")
            
            path_obj = Path(path)
            
            if path_obj.is_dir():
                return self._get_directory_info(path_obj)
            else:
                return self._get_file_info(path_obj, include_content_preview, max_preview_size)
                
        except PermissionError:
            return self._error_result(f"Permission denied: {path}")
        except Exception as e:
            return self._error_result(f"Error accessing path: {str(e)}")
    
    def _get_file_info(self, path_obj: Path, include_content_preview: bool, max_preview_size: int) -> Dict[str, Any]:
        """Get comprehensive file information."""
        try:
            file_stat = path_obj.stat()
            
            # Basic file information
            info = {
                'path': str(path_obj.absolute()),
                'name': path_obj.name,
                'type': 'file',
                'size': file_stat.st_size,
                'size_human': self._format_size(file_stat.st_size),
                
                # Timestamps
                'created': file_stat.st_ctime,
                'created_iso': self._timestamp_to_iso(file_stat.st_ctime),
                'modified': file_stat.st_mtime,
                'modified_iso': self._timestamp_to_iso(file_stat.st_mtime),
                'accessed': file_stat.st_atime,
                'accessed_iso': self._timestamp_to_iso(file_stat.st_atime),
                
                # Permissions and ownership
                'permissions': self._get_permissions_string(file_stat.st_mode),
                'permissions_octal': oct(file_stat.st_mode)[-3:],
                'owner_id': file_stat.st_uid,
                'group_id': file_stat.st_gid,
                
                # File properties
                'extension': path_obj.suffix.lower(),
                'parent_directory': str(path_obj.parent),
                'is_readable': os.access(path_obj, os.R_OK),
                'is_writable': os.access(path_obj, os.W_OK),
                'is_executable': os.access(path_obj, os.X_OK),
                
                # System information
                'filesystem_encoding': sys.getfilesystemencoding(),
            }
            
            # Add MIME type if available
            mime_type = self._detect_mime_type(path_obj)
            if mime_type:
                info['mime_type'] = mime_type
            
            # Add file category
            info['category'] = self._categorize_file(path_obj)
            
            # Add content preview for text files
            if include_content_preview and self._is_text_file(path_obj):
                content_preview = self._get_content_preview(path_obj, max_preview_size)
                if content_preview:
                    info['content_preview'] = content_preview
                    info['content_preview_encoding'] = self._detect_encoding(path_obj)
            
            # Add hash for small files
            if file_stat.st_size < 10 * 1024 * 1024:  # Only for files < 10MB
                info['md5_hash'] = self._calculate_file_hash(path_obj)
            
            # Add additional metadata based on file type
            self._add_file_type_metadata(info, path_obj)
            
            return info
            
        except Exception as e:
            return self._error_result(f"Error getting file info: {str(e)}")
    
    def _get_directory_info(self, path_obj: Path) -> Dict[str, Any]:
        """Get directory information."""
        try:
            dir_stat = path_obj.stat()
            
            # Count files and subdirectories
            try:
                items = list(path_obj.iterdir())
                file_count = len([item for item in items if item.is_file()])
                dir_count = len([item for item in items if item.is_dir()])
                total_size = sum(item.stat().st_size for item in items if item.is_file())
            except (PermissionError, OSError):
                file_count = dir_count = total_size = 0
            
            info = {
                'path': str(path_obj.absolute()),
                'name': path_obj.name,
                'type': 'directory',
                
                # Timestamps
                'created': dir_stat.st_ctime,
                'created_iso': self._timestamp_to_iso(dir_stat.st_ctime),
                'modified': dir_stat.st_mtime,
                'modified_iso': self._timestamp_to_iso(dir_stat.st_mtime),
                'accessed': dir_stat.st_atime,
                'accessed_iso': self._timestamp_to_iso(dir_stat.st_atime),
                
                # Permissions
                'permissions': self._get_permissions_string(dir_stat.st_mode),
                'permissions_octal': oct(dir_stat.st_mode)[-3:],
                'owner_id': dir_stat.st_uid,
                'group_id': dir_stat.st_gid,
                
                # Directory contents
                'file_count': file_count,
                'directory_count': dir_count,
                'total_size': total_size,
                'total_size_human': self._format_size(total_size),
                
                # Access properties
                'is_readable': os.access(path_obj, os.R_OK),
                'is_writable': os.access(path_obj, os.W_OK),
                'is_executable': os.access(path_obj, os.X_OK),
                'is_empty': file_count + dir_count == 0,
                
                'parent_directory': str(path_obj.parent),
            }
            
            return info
            
        except Exception as e:
            return self._error_result(f"Error getting directory info: {str(e)}")
    
    def _get_permissions_string(self, mode: int) -> str:
        """Convert file mode to permission string (e.g., 'rwxr-xr--')."""
        permission_chars = ['r', 'w', 'x']
        result = []
        
        # Convert mode to binary and take last 9 bits
        perm_bits = [(mode >> bit) & 1 for bit in range(8, -1, -1)]
        
        for i, bit in enumerate(perm_bits):
            result.append(permission_chars[i % 3] if bit else '-')
        
        return ''.join(result)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024 and i < len(size_names) - 1:
            size /= 1024
            i += 1
        
        return f"{size:.2f} {size_names[i]}"
    
    def _timestamp_to_iso(self, timestamp: float) -> str:
        """Convert timestamp to ISO format."""
        return datetime.fromtimestamp(timestamp).isoformat()
    
    def _detect_mime_type(self, path_obj: Path) -> Optional[str]:
        """Detect MIME type of file."""
        try:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(path_obj))
            return mime_type
        except:
            return None
    
    def _categorize_file(self, path_obj: Path) -> str:
        """Categorize file by extension and content."""
        extension = path_obj.suffix.lower()
        
        categories = {
            '.txt': 'text', '.md': 'text', '.rst': 'text',
            '.py': 'code', '.js': 'code', '.html': 'code', '.css': 'code',
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image',
            '.pdf': 'document', '.doc': 'document', '.docx': 'document',
            '.xls': 'spreadsheet', '.xlsx': 'spreadsheet', '.csv': 'spreadsheet',
            '.zip': 'archive', '.tar': 'archive', '.gz': 'archive',
            '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio',
            '.mp4': 'video', '.avi': 'video', '.mov': 'video',
        }
        
        return categories.get(extension, 'unknown')
    
    def _is_text_file(self, path_obj: Path) -> bool:
        """Check if file is likely a text file."""
        text_extensions = {'.txt', '.md', '.rst', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv'}
        if path_obj.suffix.lower() in text_extensions:
            return True
        
        # Check first few bytes for text content
        try:
            with open(path_obj, 'rb') as f:
                chunk = f.read(1024)
                # Basic heuristic: if mostly printable ASCII, it's text
                printable = sum(32 <= b < 127 or b in [9, 10, 13] for b in chunk)
                return printable / len(chunk) > 0.8 if chunk else False
        except:
            return False
    
    def _get_content_preview(self, path_obj: Path, max_size: int) -> Optional[str]:
        """Get preview of file content."""
        try:
            with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_size)
                if len(content) == max_size:
                    content += "... (truncated)"
                return content
        except:
            return None
    
    def _detect_encoding(self, path_obj: Path) -> str:
        """Simple encoding detection."""
        try:
            import chardet
            with open(path_obj, 'rb') as f:
                raw_data = f.read(4096)
                result = chardet.detect(raw_data)
                return result.get('encoding', 'unknown')
        except:
            return 'unknown'
    
    def _calculate_file_hash(self, path_obj: Path) -> Optional[str]:
        """Calculate MD5 hash of file."""
        try:
            import hashlib
            hash_md5 = hashlib.md5()
            with open(path_obj, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def _add_file_type_metadata(self, info: Dict[str, Any], path_obj: Path):
        """Add file type specific metadata."""
        extension = path_obj.suffix.lower()
        
        if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            self._add_image_metadata(info, path_obj)
        elif extension == '.pdf':
            self._add_pdf_metadata(info, path_obj)
    
    def _add_image_metadata(self, info: Dict[str, Any], path_obj: Path):
        """Add image-specific metadata."""
        try:
            from PIL import Image
            with Image.open(path_obj) as img:
                info.update({
                    'image_width': img.width,
                    'image_height': img.height,
                    'image_mode': img.mode,
                    'image_format': img.format,
                })
        except ImportError:
            info['image_metadata'] = 'PIL not available for image analysis'
        except Exception as e:
            info['image_metadata_error'] = str(e)
    
    def _add_pdf_metadata(self, info: Dict[str, Any], path_obj: Path):
        """Add PDF-specific metadata."""
        try:
            from PyPDF2 import PdfReader
            with open(path_obj, 'rb') as f:
                reader = PdfReader(f)
                info.update({
                    'pdf_page_count': len(reader.pages),
                    'pdf_encrypted': reader.is_encrypted,
                })
                if reader.metadata:
                    info['pdf_metadata'] = dict(reader.metadata)
        except ImportError:
            info['pdf_metadata'] = 'PyPDF2 not available for PDF analysis'
        except Exception as e:
            info['pdf_metadata_error'] = str(e)
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy file info tool (maintains original interface)."""
    
    name = 'get_file_info'
    description = 'Get comprehensive file information'
    
    def __init__(self):
        self.enhanced_tool = FileInfoTool()
    
    def run(self, path: str, **kwargs) -> Dict[str, Any]:
        """
        Get file information with enhanced capabilities.
        
        Args:
            path: File or directory path
            **kwargs: Additional options (include_content_preview, max_preview_size)
            
        Returns:
            File information dictionary
        """
        return self.enhanced_tool.run(path, **kwargs)


# Utility functions
def get_multiple_file_info(paths: list, **kwargs) -> Dict[str, Any]:
    """Get information for multiple files/directories."""
    tool = FileInfoTool()
    results = {}
    
    for path in paths:
        results[path] = tool.run(path, **kwargs)
    
    return {
        'total_paths': len(paths),
        'results': results
    }


# Example usage
if __name__ == "__main__":
    tool = FileInfoTool()
    
    # Test with a file
    file_info = tool.run(__file__, include_content_preview=True)
    print("File Information:")
    for key, value in file_info.items():
        print(f"  {key}: {value}")
    
    # Test with a directory
    dir_info = tool.run(".", include_content_preview=False)
    print("\nDirectory Information:")
    for key, value in dir_info.items():
        print(f"  {key}: {value}")