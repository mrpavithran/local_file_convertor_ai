"""
Enhanced File Type Detection with extension mapping, content analysis, and MIME type detection.
Supports comprehensive file type identification for documents, images, archives, and more.
"""
import os
import mimetypes
import magic
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import struct

# Optional imports with fallbacks
try:
    import filetype
    FILETYPE_AVAILABLE = True
except ImportError:
    FILETYPE_AVAILABLE = False
    logging.warning("filetype library not available. Install with: pip install filetype")

try:
    import magic
    PYTHON_MAGIC_AVAILABLE = True
except ImportError:
    PYTHON_MAGIC_AVAILABLE = False
    logging.warning("python-magic not available. Install with: pip install python-magic")

logger = logging.getLogger(__name__)

class FileTypeDetector:
    """
    Enhanced file type detection with multiple fallback methods:
    1. Extension-based mapping
    2. MIME type detection
    3. Magic number/content analysis
    4. File signature detection
    """
    
    def __init__(self):
        self._init_mime_types()
        self._init_extension_map()
        self._init_magic_signatures()
        
        # Initialize magic library if available
        self.magic_obj = None
        if PYTHON_MAGIC_AVAILABLE:
            try:
                self.magic_obj = magic.Magic(mime=True)
            except Exception as e:
                logger.warning(f"Failed to initialize python-magic: {e}")
    
    def _init_mime_types(self):
        """Initialize MIME type database with additional types."""
        # Add missing MIME types
        mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')
        mimetypes.add_type('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '.xlsx')
        mimetypes.add_type('application/vnd.openxmlformats-officedocument.presentationml.presentation', '.pptx')
        mimetypes.add_type('application/x-zip-compressed', '.zip')
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('text/x-python', '.py')
        mimetypes.add_type('application/json', '.json')
        mimetypes.add_type('text/csv', '.csv')
        mimetypes.add_type('application/xml', '.xml')
        mimetypes.add_type('text/yaml', '.yml')
        mimetypes.add_type('text/yaml', '.yaml')
    
    def _init_extension_map(self):
        """Initialize comprehensive extension to category mapping."""
        self.extension_categories = {
            # Documents
            '.pdf': 'document',
            '.doc': 'document', '.docx': 'document',
            '.txt': 'document', '.rtf': 'document', '.odt': 'document',
            '.md': 'document', '.tex': 'document',
            
            # Spreadsheets
            '.xls': 'spreadsheet', '.xlsx': 'spreadsheet', '.csv': 'spreadsheet',
            '.ods': 'spreadsheet',
            
            # Presentations
            '.ppt': 'presentation', '.pptx': 'presentation', '.odp': 'presentation',
            
            # Images
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image',
            '.bmp': 'image', '.tiff': 'image', '.tif': 'image', '.webp': 'image',
            '.svg': 'image', '.ico': 'image', '.raw': 'image',
            
            # Audio
            '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio', '.aac': 'audio',
            '.ogg': 'audio', '.m4a': 'audio', '.wma': 'audio',
            
            # Video
            '.mp4': 'video', '.avi': 'video', '.mov': 'video', '.wmv': 'video',
            '.flv': 'video', '.webm': 'video', '.mkv': 'video', '.m4v': 'video',
            
            # Archives
            '.zip': 'archive', '.rar': 'archive', '.7z': 'archive', '.tar': 'archive',
            '.gz': 'archive', '.bz2': 'archive',
            
            # Code
            '.py': 'code', '.js': 'code', '.html': 'code', '.css': 'code',
            '.java': 'code', '.cpp': 'code', '.c': 'code', '.h': 'code',
            '.php': 'code', '.rb': 'code', '.go': 'code', '.rs': 'code',
            '.json': 'code', '.xml': 'code', '.yml': 'code', '.yaml': 'code',
            '.sql': 'code',
            
            # Executables
            '.exe': 'executable', '.msi': 'executable', '.dmg': 'executable',
            '.app': 'executable', '.deb': 'executable', '.rpm': 'executable',
            
            # Fonts
            '.ttf': 'font', '.otf': 'font', '.woff': 'font', '.woff2': 'font',
            
            # Database
            '.db': 'database', '.sqlite': 'database', '.mdb': 'database',
            
            # Configuration
            '.ini': 'config', '.cfg': 'config', '.conf': 'config', '.properties': 'config',
        }
        
        # File type descriptions
        self.type_descriptions = {
            'document': 'Text document or word processing file',
            'spreadsheet': 'Spreadsheet or tabular data file',
            'presentation': 'Presentation or slideshow file',
            'image': 'Image or graphic file',
            'audio': 'Audio or sound file',
            'video': 'Video or movie file',
            'archive': 'Compressed or archived file',
            'code': 'Source code or script file',
            'executable': 'Executable program or application',
            'font': 'Font or typeface file',
            'database': 'Database file',
            'config': 'Configuration or settings file',
        }
    
    def _init_magic_signatures(self):
        """Initialize file signature magic numbers."""
        self.magic_signatures = {
            b'%PDF': 'application/pdf',
            b'PK\x03\x04': 'application/zip',  # ZIP, DOCX, XLSX, etc.
            b'\x50\x4B\x03\x04': 'application/zip',
            b'\xD0\xCF\x11\xE0': 'application/msword',  # MS Office older formats
            b'\x25\x50\x44\x46': 'application/pdf',
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'\xFF\xD8\xFF': 'image/jpeg',
            b'GIF8': 'image/gif',
            b'BM': 'image/bmp',
            b'RIFF': 'video/avi',
            b'\x1A\x45\xDF\xA3': 'video/webm',
            b'\x00\x00\x00\x18': 'video/mp4',
            b'ID3': 'audio/mpeg',
            b'OggS': 'audio/ogg',
            b'Rar!\x1A\x07': 'application/x-rar-compressed',
            b'7z\xBC\xAF\x27\x1C': 'application/x-7z-compressed',
            b'\x1F\x8B\x08': 'application/gzip',
            b'BZh': 'application/x-bzip2',
            b'\xFD7zXZ': 'application/x-xz',
        }
    
    def detect_type(self, path: str, use_content_analysis: bool = True) -> Dict[str, Any]:
        """
        Comprehensive file type detection with multiple methods.
        
        Args:
            path: Path to the file to analyze
            use_content_analysis: Whether to analyze file content (slower but more accurate)
            
        Returns:
            Dictionary with comprehensive file type information
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        if os.path.isdir(path):
            return self._detect_directory_type(path)
        
        file_stats = self._get_file_stats(path)
        extension = Path(path).suffix.lower()
        
        # Try multiple detection methods
        detection_methods = [
            self._detect_by_extension,
            self._detect_by_mime,
        ]
        
        if use_content_analysis:
            detection_methods.extend([
                self._detect_by_magic,
                self._detect_by_signature,
                self._detect_by_filetype_lib,
            ])
        
        results = {}
        for method in detection_methods:
            try:
                result = method(path, extension)
                if result and result.get('mime_type'):
                    results = result
                    break
            except Exception as e:
                logger.debug(f"Detection method {method.__name__} failed: {e}")
        
        # Fallback to basic detection
        if not results:
            results = self._fallback_detection(path, extension)
        
        # Combine with file stats
        results.update(file_stats)
        results['detection_method'] = method.__name__ if 'method' in locals() else 'fallback'
        
        return results
    
    def _get_file_stats(self, path: str) -> Dict[str, Any]:
        """Get basic file statistics."""
        stat = os.stat(path)
        return {
            'file_size': stat.st_size,
            'file_size_human': self._format_file_size(stat.st_size),
            'modified_time': stat.st_mtime,
            'is_readable': os.access(path, os.R_OK),
            'is_executable': os.access(path, os.X_OK),
        }
    
    def _detect_by_extension(self, path: str, extension: str) -> Dict[str, Any]:
        """Detect file type by extension."""
        mime_type, encoding = mimetypes.guess_type(path)
        category = self.extension_categories.get(extension, 'unknown')
        
        return {
            'mime_type': mime_type or 'application/octet-stream',
            'category': category,
            'category_description': self.type_descriptions.get(category, 'Unknown file type'),
            'extension': extension,
            'method': 'extension',
            'confidence': 0.7 if mime_type else 0.5,
        }
    
    def _detect_by_mime(self, path: str, extension: str) -> Dict[str, Any]:
        """Detect MIME type using system MIME database."""
        mime_type, encoding = mimetypes.guess_type(path)
        if mime_type:
            category = self._mime_to_category(mime_type)
            return {
                'mime_type': mime_type,
                'category': category,
                'category_description': self.type_descriptions.get(category, 'Unknown file type'),
                'extension': extension,
                'method': 'mime',
                'confidence': 0.8,
            }
        return {}
    
    def _detect_by_magic(self, path: str, extension: str) -> Dict[str, Any]:
        """Detect file type using python-magic."""
        if not self.magic_obj:
            return {}
        
        try:
            mime_type = self.magic_obj.from_file(path)
            category = self._mime_to_category(mime_type)
            
            return {
                'mime_type': mime_type,
                'category': category,
                'category_description': self.type_descriptions.get(category, 'Unknown file type'),
                'extension': extension,
                'method': 'magic',
                'confidence': 0.95,
            }
        except Exception as e:
            logger.debug(f"Magic detection failed: {e}")
            return {}
    
    def _detect_by_signature(self, path: str, extension: str) -> Dict[str, Any]:
        """Detect file type by reading magic number signatures."""
        try:
            with open(path, 'rb') as f:
                header = f.read(16)  # Read first 16 bytes
            
            for signature, mime_type in self.magic_signatures.items():
                if header.startswith(signature):
                    category = self._mime_to_category(mime_type)
                    return {
                        'mime_type': mime_type,
                        'category': category,
                        'category_description': self.type_descriptions.get(category, 'Unknown file type'),
                        'extension': extension,
                        'method': 'signature',
                        'confidence': 0.99,
                    }
        except Exception as e:
            logger.debug(f"Signature detection failed: {e}")
        
        return {}
    
    def _detect_by_filetype_lib(self, path: str, extension: str) -> Dict[str, Any]:
        """Detect file type using filetype library."""
        if not FILETYPE_AVAILABLE:
            return {}
        
        try:
            kind = filetype.guess(path)
            if kind:
                category = self._mime_to_category(kind.mime)
                return {
                    'mime_type': kind.mime,
                    'category': category,
                    'category_description': self.type_descriptions.get(category, 'Unknown file type'),
                    'extension': kind.extension,
                    'method': 'filetype_lib',
                    'confidence': 0.9,
                }
        except Exception as e:
            logger.debug(f"Filetype library detection failed: {e}")
        
        return {}
    
    def _fallback_detection(self, path: str, extension: str) -> Dict[str, Any]:
        """Fallback detection when other methods fail."""
        # Check if it's a text file
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                # Basic text file detection
                if all(b in chunk for b in range(32, 127)) or not any(b > 127 for b in chunk):
                    return {
                        'mime_type': 'text/plain',
                        'category': 'document',
                        'category_description': 'Text document',
                        'extension': extension,
                        'method': 'fallback_text',
                        'confidence': 0.3,
                    }
        except:
            pass
        
        return {
            'mime_type': 'application/octet-stream',
            'category': 'unknown',
            'category_description': 'Unknown file type',
            'extension': extension,
            'method': 'fallback',
            'confidence': 0.1,
        }
    
    def _detect_directory_type(self, path: str) -> Dict[str, Any]:
        """Detect directory type."""
        return {
            'mime_type': 'inode/directory',
            'category': 'directory',
            'category_description': 'Directory or folder',
            'extension': '',
            'method': 'directory',
            'confidence': 1.0,
            'file_count': len(list(Path(path).iterdir())),
        }
    
    def _mime_to_category(self, mime_type: str) -> str:
        """Map MIME type to category."""
        mime_to_category = {
            # Documents
            'application/pdf': 'document',
            'application/msword': 'document',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
            'text/plain': 'document',
            'text/rtf': 'document',
            'application/rtf': 'document',
            'text/markdown': 'document',
            
            # Spreadsheets
            'application/vnd.ms-excel': 'spreadsheet',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'spreadsheet',
            'text/csv': 'spreadsheet',
            
            # Images
            'image/jpeg': 'image',
            'image/png': 'image',
            'image/gif': 'image',
            'image/bmp': 'image',
            'image/tiff': 'image',
            'image/webp': 'image',
            'image/svg+xml': 'image',
            
            # Audio
            'audio/mpeg': 'audio',
            'audio/wav': 'audio',
            'audio/flac': 'audio',
            'audio/aac': 'audio',
            'audio/ogg': 'audio',
            
            # Video
            'video/mp4': 'video',
            'video/avi': 'video',
            'video/quicktime': 'video',
            'video/webm': 'video',
            'video/x-ms-wmv': 'video',
            
            # Archives
            'application/zip': 'archive',
            'application/x-rar-compressed': 'archive',
            'application/x-7z-compressed': 'archive',
            'application/gzip': 'archive',
            'application/x-tar': 'archive',
            
            # Code
            'text/x-python': 'code',
            'application/javascript': 'code',
            'text/html': 'code',
            'text/css': 'code',
            'text/x-java-source': 'code',
            'application/json': 'code',
            'application/xml': 'code',
        }
        
        # Check prefix matches
        for prefix, category in [
            ('text/', 'document'),
            ('image/', 'image'),
            ('audio/', 'audio'),
            ('video/', 'video'),
            ('application/', 'binary'),
        ]:
            if mime_type.startswith(prefix):
                return category
        
        return mime_to_category.get(mime_type, 'unknown')
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def batch_detect(self, directory: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Detect file types for all files in a directory.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            Dictionary with file type statistics
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern = "**/*" if recursive else "*"
        files = [f for f in Path(directory).glob(pattern) if f.is_file()]
        
        results = {
            'total_files': len(files),
            'file_types': {},
            'categories': {},
            'largest_file': None,
            'smallest_file': None,
            'detections': []
        }
        
        max_size = 0
        min_size = float('inf')
        
        for file_path in files:
            try:
                detection = self.detect_type(str(file_path))
                results['detections'].append(detection)
                
                # Update file type statistics
                mime_type = detection['mime_type']
                category = detection['category']
                
                results['file_types'][mime_type] = results['file_types'].get(mime_type, 0) + 1
                results['categories'][category] = results['categories'].get(category, 0) + 1
                
                # Track largest/smallest files
                file_size = detection['file_size']
                if file_size > max_size:
                    max_size = file_size
                    results['largest_file'] = detection
                if file_size < min_size:
                    min_size = file_size
                    results['smallest_file'] = detection
                    
            except Exception as e:
                logger.error(f"Failed to detect type for {file_path}: {e}")
        
        return results


# MCP Tool class
class FileTypeDetectionTool:
    """MCP-compatible file type detection tool."""
    
    name = "detect_file_type"
    description = "Comprehensive file type detection with multiple methods"
    
    def __init__(self):
        self.detector = FileTypeDetector()
    
    def run(self, path: str, use_content_analysis: bool = True) -> Dict[str, Any]:
        """
        Detect file type with comprehensive information.
        
        Args:
            path: Path to file or directory
            use_content_analysis: Whether to analyze file content
            
        Returns:
            File type detection results
        """
        return self.detector.detect_type(path, use_content_analysis)


# Legacy function for backward compatibility
def detect_type(path: str) -> str:
    """Legacy function that returns only MIME type."""
    detector = FileTypeDetector()
    result = detector.detect_type(path, use_content_analysis=False)
    return result.get('mime_type', 'application/octet-stream')


# Command-line interface
def main():
    """Command-line interface for file type detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced File Type Detection')
    parser.add_argument('path', help='File or directory path to analyze')
    parser.add_argument('--batch', action='store_true', help='Batch analyze directory')
    parser.add_argument('--recursive', action='store_true', help='Recursive directory scanning')
    parser.add_argument('--no-content', action='store_true', help='Disable content analysis')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    detector = FileTypeDetector()
    
    if args.batch:
        results = detector.batch_detect(args.path, args.recursive)
        print(f"Batch analysis of {results['total_files']} files:")
        print("\nFile Types:")
        for mime_type, count in sorted(results['file_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {mime_type}: {count} files")
        print("\nCategories:")
        for category, count in sorted(results['categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} files")
    else:
        result = detector.detect_type(args.path, not args.no_content)
        print(f"File: {args.path}")
        print(f"MIME Type: {result['mime_type']}")
        print(f"Category: {result['category']} ({result['category_description']})")
        print(f"Extension: {result['extension']}")
        print(f"Size: {result['file_size_human']}")
        print(f"Detection Method: {result['method']}")
        print(f"Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    main()