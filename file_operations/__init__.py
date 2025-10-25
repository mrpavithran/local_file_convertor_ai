"""
File operations module for AI-powered file system operations.
"""

from .file_manager import FileManager
from .type_detector import TypeDetector
from .batch_processor import BatchProcessor
from .output_manager import OutputManager
from .error_handler import ErrorHandler

__version__ = "1.0.0"
__all__ = [
    'FileManager',
    'TypeDetector', 
    'BatchProcessor',
    'OutputManager',
    'ErrorHandler'
]