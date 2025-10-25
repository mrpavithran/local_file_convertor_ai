"""
CLI module for AI File System operations.
"""

from .main import main
from .command_parser import CommandParser
from .interactive_mode import InteractiveMode

__version__ = "1.0.0"
__all__ = ['main', 'CommandParser', 'InteractiveMode']