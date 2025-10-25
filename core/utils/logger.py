"""
Logging Utilities
Configures and manages system logging.
"""

import logging
import logging.handlers
from pathlib import Path
import sys
from typing import Optional
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging(log_dir: str = "data/logs", level: str = "INFO") -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(module)s:%(funcName)s:%(lineno)d]'
    )
    json_formatter = JSONFormatter()
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / 'system_operations.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    
    # JSON handler for structured logging
    json_handler = logging.handlers.RotatingFileHandler(
        log_path / 'structured_operations.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    json_handler.setLevel(level)
    json_handler.setFormatter(json_formatter)
    
    # Error handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / 'conversion_errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Performance handler
    performance_handler = logging.handlers.RotatingFileHandler(
        log_path / 'system_performance.log',
        maxBytes=5*1024*1024,
        backupCount=3
    )
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(json_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(json_handler)
    root_logger.addHandler(error_handler)
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.addHandler(performance_handler)
    perf_logger.propagate = False

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name
        level: Optional specific level for this logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level))
    
    return logger

# Initialize logging when module is imported
setup_logging()