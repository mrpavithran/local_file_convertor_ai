"""
Enhanced utility functions for MCP Server
Provides comprehensive utilities for validation, execution, logging, and more
"""

import asyncio
import inspect
import json
import logging
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from datetime import datetime
from functools import wraps
from contextlib import asynccontextmanager
import hashlib
import re
import os

# Enhanced logging configuration
class CustomFormatter(logging.Formatter):
    """Custom log formatter with colors and detailed formatting"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup enhanced logger with custom formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Global logger instance
logger = setup_logger("mcp_server")

# Validation utilities
def validate_tool_definition(tool_def: Dict[str, Any]) -> bool:
    """
    Validate tool definition schema
    
    Args:
        tool_def: Tool definition dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If tool definition is invalid
    """
    required_fields = ['name', 'description']
    
    for field in required_fields:
        if field not in tool_def:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(tool_def['name'], str) or not tool_def['name'].strip():
        raise ValueError("Tool name must be a non-empty string")
    
    if not isinstance(tool_def['description'], str):
        raise ValueError("Tool description must be a string")
    
    # Validate name format (alphanumeric + underscores)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tool_def['name']):
        raise ValueError("Tool name must contain only alphanumeric characters and underscores")
    
    # Validate parameters schema if provided
    if 'parameters' in tool_def and tool_def['parameters']:
        if not isinstance(tool_def['parameters'], dict):
            raise ValueError("Tool parameters must be a dictionary")
    
    # Validate returns schema if provided
    if 'returns' in tool_def and tool_def['returns']:
        if not isinstance(tool_def['returns'], dict):
            raise ValueError("Tool returns must be a dictionary")
    
    logger.debug(f"Validated tool definition: {tool_def['name']}")
    return True

def validate_parameters(parameters: Dict[str, Any], expected_params: Dict[str, Any]) -> bool:
    """
    Validate parameters against expected schema
    
    Args:
        parameters: Actual parameters
        expected_params: Expected parameter schema
        
    Returns:
        True if parameters are valid
    """
    if not expected_params or not expected_params.get('properties'):
        return True
    
    required_params = expected_params.get('required', [])
    
    # Check required parameters
    for param_name in required_params:
        if param_name not in parameters:
            raise ValueError(f"Missing required parameter: {param_name}")
    
    # Check for unknown parameters
    expected_props = expected_params.get('properties', {})
    for param_name in parameters:
        if param_name not in expected_props:
            logger.warning(f"Unexpected parameter: {param_name}")
    
    return True

# Execution utilities
async def safe_execute_tool(
    func: Callable, 
    parameters: Dict[str, Any], 
    timeout: Optional[int] = None,
    tool_name: str = "unknown"
) -> Any:
    """
    Safely execute a tool function with timeout and error handling
    
    Args:
        func: Tool function to execute
        parameters: Function parameters
        timeout: Execution timeout in seconds
        tool_name: Tool name for logging
        
    Returns:
        Function execution result
    """
    start_time = time.time()
    
    try:
        # Check if function is async
        if inspect.iscoroutinefunction(func):
            # Async execution
            if timeout:
                result = await asyncio.wait_for(func(**parameters), timeout=timeout)
            else:
                result = await func(**parameters)
        else:
            # Sync execution in thread pool
            loop = asyncio.get_event_loop()
            if timeout:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(**parameters)),
                    timeout=timeout
                )
            else:
                result = await loop.run_in_executor(None, lambda: func(**parameters))
        
        execution_time = time.time() - start_time
        logger.debug(f"Tool {tool_name} executed successfully in {execution_time:.3f}s")
        
        return result
        
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        error_msg = f"Tool {tool_name} timed out after {execution_time:.3f}s"
        logger.error(error_msg)
        raise TimeoutError(error_msg)
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Tool {tool_name} failed after {execution_time:.3f}s: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise

def format_tool_result(result: Any) -> Any:
    """
    Format tool result for JSON serialization
    
    Args:
        result: Raw tool result
        
    Returns:
        JSON-serializable result
    """
    if result is None:
        return {"result": None, "type": "null"}
    
    try:
        # Try to serialize to check if it's JSON serializable
        json.dumps(result)
        return result
    except (TypeError, ValueError):
        # Convert non-serializable objects to string
        logger.warning("Tool result is not JSON serializable, converting to string")
        return {
            "result": str(result),
            "type": "string",
            "original_type": type(result).__name__,
            "note": "Result was converted to string for JSON serialization"
        }

def create_error_response(
    error: Exception, 
    context: str = "", 
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error: Exception instance
        context: Error context description
        include_traceback: Whether to include traceback
        
    Returns:
        Standardized error response
    """
    error_id = str(uuid.uuid4())
    error_response = {
        "error_id": error_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
        "context": context
    }
    
    if include_traceback:
        error_response["traceback"] = traceback.format_exc()
    
    return error_response

# Performance monitoring
class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.starts: Dict[str, float] = {}
    
    def start_timer(self, operation: str):
        """Start timer for an operation"""
        self.starts[operation] = time.time()
    
    def stop_timer(self, operation: str) -> float:
        """Stop timer and return duration"""
        if operation not in self.starts:
            raise ValueError(f"No timer started for operation: {operation}")
        
        duration = time.time() - self.starts[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        return duration
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = self.metrics[operation]
        return {
            "count": len(durations),
            "total_time": sum(durations),
            "average_time": sum(durations) / len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "last_time": durations[-1]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        return {op: self.get_stats(op) for op in self.metrics}

# Caching utilities
class SimpleCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value with TTL"""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
    
    def get(self, key: str) -> Any:
        """Get cache value if not expired"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() > entry['expires']:
            del self._cache[key]
            return None
        
        return entry['value']
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
    
    def cleanup_expired(self):
        """Cleanup expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry['expires']
        ]
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)

# Security utilities
def sanitize_input(input_data: Any) -> Any:
    """
    Basic input sanitization to prevent injection attacks
    
    Args:
        input_data: Input data to sanitize
        
    Returns:
        Sanitized input data
    """
    if isinstance(input_data, str):
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&\"\']', '', input_data)
        return sanitized.strip()
    elif isinstance(input_data, dict):
        return {k: sanitize_input(v) for k, v in input_data.items()}
    elif isinstance(input_data, list):
        return [sanitize_input(item) for item in input_data]
    else:
        return input_data

def generate_api_key(length: int = 32) -> str:
    """Generate a random API key"""
    import secrets
    import string
    
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# Rate limiting
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        
        return False
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests"""
        now = time.time()
        
        if identifier not in self.requests:
            return self.max_requests
        
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        return max(0, self.max_requests - len(self.requests[identifier]))

# Context manager for timing
@asynccontextmanager
async def async_timer(operation: str):
    """Context manager for timing async operations"""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.debug(f"{operation} completed in {duration:.3f}s")

# Utility functions for common operations
def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if (key in result and isinstance(result[key], dict) 
            and isinstance(value, dict)):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def get_size(obj: Any) -> int:
    """Get approximate memory size of an object"""
    try:
        return len(json.dumps(obj))
    except (TypeError, ValueError):
        return len(str(obj))

def hash_string(text: str) -> str:
    """Generate SHA-256 hash of a string"""
    return hashlib.sha256(text.encode()).hexdigest()

# File system utilities
def ensure_directory(path: str) -> bool:
    """Ensure directory exists, create if not"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase"""
    return os.path.splitext(file_path)[1].lower()

def is_supported_file(file_path: str, supported_extensions: List[str]) -> bool:
    """Check if file has supported extension"""
    return get_file_extension(file_path) in supported_extensions

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

# Global instances
performance_monitor = PerformanceMonitor()
cache = SimpleCache()
rate_limiter = RateLimiter(max_requests=100, time_window=60)  # 100 requests per minute