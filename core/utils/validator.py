"""
Validation Utilities
Validates files, data, and system inputs.
"""

import os
import magic
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
import json
import yaml

def validate_file(file_path: str, allowed_extensions: List[str] = None, 
                 max_size_mb: int = 100) -> Dict[str, Any]:
    """
    Validate a file for processing.
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed file extensions
        max_size_mb: Maximum file size in MB
        
    Returns:
        Validation results
    """
    path = Path(file_path)
    
    validation_result = {
        'valid': False,
        'file_path': str(path),
        'exists': False,
        'is_file': False,
        'readable': False,
        'size_bytes': 0,
        'size_mb': 0,
        'extension': '',
        'mime_type': '',
        'errors': []
    }
    
    try:
        # Check existence
        if not path.exists():
            validation_result['errors'].append('File does not exist')
            return validation_result
        
        validation_result['exists'] = True
        
        # Check if it's a file
        if not path.is_file():
            validation_result['errors'].append('Path is not a file')
            return validation_result
        
        validation_result['is_file'] = True
        
        # Check readability
        if not os.access(file_path, os.R_OK):
            validation_result['errors'].append('File is not readable')
            return validation_result
        
        validation_result['readable'] = True
        
        # Get file size
        size = path.stat().st_size
        validation_result['size_bytes'] = size
        validation_result['size_mb'] = size / (1024 * 1024)
        
        # Check size limit
        if validation_result['size_mb'] > max_size_mb:
            validation_result['errors'].append(f'File size {validation_result["size_mb"]:.2f}MB exceeds limit of {max_size_mb}MB')
        
        # Get extension
        extension = path.suffix.lower().lstrip('.')
        validation_result['extension'] = extension
        
        # Check allowed extensions
        if allowed_extensions and extension not in allowed_extensions:
            validation_result['errors'].append(f'File extension .{extension} not in allowed list: {allowed_extensions}')
        
        # Detect MIME type
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(file_path)
            validation_result['mime_type'] = mime_type
        except Exception:
            validation_result['mime_type'] = 'unknown'
        
        # Calculate file hash
        try:
            file_hash = calculate_file_hash(file_path)
            validation_result['file_hash'] = file_hash
        except Exception:
            validation_result['file_hash'] = 'unknown'
        
        # Final validation
        validation_result['valid'] = len(validation_result['errors']) == 0
        
    except Exception as e:
        validation_result['errors'].append(f'Validation error: {str(e)}')
    
    return validation_result

def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate file hash for integrity checking.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        File hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def validate_json_file(file_path: str) -> Dict[str, Any]:
    """
    Validate JSON file structure and syntax.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'data': None
    }
    
    file_validation = validate_file(file_path, ['json'])
    if not file_validation['valid']:
        result['errors'] = file_validation['errors']
        return result
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result['data'] = data
        result['valid'] = True
        
    except json.JSONDecodeError as e:
        result['errors'].append(f'JSON syntax error: {e}')
    except Exception as e:
        result['errors'].append(f'File reading error: {e}')
    
    return result

def validate_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Validate YAML file structure and syntax.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'data': None
    }
    
    file_validation = validate_file(file_path, ['yaml', 'yml'])
    if not file_validation['valid']:
        result['errors'] = file_validation['errors']
        return result
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        result['data'] = data
        result['valid'] = True
        
    except yaml.YAMLError as e:
        result['errors'].append(f'YAML syntax error: {e}')
    except Exception as e:
        result['errors'].append(f'File reading error: {e}')
    
    return result

def validate_directory(directory_path: str, must_exist: bool = True) -> Dict[str, Any]:
    """
    Validate a directory for operations.
    
    Args:
        directory_path: Path to directory
        must_exist: Whether directory must exist
        
    Returns:
        Validation results
    """
    path = Path(directory_path)
    
    result = {
        'valid': False,
        'directory_path': str(path),
        'exists': False,
        'is_directory': False,
        'writable': False,
        'errors': []
    }
    
    try:
        # Check existence
        if not path.exists():
            if must_exist:
                result['errors'].append('Directory does not exist')
            else:
                # Directory doesn't exist but that's OK
                result['valid'] = True
            return result
        
        result['exists'] = True
        
        # Check if it's a directory
        if not path.is_dir():
            result['errors'].append('Path is not a directory')
            return result
        
        result['is_directory'] = True
        
        # Check writability
        if not os.access(directory_path, os.W_OK):
            result['errors'].append('Directory is not writable')
            return result
        
        result['writable'] = True
        result['valid'] = True
        
    except Exception as e:
        result['errors'].append(f'Validation error: {str(e)}')
    
    return result