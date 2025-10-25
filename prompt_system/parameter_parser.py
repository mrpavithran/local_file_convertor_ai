"""
Parameter Parsing and Validation
Handles parsing of template parameters and input validation.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ParameterParser:
    """Parse and validate parameters for prompt templates."""
    
    def __init__(self):
        self.validators = {
            'file_path': self._validate_file_path,
            'content': self._validate_content,
            'file_type': self._validate_file_type,
            'target_format': self._validate_target_format,
            'target_language': self._validate_language
        }
    
    def parse_parameters(self, template_name: str, raw_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate parameters for a template.
        
        Args:
            template_name: Name of the template
            raw_parameters: Raw parameter dictionary
            
        Returns:
            Parsed and validated parameters
        """
        try:
            # Basic parameter cleaning
            cleaned_params = self._clean_parameters(raw_parameters)
            
            # Type conversion and validation
            validated_params = {}
            for key, value in cleaned_params.items():
                validated_value = self._validate_parameter(key, value)
                if validated_value is not None:
                    validated_params[key] = validated_value
            
            logger.debug(f"Parsed parameters for {template_name}: {list(validated_params.keys())}")
            return validated_params
            
        except Exception as e:
            logger.error(f"Parameter parsing failed for {template_name}: {e}")
            return {}
    
    def _clean_parameters(self, raw_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize parameter values."""
        cleaned = {}
        
        for key, value in raw_parameters.items():
            if value is None:
                continue
                
            if isinstance(value, str):
                # Remove extra whitespace
                cleaned_value = value.strip()
                if cleaned_value:
                    cleaned[key] = cleaned_value
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _validate_parameter(self, key: str, value: Any) -> Any:
        """Validate a single parameter based on its key."""
        validator = self.validators.get(key, self._validate_generic)
        return validator(value)
    
    def _validate_file_path(self, file_path: str) -> Optional[str]:
        """Validate file path parameter."""
        try:
            path = Path(file_path)
            if path.exists():
                return str(path.absolute())
            else:
                logger.warning(f"File path does not exist: {file_path}")
                return file_path  # Still return for reference
        except Exception as e:
            logger.error(f"Invalid file path '{file_path}': {e}")
            return None
    
    def _validate_content(self, content: Any) -> str:
        """Validate and normalize content parameter."""
        if isinstance(content, str):
            # Limit content length for practical reasons
            if len(content) > 100000:  # ~100KB
                logger.warning("Content truncated to 100KB")
                return content[:100000]
            return content
        elif isinstance(content, (dict, list)):
            return json.dumps(content, indent=2)
        else:
            return str(content)
    
    def _validate_file_type(self, file_type: str) -> str:
        """Validate file type parameter."""
        valid_types = {'txt', 'pdf', 'docx', 'json', 'yaml', 'xml', 'html', 'md', 'py', 'js'}
        file_type_clean = file_type.lower().strip('.').split('.')[-1]
        
        if file_type_clean in valid_types:
            return file_type_clean
        else:
            logger.warning(f"Uncommon file type: {file_type}")
            return file_type_clean
    
    def _validate_target_format(self, target_format: str) -> str:
        """Validate target format parameter."""
        valid_formats = {'markdown', 'html', 'json', 'yaml', 'text', 'pdf', 'docx'}
        format_clean = target_format.lower().strip()
        
        if format_clean in valid_formats:
            return format_clean
        else:
            logger.warning(f"Uncommon target format: {target_format}")
            return format_clean
    
    def _validate_language(self, language: str) -> str:
        """Validate language parameter."""
        # Basic language code validation
        if re.match(r'^[a-z]{2,3}(-[A-Z]{2,3})?$', language.strip()):
            return language.strip()
        else:
            logger.warning(f"Invalid language code format: {language}")
            return language.strip()
    
    def _validate_generic(self, value: Any) -> Any:
        """Generic parameter validation."""
        if isinstance(value, str) and not value.strip():
            return None
        return value
    
    def extract_parameters_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract template parameters from a broader context.
        
        Args:
            context: Context dictionary with potential parameters
            
        Returns:
            Extracted parameters
        """
        parameter_keys = {
            'file_path', 'path', 'filename', 'file',
            'content', 'text', 'data',
            'file_type', 'type', 'format',
            'target_format', 'output_format', 'conversion_format',
            'target_language', 'language', 'lang'
        }
        
        extracted = {}
        for key, value in context.items():
            if key in parameter_keys and value is not None:
                # Normalize key names
                normalized_key = self._normalize_parameter_key(key)
                extracted[normalized_key] = value
        
        return extracted
    
    def _normalize_parameter_key(self, key: str) -> str:
        """Normalize parameter key names."""
        key_mapping = {
            'path': 'file_path',
            'filename': 'file_path', 
            'file': 'file_path',
            'text': 'content',
            'data': 'content',
            'type': 'file_type',
            'format': 'file_type',
            'output_format': 'target_format',
            'conversion_format': 'target_format',
            'language': 'target_language',
            'lang': 'target_language'
        }
        
        return key_mapping.get(key, key)