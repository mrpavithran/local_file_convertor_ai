"""
Configuration Manager
Handles system configuration loading, validation, and management.
"""

import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime
import jsonschema
from jsonschema import validate

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages system configuration with validation and versioning."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.config_schemas: Dict[str, Any] = {}
        self.config_versions: Dict[str, str] = {}
        self._load_config_schemas()
        self._load_all_configs()
    
    def _load_config_schemas(self):
        """Load JSON schemas for configuration validation."""
        self.config_schemas = {
            'system': {
                "type": "object",
                "properties": {
                    "system_name": {"type": "string"},
                    "version": {"type": "string"},
                    "max_concurrent_operations": {"type": "integer", "minimum": 1},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "temp_directory": {"type": "string"},
                    "backup_enabled": {"type": "boolean"}
                },
                "required": ["system_name", "version", "max_concurrent_operations"]
            },
            'paths': {
                "type": "object", 
                "properties": {
                    "input_directory": {"type": "string"},
                    "output_directory": {"type": "string"},
                    "log_directory": {"type": "string"},
                    "temp_directory": {"type": "string"},
                    "backup_directory": {"type": "string"}
                },
                "required": ["input_directory", "output_directory", "log_directory"]
            },
            'tools': {
                "type": "object",
                "properties": {
                    "ollama": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "default_model": {"type": "string"},
                            "timeout": {"type": "integer", "minimum": 1},
                            "max_tokens": {"type": "integer", "minimum": 1}
                        },
                        "required": ["enabled", "default_model"]
                    },
                    "file_operations": {
                        "type": "object",
                        "properties": {
                            "max_file_size_mb": {"type": "integer", "minimum": 1},
                            "allowed_extensions": {"type": "array", "items": {"type": "string"}},
                            "backup_original": {"type": "boolean"}
                        }
                    }
                },
                "required": ["ollama", "file_operations"]
            }
        }
    
    def _load_all_configs(self):
        """Load all configuration files from config directory."""
        config_files = {
            'system': 'system_config.yaml',
            'paths': 'paths_config.yaml', 
            'tools': 'tool_config.yaml',
            'prompts': 'prompt_config.yaml'
        }
        
        for config_name, filename in config_files.items():
            self.load_config(config_name, filename)
    
    def load_config(self, config_name: str, filename: str) -> bool:
        """
        Load a specific configuration file.
        
        Args:
            config_name: Name for the configuration
            filename: Configuration file name
            
        Returns:
            Success status
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Validate configuration
            if config_name in self.config_schemas:
                try:
                    validate(instance=config_data, schema=self.config_schemas[config_name])
                except jsonschema.ValidationError as e:
                    logger.error(f"Config validation failed for {config_name}: {e}")
                    return False
            
            self.configs[config_name] = config_data
            self.config_versions[config_name] = f"{datetime.now().timestamp()}"
            
            logger.info(f"Loaded configuration: {config_name} from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config {config_name} from {filename}: {e}")
            return False
    
    def get_config(self, config_name: str, default: Any = None) -> Any:
        """Get configuration by name."""
        return self.configs.get(config_name, default)
    
    def get_value(self, config_name: str, key: str, default: Any = None) -> Any:
        """Get a specific value from configuration."""
        config = self.get_config(config_name)
        if config is None:
            return default
        
        # Support nested keys with dot notation
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration values.
        
        Args:
            config_name: Configuration to update
            updates: Key-value pairs to update
            
        Returns:
            Success status
        """
        if config_name not in self.configs:
            logger.error(f"Configuration {config_name} not found")
            return False
        
        try:
            # Update configuration
            for key, value in updates.items():
                keys = key.split('.')
                current = self.configs[config_name]
                
                # Navigate to the parent level
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Set the final value
                current[keys[-1]] = value
            
            # Validate updated configuration
            if config_name in self.config_schemas:
                validate(instance=self.configs[config_name], schema=self.config_schemas[config_name])
            
            self.config_versions[config_name] = f"{datetime.now().timestamp()}"
            logger.info(f"Updated configuration: {config_name}")
            return True
            
        except jsonschema.ValidationError as e:
            logger.error(f"Config validation failed after update: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to update config {config_name}: {e}")
            return False
    
    def save_config(self, config_name: str, filename: str = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_name: Configuration to save
            filename: Optional different filename
            
        Returns:
            Success status
        """
        if config_name not in self.configs:
            logger.error(f"Configuration {config_name} not found")
            return False
        
        if filename is None:
            # Determine filename based on config name
            filename = f"{config_name}_config.yaml"
        
        config_path = self.config_dir / filename
        
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.configs[config_name], f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration {config_name} to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config {config_name}: {e}")
            return False
    
    def list_configs(self) -> Dict[str, Any]:
        """List all loaded configurations."""
        return {
            'configurations': list(self.configs.keys()),
            'versions': self.config_versions,
            'schemas_available': list(self.config_schemas.keys())
        }
    
    def validate_config(self, config_name: str) -> Dict[str, Any]:
        """Validate a configuration against its schema."""
        if config_name not in self.configs:
            return {'valid': False, 'error': f"Configuration {config_name} not found"}
        
        if config_name not in self.config_schemas:
            return {'valid': True, 'message': f"No schema defined for {config_name}"}
        
        try:
            validate(instance=self.configs[config_name], schema=self.config_schemas[config_name])
            return {'valid': True, 'message': f"Configuration {config_name} is valid"}
        except jsonschema.ValidationError as e:
            return {'valid': False, 'error': str(e)}