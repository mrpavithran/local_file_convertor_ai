"""
Configuration management for AI File System.
FIXED VERSION - Functional with proper error handling and integration
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manage application configuration with file system operations."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._config_paths = self._get_config_paths()
            self._config = None
            self._initialized = True
            self.initialize()
    
    def _get_config_paths(self):
        """Get list of possible configuration file paths."""
        return [
            Path.home() / '.ai_file_system' / 'config.json',
            Path.home() / '.ai_file_system' / 'config.yaml',
            Path.home() / '.ai_file_system' / 'config.yml',
            Path('config') / 'system_config.json',
            Path('config') / 'system_config.yaml',
            Path('config') / 'system_config.yml',
            Path('config.json'),
            Path('config.yaml'),
            Path('config.yml'),
        ]
    
    def initialize(self, config_path: Optional[Path] = None) -> bool:
        """Initialize configuration manager."""
        try:
            if config_path:
                return self._load_config_from_path(config_path)
            
            # Try to find and load configuration from standard paths
            for path in self._config_paths:
                if path.exists():
                    logger.info(f"Loading configuration from: {path}")
                    return self._load_config_from_path(path)
            
            # Create default configuration
            logger.info("No configuration file found. Creating default configuration.")
            self._create_default_config()
            return True
            
        except Exception as e:
            logger.error(f"Error initializing configuration: {e}")
            self._create_default_config()
            return False
    
    def _load_config_from_path(self, config_path: Path) -> bool:
        """Load configuration from a specific path."""
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            
            # Ensure we have at least default structure
            self._ensure_default_structure()
            logger.info(f"Configuration loaded successfully from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            self._create_default_config()
            return False
    
    def _create_default_config(self):
        """Create default configuration."""
        self._config = {
            'system': {
                'name': 'AI File Converter',
                'version': '1.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            'ai': {
                'provider': 'ollama',
                'model': 'mistral',
                'base_url': 'http://localhost:11434',
                'timeout': 120,
                'temperature': 0.7,
                'max_tokens': 2048,
                'enabled': True
            },
            'file_operations': {
                'default_input_dir': './input',
                'default_output_dir': './output',
                'backup_files': True,
                'max_file_size_mb': 100,
                'supported_formats': {
                    'documents': ['.pdf', '.docx', '.doc', '.txt', '.md'],
                    'spreadsheets': ['.csv', '.xlsx', '.xls'],
                    'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
                }
            },
            'conversion': {
                'default_format': 'txt',
                'preserve_structure': True,
                'overwrite_existing': False,
                'create_backups': True,
                'supported_conversions': {
                    'csv_to_xlsx': True,
                    'docx_to_pdf': True,
                    'pdf_to_docx': True
                }
            },
            'paths': {
                'data_dir': './data',
                'logs_dir': './data/logs',
                'temp_dir': './temp',
                'chroma_db': './data/chroma_db'
            },
            'mcp': {
                'enabled': True,
                'host': 'localhost',
                'port': 8000,
                'tools_enabled': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'ai_file_system.log',
                'max_size_mb': 10,
                'backup_count': 5
            }
        }
        
        # Save default config
        self._save_default_config()
    
    def _save_default_config(self):
        """Save default configuration to file."""
        config_dir = Path.home() / '.ai_file_system'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / 'config.json'
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Default configuration saved to: {config_file}")
        except Exception as e:
            logger.warning(f"Could not save default config: {e}")
    
    def _ensure_default_structure(self):
        """Ensure the configuration has all required sections."""
        default_structure = {
            'system': {
                'name': 'AI File Converter',
                'version': '1.0.0',
                'debug': False
            },
            'ai': {
                'provider': 'ollama',
                'model': 'mistral',
                'base_url': 'http://localhost:11434',
                'timeout': 120,
                'enabled': True
            },
            'file_operations': {
                'default_input_dir': './input',
                'default_output_dir': './output',
                'backup_files': True
            },
            'conversion': {
                'preserve_structure': True,
                'overwrite_existing': False
            },
            'paths': {
                'data_dir': './data',
                'logs_dir': './data/logs'
            },
            'mcp': {
                'enabled': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        # Merge with existing config, preserving user settings
        for section, defaults in default_structure.items():
            if section not in self._config:
                self._config[section] = defaults
            else:
                for key, value in defaults.items():
                    if key not in self._config[section]:
                        self._config[section][key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        if self._config is None:
            self.initialize()
        
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value using dot notation."""
        if self._config is None:
            self.initialize()
        
        try:
            keys = key.split('.')
            config_ptr = self._config
            
            # Navigate to the parent of the final key
            for k in keys[:-1]:
                if k not in config_ptr:
                    config_ptr[k] = {}
                config_ptr = config_ptr[k]
            
            # Set the final key
            config_ptr[keys[-1]] = value
            return True
        except Exception as e:
            logger.error(f"Error setting config key {key}: {e}")
            return False
    
    def get_config(self, section: Optional[str] = None, key: Optional[str] = None) -> Any:
        """Get configuration value (legacy method for compatibility)."""
        if self._config is None:
            self.initialize()
        
        if section is None:
            return self._config
        
        if key is None:
            return self._config.get(section, {})
        
        return self._config.get(section, {}).get(key)
    
    def set_config(self, section: str, key: str, value: Any) -> bool:
        """Set configuration value (legacy method for compatibility)."""
        return self.set(f"{section}.{key}", value)
    
    def save_config(self, config_path: Optional[Path] = None) -> bool:
        """Save configuration to file."""
        if self._config is None:
            return False
        
        try:
            if config_path is None:
                config_path = Path.home() / '.ai_file_system' / 'config.json'
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def show_config(self):
        """Display current configuration."""
        if self._config is None:
            self.initialize()
        
        try:
            # Try to use rich if available, fallback to simple print
            from rich.console import Console
            from rich.panel import Panel
            from rich.json import JSON
            
            console = Console()
            config_json = json.dumps(self._config, indent=2, ensure_ascii=False)
            console.print(Panel(JSON(config_json), title="Current Configuration", style="blue"))
        except ImportError:
            # Fallback to simple print
            print("Current Configuration:")
            print(json.dumps(self._config, indent=2, ensure_ascii=False))
    
    def validate_ai_config(self) -> bool:
        """Validate AI configuration settings."""
        ai_config = self.get_config('ai')
        
        # Check if AI is enabled
        if not ai_config.get('enabled', True):
            logger.info("AI features are disabled in configuration")
            return True
        
        required_fields = ['provider', 'model', 'base_url']
        for field in required_fields:
            if not ai_config.get(field):
                logger.error(f"Missing required AI configuration: {field}")
                return False
        
        # Validate URL format
        base_url = ai_config.get('base_url', '')
        if not base_url.startswith(('http://', 'https://')):
            logger.error(f"Invalid AI base URL: {base_url}")
            return False
        
        return True
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI-specific configuration."""
        return self.get_config('ai')
    
    def get_file_operations_config(self) -> Dict[str, Any]:
        """Get file operations configuration."""
        return self.get_config('file_operations')
    
    def get_conversion_config(self) -> Dict[str, Any]:
        """Get conversion configuration."""
        return self.get_config('conversion')
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.get_config('paths')
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP configuration."""
        return self.get_config('mcp')
    
    def ensure_directories(self) -> bool:
        """Ensure all required directories exist."""
        try:
            paths_config = self.get_paths_config()
            directories = [
                paths_config.get('data_dir', './data'),
                paths_config.get('logs_dir', './data/logs'),
                paths_config.get('temp_dir', './temp'),
                paths_config.get('chroma_db', './data/chroma_db'),
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False
    
    def reload(self) -> bool:
        """Reload configuration from disk."""
        self._config = None
        return self.initialize()


# Global instance for easy access
_config_manager_instance = None

def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager()
    return _config_manager_instance