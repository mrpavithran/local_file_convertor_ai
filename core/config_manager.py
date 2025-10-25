"""
Configuration management for AI File System.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel


class ConfigManager:
    """Manage application configuration with file system operations."""
    
    _instance = None
    _config = None
    _config_paths = [
        Path.home() / '.ai_file_system' / 'config.json',
        Path.home() / '.ai_file_system' / 'config.yaml',
        Path.home() / '.ai_file_system' / 'config.yml',
        Path('config.json'),
        Path('config.yaml'),
        Path('config.yml'),
    ]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, config_path: Optional[Path] = None) -> bool:
        """Initialize configuration manager."""
        console = Console()
        
        try:
            if config_path:
                return cls._load_config_from_path(config_path)
            
            # Try to find and load configuration from standard paths
            for path in cls._config_paths:
                if path.exists():
                    console.print(f"[green]Loading configuration from: {path}[/green]")
                    return cls._load_config_from_path(path)
            
            # Create default configuration
            console.print("[yellow]No configuration file found. Creating default configuration.[/yellow]")
            cls._create_default_config()
            return True
            
        except Exception as e:
            console.print(f"[red]Error initializing configuration: {e}[/red]")
            cls._create_default_config()
            return False
    
    @classmethod
    def _load_config_from_path(cls, config_path: Path) -> bool:
        """Load configuration from a specific path."""
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = json.load(f)
            
            # Ensure we have at least default structure
            cls._ensure_default_structure()
            return True
            
        except Exception as e:
            console = Console()
            console.print(f"[red]Error loading config from {config_path}: {e}[/red]")
            cls._create_default_config()
            return False
    
    @classmethod
    def _create_default_config(cls):
        """Create default configuration."""
        cls._config = {
            'ai': {
                'provider': 'ollama',
                'model': 'llama2',
                'base_url': 'http://localhost:11434',
                'timeout': 30,
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'file_operations': {
                'default_input_dir': './input',
                'default_output_dir': './output',
                'backup_files': True,
                'max_file_size_mb': 50,
                'supported_formats': {
                    'text': ['.txt', '.md', '.docx', '.pdf'],
                    'code': ['.py', '.js', '.java', '.cpp', '.html', '.css'],
                    'images': ['.jpg', '.png', '.gif', '.bmp']
                }
            },
            'conversion': {
                'default_format': 'txt',
                'preserve_structure': True,
                'overwrite_existing': False
            },
            'enhancement': {
                'default_type': 'grammar',
                'create_backups': True,
                'preview_changes': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'ai_file_system.log',
                'max_size_mb': 10
            }
        }
        
        # Save default config
        cls._save_default_config()
    
    @classmethod
    def _save_default_config(cls):
        """Save default configuration to file."""
        config_dir = Path.home() / '.ai_file_system'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / 'config.json'
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(cls._config, f, indent=2, ensure_ascii=False)
            
            console = Console()
            console.print(f"[green]Default configuration saved to: {config_file}[/green]")
        except Exception as e:
            console = Console()
            console.print(f"[yellow]Could not save default config: {e}[/yellow]")
    
    @classmethod
    def _ensure_default_structure(cls):
        """Ensure the configuration has all required sections."""
        default_structure = {
            'ai': {
                'provider': 'ollama',
                'model': 'llama2',
                'base_url': 'http://localhost:11434',
                'timeout': 30
            },
            'file_operations': {
                'default_input_dir': './input',
                'default_output_dir': './output',
                'backup_files': True
            },
            'conversion': {
                'default_format': 'txt',
                'preserve_structure': True
            },
            'enhancement': {
                'default_type': 'grammar',
                'create_backups': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        # Merge with existing config, preserving user settings
        for section, defaults in default_structure.items():
            if section not in cls._config:
                cls._config[section] = defaults
            else:
                for key, value in defaults.items():
                    if key not in cls._config[section]:
                        cls._config[section][key] = value
    
    @classmethod
    def get_config(cls, section: Optional[str] = None, key: Optional[str] = None) -> Any:
        """Get configuration value."""
        if cls._config is None:
            cls.initialize()
        
        if section is None:
            return cls._config
        
        if key is None:
            return cls._config.get(section, {})
        
        return cls._config.get(section, {}).get(key)
    
    @classmethod
    def set_config(cls, section: str, key: str, value: Any) -> bool:
        """Set configuration value."""
        if cls._config is None:
            cls.initialize()
        
        try:
            if section not in cls._config:
                cls._config[section] = {}
            
            cls._config[section][key] = value
            return True
        except Exception:
            return False
    
    @classmethod
    def save_config(cls, config_path: Optional[Path] = None) -> bool:
        """Save configuration to file."""
        if cls._config is None:
            return False
        
        try:
            if config_path is None:
                config_path = Path.home() / '.ai_file_system' / 'config.json'
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(cls._config, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(cls._config, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            console = Console()
            console.print(f"[red]Error saving configuration: {e}[/red]")
            return False
    
    @classmethod
    def show_config(cls):
        """Display current configuration."""
        if cls._config is None:
            cls.initialize()
        
        console = Console()
        
        config_display = json.dumps(cls._config, indent=2, ensure_ascii=False)
        console.print(Panel(config_display, title="Current Configuration", style="blue"))
    
    @classmethod
    def validate_ai_config(cls) -> bool:
        """Validate AI configuration settings."""
        ai_config = cls.get_config('ai')
        
        required_fields = ['provider', 'model', 'base_url']
        for field in required_fields:
            if not ai_config.get(field):
                console = Console()
                console.print(f"[red]Missing required AI configuration: {field}[/red]")
                return False
        
        # Validate URL format
        base_url = ai_config.get('base_url', '')
        if not base_url.startswith(('http://', 'https://')):
            console = Console()
            console.print(f"[red]Invalid AI base URL: {base_url}[/red]")
            return False
        
        return True
    
    @classmethod
    def get_ai_config(cls) -> Dict[str, Any]:
        """Get AI-specific configuration."""
        return cls.get_config('ai')
    
    @classmethod
    def get_file_operations_config(cls) -> Dict[str, Any]:
        """Get file operations configuration."""
        return cls.get_config('file_operations')
    
    @classmethod
    def get_conversion_config(cls) -> Dict[str, Any]:
        """Get conversion configuration."""
        return cls.get_config('conversion')
    
    @classmethod
    def get_enhancement_config(cls) -> Dict[str, Any]:
        """Get enhancement configuration."""
        return cls.get_config('enhancement')