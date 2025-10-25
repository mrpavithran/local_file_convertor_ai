"""
Prompt Execution Tool
Executes prompts using Ollama models with configurable generation settings.
"""

import subprocess
import time
from typing import Dict, Any
import yaml
from pathlib import Path

class Tool:
    name = 'prompt_executor'
    description = 'Execute prompts using Ollama models with configurable settings'
    
    def __init__(self, config_path: str = "model_config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._available = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with sensible defaults."""
        default_config = {
            'default_model': 'mistral',
            'generation_settings': {
                'timeout': 120,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'num_predict': 2048
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
                    # Ensure all default settings are present
                    if 'generation_settings' not in loaded_config:
                        loaded_config['generation_settings'] = {}
                    
                    # Merge with defaults
                    loaded_config['generation_settings'] = {
                        **default_config['generation_settings'],
                        **loaded_config['generation_settings']
                    }
                    return loaded_config
            except Exception as e:
                print(f"Config loading warning: {e}, using defaults")
                return default_config
        
        return default_config
    
    def run(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """
        Execute a prompt using specified Ollama model.
        
        Args:
            prompt: The input prompt to execute
            model: Model to use (defaults to config's default_model)
            **kwargs: Generation parameter overrides (temperature, top_p, timeout, etc.)
            
        Returns:
            Generation results with comprehensive metadata
        """
        # Determine model to use
        if model is None:
            model = self._config.get('default_model', 'mistral')
        
        try:
            start_time = time.time()
            
            # Build command with configuration and overrides
            cmd = self._build_command(model, prompt, kwargs)
            timeout = kwargs.get('timeout', self._config['generation_settings']['timeout'])
            
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            generation_time = time.time() - start_time
            
            if result.returncode == 0:
                response_text = result.stdout.strip()
                return {
                    "success": True,
                    "model": model,
                    "prompt": prompt,
                    "response": response_text,
                    "generation_time_seconds": round(generation_time, 2),
                    "response_length": len(response_text),
                    "word_count": len(response_text.split()),
                    "char_count": len(response_text),
                    "used_settings": self._get_used_settings(kwargs),
                    "config_used": True
                }
            else:
                return {
                    "success": False,
                    "model": model, 
                    "prompt": prompt,
                    "error": result.stderr.strip(),
                    "exit_code": result.returncode,
                    "config_used": True
                }
                
        except subprocess.TimeoutExpired:
            timeout_val = kwargs.get('timeout', self._config['generation_settings']['timeout'])
            return {
                "success": False,
                "error": f"Generation timeout after {timeout_val} seconds",
                "model": model,
                "prompt": prompt,
                "config_used": True
            }
        except FileNotFoundError:
            return {
                "success": False, 
                "error": "Ollama not found in PATH. Please install Ollama.",
                "model": model,
                "prompt": prompt,
                "config_used": True
            }
        except Exception as e:
            return {
                "success": False, 
                "error": f"Unexpected error: {str(e)}",
                "model": model,
                "prompt": prompt,
                "config_used": True
            }
    
    def _build_command(self, model: str, prompt: str, overrides: Dict) -> list:
        """Build Ollama command with configuration settings and overrides."""
        cmd = ["ollama", "run", model]
        
        # Start with config settings, apply overrides
        settings = self._config['generation_settings'].copy()
        settings.update(overrides)
        
        # Add generation parameters to command
        for key in ['temperature', 'top_p', 'top_k', 'num_predict']:
            if key in settings and settings[key] is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(settings[key])])
        
        cmd.append(prompt)
        return cmd
    
    def _get_used_settings(self, overrides: Dict) -> Dict[str, Any]:
        """Get the actual settings used for generation."""
        settings = self._config['generation_settings'].copy()
        settings.update(overrides)
        # Only return the generation parameters that were used
        return {
            k: v for k, v in settings.items() 
            if k in ['temperature', 'top_p', 'top_k', 'num_predict', 'timeout']
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()
    
    def is_available(self) -> bool:
        """Check if Ollama and the default model are available."""
        if self._available is None:
            try:
                # Check if ollama is available
                subprocess.run(["ollama", "--version"], capture_output=True, check=True)
                # Check if default model is available
                default_model = self._config.get('default_model', 'mistral')
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
                self._available = default_model in result.stdout
            except (subprocess.CalledProcessError, FileNotFoundError):
                self._available = False
        return self._available

# Module-level convenience function
def execute_prompt(prompt: str, model: str = "mistral") -> str:
    """
    Simple prompt execution without configuration.
    
    Args:
        prompt: Input prompt
        model: Model to use
        
    Returns:
        Model response as string
    """
    cmd = ["ollama", "run", model, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()