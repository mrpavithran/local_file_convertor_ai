"""
Model Management Tool
Handles Ollama model operations: availability checks, model pulling, and listing.
"""

import subprocess
import time
from typing import Dict, Any, List

class Tool:
    name = 'model_manager'
    description = 'Manage Ollama models: check availability, pull models, list models'
    
    def __init__(self):
        self._available = None
    
    def run(self, action: str = "check_availability", model: str = "mistral") -> Dict[str, Any]:
        """
        Perform model management operations.
        
        Args:
            action: Operation to perform - 
                   "check_availability", "pull_model", "list_models", "version"
            model: Model name (for pull operations)
            
        Returns:
            Operation results
        """
        if action == "check_availability":
            return self._check_availability()
        elif action == "pull_model":
            return self._pull_model(model)
        elif action == "list_models":
            return self._list_models()
        elif action == "version":
            return self._get_version()
        else:
            return {"error": f"Unknown action: {action}"}
    
    def _check_availability(self) -> Dict[str, Any]:
        """Check if Ollama is available and Mistral is installed."""
        available = self.is_ollama_available()
        result = {
            "ollama_available": available,
            "mistral_available": False,
            "status": "not_available"
        }
        
        if available:
            try:
                # Check version
                version_result = self._get_version()
                result.update(version_result)
                
                # Check if Mistral is installed
                models_result = self._list_models()
                if models_result.get("available", False):
                    mistral_installed = any(
                        "mistral" in model.get("name", "").lower() 
                        for model in models_result.get("models", [])
                    )
                    result["mistral_available"] = mistral_installed
                    result["status"] = "ready" if mistral_installed else "missing_mistral"
                else:
                    result["status"] = "cannot_list_models"
                    
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
        else:
            result["status"] = "ollama_not_found"
            
        return result
    
    def _pull_model(self, model: str) -> Dict[str, Any]:
        """Pull a model from Ollama registry."""
        if not self.is_ollama_available():
            return {"success": False, "error": "Ollama not available"}
        
        try:
            start_time = time.time()
            
            subprocess.run(["ollama", "pull", model], check=True, capture_output=True)
            
            pull_time = time.time() - start_time
            
            return {
                "success": True,
                "model": model,
                "pull_time_seconds": round(pull_time, 2),
                "message": f"Model '{model}' pulled successfully"
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Failed to pull {model}: {e}",
                "model": model
            }
    
    def _list_models(self) -> Dict[str, Any]:
        """List available Ollama models."""
        if not self.is_ollama_available():
            return {"available": False, "models": []}
        
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            models = []
            
            # Parse model list (skip header line)
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        model_info = {
                            "name": parts[0],
                            "size": parts[1],
                            "modified": ' '.join(parts[2:4]) if len(parts) > 3 else parts[2]
                        }
                        if len(parts) > 4:
                            model_info["digest"] = parts[4]
                        models.append(model_info)
            
            return {
                "available": True,
                "models": models,
                "count": len(models)
            }
        except subprocess.CalledProcessError as e:
            return {
                "available": False,
                "error": f"Failed to list models: {e}",
                "models": []
            }
    
    def _get_version(self) -> Dict[str, Any]:
        """Get Ollama version information."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            return {
                "version_available": True,
                "version": result.stdout.strip()
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"version_available": False}
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama CLI is available in PATH."""
        if self._available is None:
            try:
                subprocess.run(
                    ["ollama", "--version"], 
                    capture_output=True, 
                    check=True
                )
                self._available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self._available = False
        return self._available

# Module-level function for direct access
def check_ollama_availability() -> Dict[str, Any]:
    """Quick check if Ollama is available."""
    tool = Tool()
    return tool._check_availability()