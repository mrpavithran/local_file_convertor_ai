"""
Ollama Model Manager
Handles model operations and provides direct access to model lists for the CLI.
"""

import requests
import json
from typing import List, Dict, Any

class ModelManager:
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Ollama models via API"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            else:
                print(f"❌ API error: {response.status_code}")
                return []
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Make sure 'ollama serve' is running.")
            return []
        except Exception as e:
            print(f"❌ Error fetching models: {e}")
            return []
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        models = self.get_available_models()
        return any(model['name'] == model_name for model in models)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {}
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return {
                "status": "running",
                "models_count": len(response.json().get('models', [])),
                "version": "unknown"  # Ollama API doesn't expose version directly
            }
        except requests.exceptions.ConnectionError:
            return {"status": "not_running", "error": "Cannot connect to Ollama"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global instance for easy access
model_manager = ModelManager()