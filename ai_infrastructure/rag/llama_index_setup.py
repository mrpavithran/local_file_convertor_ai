"""
LlamaIndex setup and configuration
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LlamaIndexSetup:
    """LlamaIndex configuration and setup"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.initialized = False
        
    def setup(self) -> bool:
        """Initialize LlamaIndex with available models"""
        try:
            # Try to import LlamaIndex
            try:
                import llama_index
                logger.info(f"LlamaIndex version: {llama_index.__version__}")
            except ImportError:
                logger.warning("LlamaIndex not installed. Using basic RAG without LLM.")
                return False
            
            # Check for available LLMs
            self._check_available_models()
            self.initialized = True
            logger.info("✅ LlamaIndex setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ LlamaIndex setup failed: {e}")
            return False
    
    def _check_available_models(self):
        """Check available language models"""
        available_models = []
        
        # Check for OpenAI
        try:
            import openai
            available_models.append("OpenAI GPT models")
        except ImportError:
            pass
            
        # Check for local models via Ollama
        try:
            from ollama import Client
            client = Client()
            models = client.list()
            if models and hasattr(models, 'models'):
                available_models.append("Ollama local models")
        except:
            pass
            
        # Check for Hugging Face
        try:
            import transformers
            available_models.append("Hugging Face models")
        except ImportError:
            pass
            
        logger.info(f"Available models: {', '.join(available_models) if available_models else 'None'}")
    
    def get_llm(self):
        """Get configured LLM instance"""
        if not self.initialized:
            self.setup()
        
        try:
            # Try OpenAI first
            try:
                from llama_index.llms import OpenAI
                return OpenAI(model=self.model_name, temperature=self.temperature)
            except ImportError:
                pass
                
            # Try Ollama
            try:
                from llama_index.llms import Ollama
                return Ollama(model="mistral", temperature=self.temperature)
            except ImportError:
                pass
                
            logger.warning("No LLM backend available. Using basic retrieval only.")
            return None
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None

def setup_llama_index() -> LlamaIndexSetup:
    """Convenience function to setup LlamaIndex"""
    setup = LlamaIndexSetup()
    setup.setup()
    return setup