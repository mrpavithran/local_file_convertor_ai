"""
Ollama Model Management Package
Provides access to model management and prompt execution utilities.
"""

__all__ = ['model_manager', 'prompt_executor', 'config_file']

from . import model_manager
from . import prompt_executor

# Path or filename for the model configuration file
config_file = 'model_config.yaml'
