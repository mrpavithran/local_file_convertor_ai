"""
Prompt Templates Management
Defines and manages AI prompt templates for various operations.
"""

import yaml
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TemplateManager:
    """Manage prompt templates with versioning and validation."""
    
    def __init__(self, templates_dir: str = "config/templates"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, Any] = {}
        self.template_versions: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates from YAML files."""
        try:
            if self.templates_dir.exists():
                for template_file in self.templates_dir.glob("*.yaml"):
                    self._load_template_file(template_file)
            
            # Load default templates if no custom templates found
            if not self.templates:
                self._load_default_templates()
                
            logger.info(f"Loaded {len(self.templates)} prompt templates")
            
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            self._load_default_templates()
    
    def _load_template_file(self, file_path: Path) -> None:
        """Load a single template file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
                
            if isinstance(template_data, dict):
                for template_name, template_config in template_data.items():
                    self.templates[template_name] = template_config
                    self.template_versions[template_name] = f"{file_path.stem}_{datetime.now().timestamp()}"
                    
        except Exception as e:
            logger.error(f"Failed to load template file {file_path}: {e}")
    
    def _load_default_templates(self) -> None:
        """Load built-in default templates."""
        default_templates = {
            'file_conversion': {
                'prompt': """Analyze and convert the following file content from {file_path}:

File Type: {file_type}
Content:
{content}

Please convert this to {target_format} format while maintaining all important information and structure.""",
                'description': 'Convert files between different formats',
                'parameters': ['file_path', 'file_type', 'content', 'target_format'],
                'model': 'mistral',
                'settings': {
                    'temperature': 0.3,
                    'max_tokens': 4000
                },
                'version': '1.0'
            },
            'content_enhancement': {
                'prompt': """Enhance and improve the following content from {file_path}:

Original Content:
{content}

Please:
1. Improve clarity and readability
2. Fix any grammatical errors
3. Enhance structure and flow
4. Maintain the original meaning and tone""",
                'description': 'Enhance and improve content quality',
                'parameters': ['file_path', 'content'],
                'model': 'mistral',
                'settings': {
                    'temperature': 0.7,
                    'max_tokens': 3000
                },
                'version': '1.0'
            },
            'text_summarization': {
                'prompt': """Summarize the following content from {file_path}:

Content:
{content}

Please provide a concise summary that captures the key points and main ideas.""",
                'description': 'Create concise summaries of content',
                'parameters': ['file_path', 'content'],
                'model': 'mistral',
                'settings': {
                    'temperature': 0.4,
                    'max_tokens': 1000
                },
                'version': '1.0'
            },
            'code_analysis': {
                'prompt': """Analyze the following code from {file_path}:

Code:
{content}

Please provide:
1. Code explanation
2. Potential issues or improvements
3. Complexity analysis
4. Suggested optimizations""",
                'description': 'Analyze and review code',
                'parameters': ['file_path', 'content'],
                'model': 'codellama',
                'settings': {
                    'temperature': 0.2,
                    'max_tokens': 2000
                },
                'version': '1.0'
            }
        }
        
        self.templates.update(default_templates)
        for template_name in default_templates:
            self.template_versions[template_name] = 'default_1.0'

class PromptTemplates:
    """Main prompt templates interface."""
    
    def __init__(self, templates_dir: str = "config/templates"):
        self.manager = TemplateManager(templates_dir)
        self.templates = self.manager.templates
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific template configuration."""
        return self.templates.get(template_name)
    
    def list_templates(self) -> Dict[str, Any]:
        """List all available templates."""
        return {
            'templates': self.templates,
            'count': len(self.templates),
            'categories': self._get_template_categories()
        }
    
    def _get_template_categories(self) -> List[str]:
        """Extract unique template categories."""
        categories = set()
        for template in self.templates.values():
            if 'category' in template:
                categories.add(template['category'])
        return list(categories)
    
    def format_prompt(self, template_name: str, **kwargs) -> Dict[str, Any]:
        """
        Format a template with provided parameters.
        
        Returns:
            Dict with formatted prompt and metadata
        """
        template = self.get_template(template_name)
        if not template:
            return {
                'success': False,
                'error': f"Template '{template_name}' not found"
            }
        
        # Validate required parameters
        required_params = template.get('parameters', [])
        missing_params = [p for p in required_params if p not in kwargs]
        if missing_params:
            return {
                'success': False,
                'error': f"Missing required parameters: {missing_params}",
                'required': required_params,
                'provided': list(kwargs.keys())
            }
        
        try:
            prompt_text = template['prompt'].format(**kwargs)
            
            return {
                'success': True,
                'prompt': prompt_text,
                'template': template_name,
                'model': template.get('model', 'mistral'),
                'settings': template.get('settings', {}),
                'parameters_used': kwargs
            }
            
        except KeyError as e:
            return {
                'success': False,
                'error': f"Missing parameter in template: {e}",
                'required': required_params
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Template formatting failed: {e}"
            }
    
    def validate_parameters(self, template_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against template requirements."""
        template = self.get_template(template_name)
        if not template:
            return {'valid': False, 'error': f"Template '{template_name}' not found"}
        
        required = set(template.get('parameters', []))
        provided = set(parameters.keys())
        
        missing = required - provided
        extra = provided - required
        
        return {
            'valid': len(missing) == 0,
            'missing_parameters': list(missing),
            'extra_parameters': list(extra),
            'required_parameters': list(required),
            'provided_parameters': list(provided)
        }