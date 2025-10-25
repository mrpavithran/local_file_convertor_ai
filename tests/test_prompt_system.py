"""
Tests for prompt system including templates, parameter parsing, and workflows.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_system.prompt_templates import PromptTemplates
from prompt_system.parameter_parser import ParameterParser
from prompt_system.workflow_orchestrator import WorkflowOrchestrator


class TestPromptTemplates:
    """Test prompt template management."""
    
    @pytest.fixture
    def prompt_templates(self):
        """Create a PromptTemplates instance for testing."""
        return PromptTemplates()
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample prompt configuration for testing."""
        return {
            'prompts': {
                'conversion': {
                    'general_conversion': {
                        'system': 'Convert {source_lang} to {target_lang}',
                        'user': 'Convert this {source_lang} code to {target_lang}:\n\n{code}'
                    }
                },
                'enhancement': {
                    'add_comments': {
                        'system': 'Add comments to {language} code',
                        'user': 'Add comprehensive comments to this {language} code:\n\n{code}'
                    }
                }
            }
        }
    
    def test_template_loading(self, prompt_templates, sample_config):
        """Test loading prompt templates from configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            config_path = f.name
        
        try:
            templates = prompt_templates.load_from_file(config_path)
            assert templates is not None
            assert 'conversion' in templates
            assert 'general_conversion' in templates['conversion']
        finally:
            Path(config_path).unlink()
    
    def test_get_template(self, prompt_templates, sample_config):
        """Test retrieving specific templates."""
        prompt_templates.templates = sample_config['prompts']
        
        template = prompt_templates.get_template('conversion', 'general_conversion')
        assert template is not None
        assert 'system' in template
        assert 'user' in template
        
        # Test non-existent template
        with pytest.raises(ValueError):
            prompt_templates.get_template('nonexistent', 'template')
    
    def test_template_rendering(self, prompt_templates):
        """Test rendering templates with parameters."""
        template = {
            'system': 'Convert from {source} to {target}',
            'user': 'Convert this {source} code:\n\n{code}'
        }
        
        params = {
            'source': 'Python',
            'target': 'JavaScript',
            'code': 'print("Hello World")'
        }
        
        rendered = prompt_templates.render_template(template, params)
        
        assert rendered['system'] == 'Convert from Python to JavaScript'
        assert 'print("Hello World")' in rendered['user']
        assert 'Python' in rendered['user']
    
    def test_template_validation(self, prompt_templates):
        """Test template validation."""
        # Valid template
        valid_template = {
            'system': 'Test system prompt',
            'user': 'Test user prompt {param}'
        }
        assert prompt_templates.validate_template(valid_template) is True
        
        # Invalid template (missing system)
        invalid_template = {
            'user': 'Test user prompt'
        }
        assert prompt_templates.validate_template(invalid_template) is False
        
        # Invalid template (missing user)
        invalid_template2 = {
            'system': 'Test system prompt'
        }
        assert prompt_templates.validate_template(invalid_template2) is False


class TestParameterParser:
    """Test parameter extraction and parsing."""
    
    @pytest.fixture
    def parameter_parser(self):
        """Create a ParameterParser instance for testing."""
        return ParameterParser()
    
    def test_parameter_extraction(self, parameter_parser):
        """Test extracting parameters from text."""
        text = """
        Convert this Python code to JavaScript.
        The code should be efficient and well-commented.
        Input file: main.py
        Output file: main.js
        Language: python to javascript
        """
        
        parameters = parameter_parser.extract_parameters(text)
        
        assert isinstance(parameters, dict)
        assert 'source_lang' in parameters or 'python' in str(parameters).lower()
        assert 'target_lang' in parameters or 'javascript' in str(parameters).lower()
    
    def test_structured_parameter_extraction(self, parameter_parser):
        """Test extracting parameters from structured prompts."""
        prompt = """
        I want to convert a file from Python to JavaScript.
        
        Parameters:
        - source_language: python
        - target_language: javascript  
        - input_file: /path/to/input.py
        - output_file: /path/to/output.js
        - add_comments: true
        """
        
        params = parameter_parser.extract_structured_parameters(prompt)
        
        assert params.get('source_language') == 'python'
        assert params.get('target_language') == 'javascript'
        assert 'input_file' in params
        assert 'output_file' in params
        assert params.get('add_comments') is True
    
    def test_parameter_validation(self, parameter_parser):
        """Test parameter validation."""
        valid_params = {
            'source_lang': 'python',
            'target_lang': 'javascript',
            'input_file': '/path/to/file.py'
        }
        
        required = ['source_lang', 'target_lang', 'input_file']
        assert parameter_parser.validate_parameters(valid_params, required) is True
        
        # Missing required parameter
        invalid_params = {
            'source_lang': 'python',
            'target_lang': 'javascript'
        }
        assert parameter_parser.validate_parameters(invalid_params, required) is False
    
    def test_parameter_transformation(self, parameter_parser):
        """Test parameter transformation and normalization."""
        params = {
            'source_language': 'Python',
            'target_language': 'JavaScript',
            'FILENAME': 'test.py',
            'add-comments': 'true'
        }
        
        transformed = parameter_parser.normalize_parameters(params)
        
        # Check normalization
        assert 'source_language' in transformed
        assert 'target_language' in transformed
        assert 'filename' in transformed or 'input_file' in transformed
        assert transformed.get('add_comments') is True


class TestWorkflowOrchestrator:
    """Test workflow orchestration and execution."""
    
    @pytest.fixture
    def workflow_orchestrator(self):
        """Create a WorkflowOrchestrator instance for testing."""
        return WorkflowOrchestrator()
    
    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow definition."""
        return {
            'name': 'document_conversion',
            'steps': [
                {
                    'name': 'parse_parameters',
                    'action': 'parameter_parsing',
                    'inputs': ['user_input'],
                    'outputs': ['conversion_params']
                },
                {
                    'name': 'load_template', 
                    'action': 'template_loading',
                    'inputs': ['conversion_params'],
                    'outputs': ['prompt_template']
                },
                {
                    'name': 'execute_ai',
                    'action': 'ai_execution',
                    'inputs': ['prompt_template', 'conversion_params'],
                    'outputs': ['ai_response']
                }
            ]
        }
    
    def test_workflow_loading(self, workflow_orchestrator, sample_workflow):
        """Test loading workflow definitions."""
        workflow_orchestrator.load_workflow(sample_workflow)
        
        assert workflow_orchestrator.current_workflow is not None
        assert workflow_orchestrator.current_workflow['name'] == 'document_conversion'
        assert len(workflow_orchestrator.current_workflow['steps']) == 3
    
    def test_workflow_execution(self, workflow_orchestrator, sample_workflow):
        """Test workflow execution."""
        workflow_orchestrator.load_workflow(sample_workflow)
        
        # Mock the execution methods
        with patch.object(workflow_orchestrator, 'execute_step') as mock_execute:
            mock_execute.return_value = {'success': True}
            
            result = workflow_orchestrator.execute_workflow(
                initial_inputs={'user_input': 'Convert Python to JavaScript'}
            )
            
            assert result is not None
            assert mock_execute.called
    
    def test_step_execution(self, workflow_orchestrator):
        """Test individual step execution."""
        step_definition = {
            'name': 'test_step',
            'action': 'parameter_parsing',
            'inputs': ['user_input'],
            'outputs': ['parameters']
        }
        
        inputs = {
            'user_input': 'Convert Python code to JavaScript'
        }
        
        result = workflow_orchestrator.execute_step(step_definition, inputs)
        
        assert result is not None
        assert 'parameters' in result
    
    def test_workflow_validation(self, workflow_orchestrator, sample_workflow):
        """Test workflow validation."""
        assert workflow_orchestrator.validate_workflow(sample_workflow) is True
        
        # Invalid workflow (missing steps)
        invalid_workflow = {
            'name': 'invalid_workflow'
        }
        assert workflow_orchestrator.validate_workflow(invalid_workflow) is False
        
        # Invalid workflow (empty steps)
        invalid_workflow2 = {
            'name': 'invalid_workflow',
            'steps': []
        }
        assert workflow_orchestrator.validate_workflow(invalid_workflow2) is False


class TestIntegration:
    """Integration tests for prompt system components."""
    
    def test_full_prompt_workflow(self):
        """Test complete prompt workflow from input to execution."""
        # This would test the full integration of templates, 
        # parameter parsing, and workflow execution
        pass
    
    def test_error_handling_in_workflows(self):
        """Test error handling in workflow execution."""
        # Test how the system handles errors at various points
        # in the workflow execution
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])