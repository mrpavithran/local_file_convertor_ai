from .text_summarizer import Tool as TextSummarizerTool
from .text_translator import Tool as TextTranslatorTool
from .text_analyzer import Tool as TextAnalyzerTool
from .text_cleaner import Tool as TextCleanerTool
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TextToolsManager:
    """Enhanced manager for text processing tools with additional capabilities."""
    
    def __init__(self):
        self.tools = self._initialize_tools()
        self._tool_registry = {tool.name: tool for tool in self.tools}
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize all text processing tools."""
        return [
            TextSummarizerTool(),
            TextTranslatorTool(), 
            TextAnalyzerTool(),
            TextCleanerTool()
        ]
    
    def get_all_tools(self) -> List[Any]:
        """Get all available tools (maintains your original interface)."""
        return self.tools.copy()
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a specific tool by name."""
        return self._tool_registry.get(tool_name)
    
    def get_tool_info(self) -> List[Dict[str, str]]:
        """Get information about all available tools."""
        tool_info = []
        for tool in self.tools:
            info = {
                'name': getattr(tool, 'name', 'unknown'),
                'description': getattr(tool, 'description', 'No description available')
            }
            tool_info.append(info)
        return tool_info
    
    def run_text_pipeline(self, 
                         text: str,
                         operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run multiple text processing operations in sequence.
        
        Args:
            text: Input text to process
            operations: List of operations with tool names and parameters
            
        Returns:
            Pipeline results with intermediate outputs
        """
        current_text = text
        results = {
            'original_text': text,
            'pipeline_steps': [],
            'final_output': text
        }
        
        for i, operation in enumerate(operations):
            tool_name = operation.get('tool')
            params = operation.get('parameters', {})
            
            tool = self.get_tool(tool_name)
            if not tool:
                logger.warning(f"Tool '{tool_name}' not found, skipping step {i}")
                continue
            
            try:
                # Run the tool
                step_result = tool.run(current_text, **params)
                
                # Store step results
                step_info = {
                    'step': i + 1,
                    'tool': tool_name,
                    'parameters': params,
                    'input_length': len(current_text),
                    'output_length': len(step_result.get('cleaned', step_result.get('summary', step_result.get('translated', '')))),
                    'result': step_result
                }
                results['pipeline_steps'].append(step_info)
                
                # Update current text for next operation
                if 'cleaned' in step_result:
                    current_text = step_result['cleaned']
                elif 'summary' in step_result:
                    current_text = step_result['summary']
                elif 'translated' in step_result:
                    current_text = step_result['translated']
                # For analyzer, we don't modify the text
                
            except Exception as e:
                logger.error(f"Pipeline step {i} failed: {e}")
                results['pipeline_steps'].append({
                    'step': i + 1,
                    'tool': tool_name,
                    'error': str(e),
                    'failed': True
                })
        
        results['final_output'] = current_text
        return results
    
    def suggest_workflow(self, text: str, goal: str) -> List[Dict[str, Any]]:
        """
        Suggest a processing workflow based on text and goal.
        
        Args:
            text: Input text
            goal: Processing goal ('analyze', 'summarize', 'translate', 'clean')
            
        Returns:
            Suggested workflow steps
        """
        workflows = {
            'analyze': [
                {'tool': 'clean_text', 'parameters': {}},
                {'tool': 'analyze_text', 'parameters': {'mode': 'full'}}
            ],
            'summarize': [
                {'tool': 'clean_text', 'parameters': {}},
                {'tool': 'analyze_text', 'parameters': {'mode': 'light'}},
                {'tool': 'summarize_text', 'parameters': {'length': 200, 'strategy': 'key_sentences'}}
            ],
            'translate': [
                {'tool': 'clean_text', 'parameters': {}},
                {'tool': 'analyze_text', 'parameters': {'mode': 'light'}},
                {'tool': 'translate_text', 'parameters': {'target_lang': 'es'}}
            ],
            'clean': [
                {'tool': 'clean_text', 'parameters': {}},
                {'tool': 'analyze_text', 'parameters': {'mode': 'light'}}
            ]
        }
        
        return workflows.get(goal, [])
    
    def validate_tools(self) -> Dict[str, Any]:
        """Validate that all tools are working correctly."""
        validation_results = {}
        
        for tool in self.tools:
            tool_name = getattr(tool, 'name', 'unknown')
            try:
                # Test each tool with sample text
                sample_text = "This is a test sentence for tool validation."
                result = tool.run(sample_text)
                
                validation_results[tool_name] = {
                    'status': 'healthy',
                    'test_passed': True,
                    'response_keys': list(result.keys()) if isinstance(result, dict) else 'non_dict_output'
                }
            except Exception as e:
                validation_results[tool_name] = {
                    'status': 'error',
                    'error': str(e),
                    'test_passed': False
                }
        
        return validation_results


# Maintain your original function for backward compatibility
def get_all_tools():
    """Original function - maintains exact same interface."""
    return [TextSummarizerTool(), TextTranslatorTool(), TextAnalyzerTool(), TextCleanerTool()]


# Additional utility functions
def get_tool_names() -> List[str]:
    """Get names of all available tools."""
    tools = get_all_tools()
    return [tool.name for tool in tools if hasattr(tool, 'name')]


def get_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all available tools."""
    tools = get_all_tools()
    descriptions = {}
    for tool in tools:
        if hasattr(tool, 'name') and hasattr(tool, 'description'):
            descriptions[tool.name] = tool.description
    return descriptions


def create_text_workflow(workflow_type: str = 'analysis') -> List[Dict[str, Any]]:
    """Create predefined workflows for common text processing tasks."""
    workflows = {
        'analysis': [
            {'tool': 'clean_text', 'description': 'Normalize text'},
            {'tool': 'analyze_text', 'description': 'Comprehensive analysis', 'parameters': {'mode': 'full'}}
        ],
        'summarization': [
            {'tool': 'clean_text', 'description': 'Clean input text'},
            {'tool': 'analyze_text', 'description': 'Quick analysis', 'parameters': {'mode': 'light'}},
            {'tool': 'summarize_text', 'description': 'Create summary', 'parameters': {'length': 150}}
        ],
        'translation_prep': [
            {'tool': 'clean_text', 'description': 'Prepare text for translation'},
            {'tool': 'analyze_text', 'description': 'Analyze source text', 'parameters': {'mode': 'light'}}
        ]
    }
    return workflows.get(workflow_type, [])