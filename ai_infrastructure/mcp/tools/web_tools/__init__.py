from .web_scraper import Tool as WebScraperTool
from .url_validator import Tool as URLValidatorTool
from .http_status_checker import Tool as HTTPStatusCheckerTool
from .web_content_analyzer import Tool as WebContentAnalyzer
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class WebToolsManager:
    """Enhanced manager for web analysis tools with workflow capabilities."""
    
    def __init__(self):
        self.tools = self._initialize_tools()
        self._tool_registry = {tool.name: tool for tool in self.tools}
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize all web analysis tools."""
        return [
            WebScraperTool(),
            URLValidatorTool(), 
            HTTPStatusCheckerTool(),
            WebContentAnalyzer()
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
                'description': getattr(tool, 'description', 'No description available'),
                'category': 'web_analysis'
            }
            tool_info.append(info)
        return tool_info
    
    def run_web_analysis_pipeline(self, url: str, pipeline_type: str = 'full') -> Dict[str, Any]:
        """
        Run a complete web analysis pipeline.
        
        Args:
            url: URL to analyze
            pipeline_type: Type of analysis pipeline ('full', 'seo', 'technical', 'quick')
            
        Returns:
            Comprehensive web analysis results
        """
        pipelines = {
            'full': [
                {'tool': 'validate_url', 'description': 'Validate URL syntax and security'},
                {'tool': 'http_status', 'description': 'Check server response and performance'},
                {'tool': 'web_scraper', 'description': 'Extract page content and metadata', 
                 'parameters': {'extract': ['title', 'meta', 'links', 'text']}},
                {'tool': 'analyze_web', 'description': 'Analyze content quality and SEO',
                 'parameters': {'analysis_type': 'full'}}
            ],
            'seo': [
                {'tool': 'validate_url', 'description': 'Validate URL'},
                {'tool': 'web_scraper', 'description': 'Extract SEO elements',
                 'parameters': {'extract': ['title', 'meta', 'headers', 'images']}},
                {'tool': 'analyze_web', 'description': 'SEO analysis',
                 'parameters': {'analysis_type': 'full'}}
            ],
            'technical': [
                {'tool': 'validate_url', 'description': 'Validate URL'},
                {'tool': 'http_status', 'description': 'Check server performance'},
                {'tool': 'web_scraper', 'description': 'Extract technical metadata',
                 'parameters': {'extract': ['meta', 'links']}}
            ],
            'quick': [
                {'tool': 'validate_url', 'description': 'Quick URL validation'},
                {'tool': 'http_status', 'description': 'Basic status check'},
                {'tool': 'analyze_web', 'description': 'Quick content analysis',
                 'parameters': {'analysis_type': 'quick'}}
            ]
        }
        
        selected_pipeline = pipelines.get(pipeline_type, pipelines['full'])
        return self._execute_pipeline(url, selected_pipeline)
    
    def _execute_pipeline(self, url: str, pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a tool pipeline on a URL."""
        results = {
            'url': url,
            'pipeline_start_time': time.time(),
            'steps': [],
            'successful_steps': 0,
            'failed_steps': 0
        }
        
        current_data = {'url': url}
        
        for step_config in pipeline:
            tool_name = step_config['tool']
            tool = self.get_tool(tool_name)
            
            if not tool:
                logger.warning(f"Tool '{tool_name}' not found, skipping step")
                continue
            
            try:
                # Prepare parameters
                params = step_config.get('parameters', {})
                if 'url' in tool.run.__code__.co_varnames:
                    params['url'] = current_data.get('url', url)
                
                # Execute tool
                step_start = time.time()
                step_result = tool.run(**params)
                step_duration = time.time() - step_start
                
                # Store step results
                step_info = {
                    'tool': tool_name,
                    'description': step_config['description'],
                    'duration_seconds': round(step_duration, 2),
                    'success': 'error' not in step_result,
                    'result': step_result
                }
                
                results['steps'].append(step_info)
                
                if step_info['success']:
                    results['successful_steps'] += 1
                    
                    # Pass data to next step if applicable
                    if 'url' in step_result:
                        current_data['url'] = step_result['url']
                    if 'valid' in step_result and not step_result['valid']:
                        logger.info(f"Pipeline stopped due to invalid URL at step {tool_name}")
                        break
                else:
                    results['failed_steps'] += 1
                    logger.warning(f"Step {tool_name} failed: {step_result.get('error', 'Unknown error')}")
                
                # Be polite between requests
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Pipeline step {tool_name} crashed: {e}")
                results['failed_steps'] += 1
                results['steps'].append({
                    'tool': tool_name,
                    'description': step_config['description'],
                    'success': False,
                    'error': str(e)
                })
        
        results['pipeline_duration_seconds'] = round(time.time() - results['pipeline_start_time'], 2)
        results['overall_success'] = results['failed_steps'] == 0
        
        # Generate summary insights
        results['summary'] = self._generate_pipeline_summary(results)
        
        return results
    
    def _generate_pipeline_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from pipeline execution."""
        steps = pipeline_results['steps']
        if not steps:
            return {'insights': ['No steps executed']}
        
        insights = []
        recommendations = []
        
        # Extract key metrics from different tools
        for step in steps:
            if step['success']:
                result = step['result']
                
                # URL validation insights
                if step['tool'] == 'validate_url' and 'valid' in result:
                    if not result['valid']:
                        insights.append("❌ URL validation failed")
                        recommendations.extend(result.get('issues', []))
                    else:
                        insights.append("✅ URL is valid")
                        if result.get('uses_https'):
                            insights.append("✅ Using HTTPS")
                        else:
                            recommendations.append("🔒 Consider switching to HTTPS")
                
                # HTTP status insights
                elif step['tool'] == 'http_status' and 'status_code' in result:
                    status = result['status_code']
                    if 200 <= status < 300:
                        insights.append(f"✅ Server responding normally ({status})")
                    elif 300 <= status < 400:
                        insights.append(f"🔄 Redirects detected ({status})")
                    elif 400 <= status < 500:
                        insights.append(f"❌ Client error ({status})")
                    elif 500 <= status < 600:
                        insights.append(f"🚨 Server error ({status})")
                    
                    if 'response_time_seconds' in result:
                        rt = result['response_time_seconds']
                        if rt > 3:
                            recommendations.append("🐌 Site response time is slow")
                
                # Content analysis insights
                elif step['tool'] == 'analyze_web' and 'overall_score' in result:
                    score = result['overall_score']
                    if score > 0.7:
                        insights.append(f"📊 Good content quality ({score:.1%})")
                    else:
                        insights.append(f"📊 Needs content improvement ({score:.1%})")
                    recommendations.extend(result.get('recommendations', []))
        
        return {
            'total_steps': len(steps),
            'success_rate': f"{pipeline_results['successful_steps']}/{len(steps)}",
            'total_duration': f"{pipeline_results['pipeline_duration_seconds']}s",
            'insights': insights,
            'recommendations': recommendations[:5]  # Top 5 recommendations
        }
    
    def batch_analyze_urls(self, urls: List[str], pipeline_type: str = 'quick') -> Dict[str, Any]:
        """Analyze multiple URLs using the specified pipeline."""
        results = []
        successful = 0
        
        for url in urls:
            try:
                analysis = self.run_web_analysis_pipeline(url, pipeline_type)
                results.append(analysis)
                if analysis['overall_success']:
                    successful += 1
            except Exception as e:
                logger.error(f"Batch analysis failed for {url}: {e}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'overall_success': False
                })
        
        return {
            'results': results,
            'summary': {
                'total_urls': len(urls),
                'successful_analyses': successful,
                'success_rate': f"{(successful/len(urls))*100:.1f}%" if urls else "0%",
                'pipeline_type': pipeline_type
            }
        }
    
    def validate_tools(self) -> Dict[str, Any]:
        """Validate that all web tools are working correctly."""
        test_url = "https://httpbin.org/json"  # Reliable test endpoint
        validation_results = {}
        
        for tool in self.tools:
            tool_name = getattr(tool, 'name', 'unknown')
            try:
                # Test each tool with a simple URL
                result = tool.run(test_url)
                validation_results[tool_name] = {
                    'status': 'healthy',
                    'test_passed': 'error' not in result,
                    'response_time': 'N/A'
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
    return [WebScraperTool(), URLValidatorTool(), HTTPStatusCheckerTool(), WebContentAnalyzer()]


# Additional utility functions
def get_web_tool_names() -> List[str]:
    """Get names of all available web tools."""
    tools = get_all_tools()
    return [tool.name for tool in tools if hasattr(tool, 'name')]


def create_web_workflow(workflow_type: str = 'comprehensive') -> List[Dict[str, Any]]:
    """Create predefined workflows for common web analysis tasks."""
    workflows = {
        'comprehensive': [
            {'tool': 'validate_url', 'description': 'Validate URL syntax and security'},
            {'tool': 'http_status', 'description': 'Check server response'},
            {'tool': 'web_scraper', 'description': 'Extract page content'},
            {'tool': 'analyze_web', 'description': 'Analyze content quality'}
        ],
        'seo_audit': [
            {'tool': 'validate_url', 'description': 'Validate URL'},
            {'tool': 'web_scraper', 'description': 'Extract SEO elements'},
            {'tool': 'analyze_web', 'description': 'SEO analysis'}
        ],
        'health_check': [
            {'tool': 'validate_url', 'description': 'Validate URL'},
            {'tool': 'http_status', 'description': 'Check server status'}
        ]
    }
    return workflows.get(workflow_type, [])