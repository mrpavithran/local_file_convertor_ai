"""
Image Tools Package for MCP
Provides comprehensive image processing, analysis, and manipulation tools.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import core image tools
from .ocr_tool import Tool as OCRTool
from .image_upscaler import Tool as ImageUpscaleTool
from .image_info import Tool as ImageInfoTool
from .enhancement_tools import Tool as EnhancementTool

# Optional tools with fallback
try:
    from .image_converter import Tool as ImageConverterTool
    IMAGE_CONVERTER_AVAILABLE = True
except ImportError:
    IMAGE_CONVERTER_AVAILABLE = False
    logger.warning("Image converter tool not available")

try:
    from .face_detector import Tool as FaceDetectorTool
    FACE_DETECTOR_AVAILABLE = True
except ImportError:
    FACE_DETECTOR_AVAILABLE = False
    logger.warning("Face detector tool not available")

try:
    from .background_remover import Tool as BackgroundRemoverTool
    BACKGROUND_REMOVER_AVAILABLE = True
except ImportError:
    BACKGROUND_REMOVER_AVAILABLE = False
    logger.warning("Background remover tool not available")

__all__ = [
    "OCRTool",
    "ImageUpscaleTool", 
    "ImageInfoTool",
    "EnhancementTool",
    "get_all_tools",
    "get_tool_by_name",
    "get_image_tools_statistics",
    "batch_image_processing",
    "analyze_image_collection",
    "create_image_processing_pipeline"
]

class ImageToolsManager:
    """Manager for image tools with enhanced coordination and utilities."""
    
    def __init__(self):
        self.tools = self._discover_tools()
        self.tool_categories = self._categorize_tools()
    
    def _discover_tools(self) -> Dict[str, Any]:
        """Discover and initialize all available image tools."""
        tools = {
            'ocr_image': OCRTool(),
            'upscale_image': ImageUpscaleTool(),
            'get_image_info': ImageInfoTool(),
            'enhance_image': EnhancementTool(),
        }
        
        # Add optional tools if available
        if IMAGE_CONVERTER_AVAILABLE:
            tools['convert_image'] = ImageConverterTool()
        
        if FACE_DETECTOR_AVAILABLE:
            tools['detect_faces'] = FaceDetectorTool()
            
        if BACKGROUND_REMOVER_AVAILABLE:
            tools['remove_background'] = BackgroundRemoverTool()
        
        return tools
    
    def _categorize_tools(self) -> Dict[str, List[str]]:
        """Categorize tools by functionality."""
        return {
            'analysis': [
                'get_image_info',
                'ocr_image',
            ],
            'enhancement': [
                'enhance_image',
                'upscale_image',
            ],
            'conversion': [
                'convert_image' if IMAGE_CONVERTER_AVAILABLE else None,
            ],
            'computer_vision': [
                'detect_faces' if FACE_DETECTOR_AVAILABLE else None,
                'remove_background' if BACKGROUND_REMOVER_AVAILABLE else None,
            ]
        }
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool."""
        if tool_name not in self.tools:
            return None
        
        tool = self.tools[tool_name]
        return {
            'name': tool.name,
            'description': tool.description,
            'category': self._get_tool_category(tool_name),
            'available': True
        }
    
    def _get_tool_category(self, tool_name: str) -> str:
        """Get the category of a tool."""
        for category, tools in self.tool_categories.items():
            if tool_name in tools:
                return category
        return 'uncategorized'

# Global manager instance
_manager = ImageToolsManager()

def get_all_tools() -> List[Any]:
    """
    Get instances of all available image tools.
    
    Returns:
        List of initialized tool instances
    """
    return list(_manager.tools.values())

def get_tool_by_name(name: str) -> Optional[Any]:
    """
    Get a specific image tool by name.
    
    Args:
        name: Tool name (e.g., 'ocr_image', 'enhance_image')
        
    Returns:
        Tool instance or None if not found
    """
    return _manager.tools.get(name)

def get_image_tools_statistics() -> Dict[str, Any]:
    """
    Get statistics and information about available image tools.
    
    Returns:
        Image tools statistics and metadata
    """
    tools = get_all_tools()
    
    return {
        'total_tools': len(tools),
        'available_tools': list(_manager.tools.keys()),
        'tool_categories': _manager.tool_categories,
        'optional_tools': {
            'image_converter': IMAGE_CONVERTER_AVAILABLE,
            'face_detector': FACE_DETECTOR_AVAILABLE,
            'background_remover': BACKGROUND_REMOVER_AVAILABLE
        },
        'tools_info': {
            name: _manager.get_tool_info(name) 
            for name in _manager.tools.keys()
        }
    }

def batch_image_processing(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute multiple image processing operations in batch.
    
    Args:
        operations: List of operation specifications
            Example: [
                {
                    'tool': 'get_image_info',
                    'input_path': '/path/to/image.jpg',
                    'parameters': {'include_exif': True}
                },
                {
                    'tool': 'enhance_image',
                    'input_path': '/path/to/image.jpg', 
                    'parameters': {'operations': ['auto_enhance']}
                }
            ]
        
    Returns:
        Batch processing results
    """
    results = {
        'total_operations': len(operations),
        'successful_operations': 0,
        'failed_operations': 0,
        'operations': []
    }
    
    for i, operation in enumerate(operations):
        try:
            tool_name = operation.get('tool')
            input_path = operation.get('input_path')
            parameters = operation.get('parameters', {})
            
            if not tool_name or not input_path:
                results['operations'].append({
                    'index': i,
                    'status': 'failed',
                    'error': 'Missing tool name or input path'
                })
                results['failed_operations'] += 1
                continue
            
            tool = get_tool_by_name(tool_name)
            if not tool:
                results['operations'].append({
                    'index': i,
                    'status': 'failed', 
                    'error': f'Tool not found: {tool_name}'
                })
                results['failed_operations'] += 1
                continue
            
            # Execute operation
            operation_result = tool.run(input_path=input_path, **parameters)
            
            results['operations'].append({
                'index': i,
                'status': 'success',
                'tool': tool_name,
                'input_path': input_path,
                'result': operation_result
            })
            results['successful_operations'] += 1
            
        except Exception as e:
            results['operations'].append({
                'index': i,
                'status': 'failed',
                'error': str(e),
                'tool': operation.get('tool'),
                'input_path': operation.get('input_path')
            })
            results['failed_operations'] += 1
    
    return results

def analyze_image_collection(images: List[str], analysis_type: str = 'comprehensive') -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a collection of images.
    
    Args:
        images: List of image paths to analyze
        analysis_type: Type of analysis ('basic', 'detailed', 'comprehensive')
        
    Returns:
        Collection analysis results
    """
    from datetime import datetime
    
    analysis = {
        'collection_size': len(images),
        'analysis_type': analysis_type,
        'analysis_timestamp': datetime.now().isoformat(),
        'images_analyzed': 0,
        'analysis_results': {}
    }
    
    # Basic analysis for all images
    info_tool = get_tool_by_name('get_image_info')
    if not info_tool:
        return {'error': 'Image info tool not available'}
    
    image_stats = []
    total_size = 0
    format_distribution = {}
    resolution_stats = []
    
    for image_path in images:
        try:
            # Get basic image information
            info_result = info_tool.run(
                image_path, 
                include_exif=(analysis_type in ['detailed', 'comprehensive']),
                include_color_analysis=(analysis_type == 'comprehensive')
            )
            
            if info_result.get('success'):
                analysis['images_analyzed'] += 1
                image_stats.append(info_result)
                total_size += info_result.get('file_size', 0)
                
                # Update format distribution
                format_name = info_result.get('format', 'unknown')
                format_distribution[format_name] = format_distribution.get(format_name, 0) + 1
                
                # Collect resolution data
                resolution_stats.append({
                    'width': info_result.get('width', 0),
                    'height': info_result.get('height', 0),
                    'megapixels': info_result.get('megapixels', 0)
                })
                
        except Exception as e:
            logger.warning(f"Analysis failed for {image_path}: {e}")
    
    # Calculate collection statistics
    analysis['collection_statistics'] = {
        'total_size_bytes': total_size,
        'total_size_human': _format_size(total_size),
        'average_image_size': total_size / len(images) if images else 0,
        'format_distribution': format_distribution,
        'resolution_summary': _calculate_resolution_summary(resolution_stats),
        'success_rate': analysis['images_analyzed'] / len(images) if images else 0
    }
    
    # Detailed analysis for comprehensive mode
    if analysis_type == 'comprehensive' and analysis['images_analyzed'] > 0:
        analysis['detailed_analysis'] = _perform_detailed_analysis(image_stats)
    
    analysis['analysis_results'] = image_stats
    return analysis

def create_image_processing_pipeline(steps: List[Dict[str, Any]], input_path: str) -> Dict[str, Any]:
    """
    Create and execute an image processing pipeline.
    
    Args:
        steps: List of processing steps
            Example: [
                {
                    'tool': 'enhance_image',
                    'parameters': {'operations': ['auto_contrast', 'sharpness']}
                },
                {
                    'tool': 'upscale_image', 
                    'parameters': {'scale': 2, 'method': 'lanczos'}
                },
                {
                    'tool': 'ocr_image',
                    'parameters': {'language': 'eng'}
                }
            ]
        input_path: Input image path
        
    Returns:
        Pipeline execution results
    """
    from datetime import datetime
    import tempfile
    import shutil
    
    pipeline_results = {
        'pipeline_steps': len(steps),
        'input_image': input_path,
        'start_time': datetime.now().isoformat(),
        'steps_completed': 0,
        'steps_failed': 0,
        'intermediate_files': [],
        'step_results': []
    }
    
    current_file = input_path
    
    try:
        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix='image_pipeline_')
        pipeline_results['temp_directory'] = temp_dir
        
        for i, step in enumerate(steps):
            try:
                tool_name = step.get('tool')
                parameters = step.get('parameters', {})
                
                if not tool_name:
                    pipeline_results['step_results'].append({
                        'step': i,
                        'status': 'failed',
                        'error': 'Missing tool name'
                    })
                    pipeline_results['steps_failed'] += 1
                    continue
                
                tool = get_tool_by_name(tool_name)
                if not tool:
                    pipeline_results['step_results'].append({
                        'step': i,
                        'status': 'failed',
                        'error': f'Tool not found: {tool_name}'
                    })
                    pipeline_results['steps_failed'] += 1
                    continue
                
                # For intermediate steps (not the last one), create output path
                if i < len(steps) - 1:
                    output_path = Path(temp_dir) / f"step_{i}_{Path(current_file).stem}{Path(current_file).suffix}"
                    parameters['output_path'] = str(output_path)
                
                # Execute step
                step_result = tool.run(input_path=current_file, **parameters)
                
                pipeline_results['step_results'].append({
                    'step': i,
                    'tool': tool_name,
                    'status': 'success' if step_result.get('success') else 'failed',
                    'result': step_result
                })
                
                if step_result.get('success'):
                    pipeline_results['steps_completed'] += 1
                    
                    # Update current file for next step
                    if i < len(steps) - 1 and 'output_path' in step_result:
                        current_file = step_result['output_path']
                        pipeline_results['intermediate_files'].append(current_file)
                    elif i == len(steps) - 1:
                        # Last step - this is the final output
                        pipeline_results['final_output'] = step_result.get('output_path', current_file)
                
                else:
                    pipeline_results['steps_failed'] += 1
                    pipeline_results['step_results'][-1]['error'] = step_result.get('error', 'Unknown error')
                    
            except Exception as e:
                pipeline_results['step_results'].append({
                    'step': i,
                    'status': 'failed',
                    'error': str(e)
                })
                pipeline_results['steps_failed'] += 1
        
        pipeline_results['end_time'] = datetime.now().isoformat()
        pipeline_results['success'] = pipeline_results['steps_failed'] == 0
        
        # Clean up temporary directory if pipeline failed
        if not pipeline_results['success']:
            shutil.rmtree(temp_dir, ignore_errors=True)
            pipeline_results['temp_directory_cleaned'] = True
        
        return pipeline_results
        
    except Exception as e:
        # Clean up on overall failure
        if 'temp_directory' in pipeline_results:
            shutil.rmtree(pipeline_results['temp_directory'], ignore_errors=True)
        
        pipeline_results['error'] = f'Pipeline execution failed: {str(e)}'
        pipeline_results['success'] = False
        return pipeline_results

def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024
        i += 1
    
    return f"{size:.2f} {size_names[i]}"

def _calculate_resolution_summary(resolution_stats: List[Dict]) -> Dict[str, Any]:
    """Calculate resolution statistics from image data."""
    if not resolution_stats:
        return {}
    
    widths = [stat['width'] for stat in resolution_stats]
    heights = [stat['height'] for stat in resolution_stats]
    megapixels = [stat['megapixels'] for stat in resolution_stats]
    
    return {
        'average_width': sum(widths) / len(widths),
        'average_height': sum(heights) / len(heights),
        'average_megapixels': sum(megapixels) / len(megapixels),
        'min_resolution': f"{min(widths)}x{min(heights)}",
        'max_resolution': f"{max(widths)}x{max(heights)}",
        'total_images': len(resolution_stats)
    }

def _perform_detailed_analysis(image_stats: List[Dict]) -> Dict[str, Any]:
    """Perform detailed analysis on image collection."""
    detailed = {
        'color_analysis': {},
        'quality_assessment': {},
        'recommendations': []
    }
    
    # Color mode distribution
    color_modes = {}
    for stat in image_stats:
        mode = stat.get('color_mode', 'unknown')
        color_modes[mode] = color_modes.get(mode, 0) + 1
    
    detailed['color_analysis']['mode_distribution'] = color_modes
    
    # Quality assessment
    quality_scores = []
    for stat in image_stats:
        quality = stat.get('quality_metrics', {}).get('quality_assessment', 'unknown')
        if quality != 'unknown':
            quality_scores.append(quality)
    
    if quality_scores:
        quality_dist = {}
        for score in quality_scores:
            quality_dist[score] = quality_dist.get(score, 0) + 1
        detailed['quality_assessment']['distribution'] = quality_dist
    
    # Generate recommendations
    if color_modes.get('L', 0) > len(image_stats) * 0.5:
        detailed['recommendations'].append("Consider colorizing grayscale images for better visual appeal")
    
    return detailed

# Example usage and testing
if __name__ == "__main__":
    # Print available tools and statistics
    stats = get_image_tools_statistics()
    print("Image Tools Package Statistics:")
    print(f"Total tools: {stats['total_tools']}")
    print("Available tools:")
    for tool_name in stats['available_tools']:
        tool_info = stats['tools_info'][tool_name]
        print(f"  - {tool_name}: {tool_info['description']} ({tool_info['category']})")
    
    # Test batch processing
    print("\nTesting batch processing:")
    batch_ops = [
        {
            'tool': 'get_image_info',
            'input_path': __file__.replace('.py', '.jpg') if __file__.endswith('.py') else 'test.jpg',
            'parameters': {'include_exif': True}
        }
    ]
    
    batch_results = batch_image_processing(batch_ops)
    print(f"Batch results: {batch_results['successful_operations']}/{batch_results['total_operations']} successful")
    
    # Test pipeline creation
    print("\nTesting pipeline creation:")
    pipeline_steps = [
        {
            'tool': 'enhance_image',
            'parameters': {'operations': ['auto_contrast']}
        },
        {
            'tool': 'get_image_info',
            'parameters': {'include_color_analysis': True}
        }
    ]
    
    # This would require an actual image file to work
    # pipeline_results = create_image_processing_pipeline(pipeline_steps, "test_image.jpg")
    # print(f"Pipeline completed: {pipeline_results['steps_completed']}/{pipeline_results['pipeline_steps']} steps")