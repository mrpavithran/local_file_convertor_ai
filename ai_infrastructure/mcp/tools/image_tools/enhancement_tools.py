"""
Enhanced Image Enhancement Tool with comprehensive image processing operations.
Dependencies: Pillow, numpy, opencv-python (optional)
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Install with: pip install opencv-python")

logger = logging.getLogger(__name__)

class ImageEnhancerTool:
    """Enhanced image enhancement tool with comprehensive processing operations."""
    
    name = 'enhance_image'
    description = 'Perform comprehensive image enhancements including contrast, sharpness, color correction, and filters'
    
    # Supported enhancement operations
    SUPPORTED_OPERATIONS = {
        'contrast': 'Adjust image contrast',
        'brightness': 'Adjust image brightness',
        'sharpness': 'Enhance image sharpness',
        'color': 'Adjust color saturation',
        'auto_contrast': 'Automatically optimize contrast',
        'auto_color': 'Automatically optimize colors',
        'denoise': 'Reduce image noise',
        'sharpen': 'Apply sharpening filter',
        'blur': 'Apply blur filter',
        'edge_enhance': 'Enhance edges',
        'smooth': 'Smooth image',
        'emboss': 'Apply emboss effect',
        'find_edges': 'Detect and enhance edges',
        'grayscale': 'Convert to grayscale',
        'sepia': 'Apply sepia tone',
        'invert': 'Invert colors',
        'equalize': 'Histogram equalization',
        'auto_enhance': 'Apply multiple automatic enhancements',
    }
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
    
    def run(self, 
            input_path: str,
            operations: List[str] = None,
            output_path: Optional[str] = None,
            intensity: float = 1.0,
            preserve_metadata: bool = True,
            quality: int = 95) -> Dict[str, Any]:
        """
        Perform image enhancements with multiple operations.
        
        Args:
            input_path: Path to input image
            operations: List of enhancement operations to apply
            output_path: Output path for enhanced image
            intensity: Enhancement intensity (0.5 = reduced, 1.0 = normal, 2.0 = enhanced)
            preserve_metadata: Whether to preserve EXIF metadata
            quality: Output quality for JPEG (1-100)
            
        Returns:
            Enhancement results with before/after comparison
        """
        # Validate input
        if not os.path.exists(input_path):
            return self._error_result(f"Input image not found: {input_path}")
        
        # Validate image format
        file_ext = Path(input_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported image format: {file_ext}")
        
        # Set default operations if none provided
        if operations is None:
            operations = ['auto_enhance']
        
        # Validate operations
        invalid_ops = [op for op in operations if op not in self.SUPPORTED_OPERATIONS]
        if invalid_ops:
            return self._error_result(f"Unsupported operations: {invalid_ops}. Supported: {list(self.SUPPORTED_OPERATIONS.keys())}")
        
        try:
            # Load original image
            with Image.open(input_path) as original_img:
                # Preserve original metadata if requested
                original_info = original_img.info if preserve_metadata else {}
                
                # Get original image statistics
                original_stats = self._get_image_statistics(original_img)
                
                # Apply enhancements
                enhanced_img = self._apply_enhancements(original_img, operations, intensity)
                
                # Generate output path if not provided
                if output_path is None:
                    output_path = self._generate_output_path(input_path, operations)
                
                # Ensure output directory exists
                os.makedirs(Path(output_path).parent, exist_ok=True)
                
                # Save enhanced image
                save_kwargs = self._get_save_kwargs(output_path, quality, original_info)
                enhanced_img.save(output_path, **save_kwargs)
                
                # Get enhanced image statistics
                enhanced_stats = self._get_image_statistics(enhanced_img)
                
                # Calculate enhancement metrics
                enhancement_metrics = self._calculate_enhancement_metrics(original_stats, enhanced_stats)
                
                results = {
                    'input_file': os.path.basename(input_path),
                    'output_file': os.path.basename(output_path),
                    'output_path': os.path.abspath(output_path),
                    'operations_applied': operations,
                    'operations_descriptions': [self.SUPPORTED_OPERATIONS[op] for op in operations],
                    'intensity': intensity,
                    'enhancement_metrics': enhancement_metrics,
                    'original_statistics': original_stats,
                    'enhanced_statistics': enhanced_stats,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                logger.info(f"Image enhancement completed: {len(operations)} operations applied")
                return results
                
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            # Clean up failed output file
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            return self._error_result(f"Enhancement failed: {str(e)}")
    
    def _apply_enhancements(self, image: Image.Image, operations: List[str], intensity: float) -> Image.Image:
        """Apply enhancement operations to image."""
        enhanced_img = image.copy()
        
        for operation in operations:
            try:
                if operation == 'auto_enhance':
                    enhanced_img = self._auto_enhance(enhanced_img)
                
                elif operation == 'contrast':
                    enhanced_img = self._enhance_contrast(enhanced_img, intensity)
                
                elif operation == 'brightness':
                    enhanced_img = self._enhance_brightness(enhanced_img, intensity)
                
                elif operation == 'sharpness':
                    enhanced_img = self._enhance_sharpness(enhanced_img, intensity)
                
                elif operation == 'color':
                    enhanced_img = self._enhance_color(enhanced_img, intensity)
                
                elif operation == 'auto_contrast':
                    enhanced_img = ImageOps.autocontrast(enhanced_img)
                
                elif operation == 'auto_color':
                    enhanced_img = ImageOps.colorize(
                        ImageOps.grayscale(enhanced_img), 
                        black='black', 
                        white='white'
                    )
                
                elif operation == 'denoise':
                    enhanced_img = self._denoise_image(enhanced_img)
                
                elif operation == 'sharpen':
                    enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
                
                elif operation == 'blur':
                    enhanced_img = enhanced_img.filter(ImageFilter.BLUR)
                
                elif operation == 'edge_enhance':
                    enhanced_img = enhanced_img.filter(ImageFilter.EDGE_ENHANCE)
                
                elif operation == 'smooth':
                    enhanced_img = enhanced_img.filter(ImageFilter.SMOOTH)
                
                elif operation == 'emboss':
                    enhanced_img = enhanced_img.filter(ImageFilter.EMBOSS)
                
                elif operation == 'find_edges':
                    enhanced_img = enhanced_img.filter(ImageFilter.FIND_EDGES)
                
                elif operation == 'grayscale':
                    enhanced_img = enhanced_img.convert('L')
                
                elif operation == 'sepia':
                    enhanced_img = self._apply_sepia(enhanced_img)
                
                elif operation == 'invert':
                    enhanced_img = ImageOps.invert(enhanced_img)
                
                elif operation == 'equalize':
                    enhanced_img = ImageOps.equalize(enhanced_img)
                
            except Exception as e:
                logger.warning(f"Enhancement operation '{operation}' failed: {e}")
                continue
        
        return enhanced_img
    
    def _auto_enhance(self, image: Image.Image) -> Image.Image:
        """Apply multiple automatic enhancements."""
        enhanced = image.copy()
        
        # Auto contrast
        enhanced = ImageOps.autocontrast(enhanced)
        
        # Mild sharpening
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        # Mild color enhancement for color images
        if enhanced.mode in ('RGB', 'RGBA'):
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.1)
        
        return enhanced
    
    def _enhance_contrast(self, image: Image.Image, intensity: float) -> Image.Image:
        """Enhance image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(intensity)
    
    def _enhance_brightness(self, image: Image.Image, intensity: float) -> Image.Image:
        """Enhance image brightness."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(intensity)
    
    def _enhance_sharpness(self, image: Image.Image, intensity: float) -> Image.Image:
        """Enhance image sharpness."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(intensity)
    
    def _enhance_color(self, image: Image.Image, intensity: float) -> Image.Image:
        """Enhance image color saturation."""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(intensity)
    
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Apply noise reduction to image."""
        if CV2_AVAILABLE:
            # Use OpenCV for better denoising
            img_array = np.array(image)
            
            if len(img_array.shape) == 2:  # Grayscale
                denoised = cv2.medianBlur(img_array, 3)
            else:  # Color
                denoised = cv2.medianBlur(img_array, 3)
            
            return Image.fromarray(denoised)
        else:
            # Fallback to PIL denoising
            return image.filter(ImageFilter.MedianFilter(size=3))
    
    def _apply_sepia(self, image: Image.Image) -> Image.Image:
        """Apply sepia tone effect."""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
        
        # Create sepia matrix
        sepia_matrix = (
            0.393, 0.769, 0.189,
            0.349, 0.686, 0.168,
            0.272, 0.534, 0.131
        )
        
        return rgb_image.convert('RGB', sepia_matrix)
    
    def _get_image_statistics(self, image: Image.Image) -> Dict[str, Any]:
        """Calculate image statistics."""
        # Convert to numpy array for analysis
        if image.mode == 'L':
            img_array = np.array(image)
            mean_brightness = np.mean(img_array)
            contrast = np.std(img_array)
            return {
                'mean_brightness': float(mean_brightness),
                'contrast': float(contrast),
                'min_value': int(np.min(img_array)),
                'max_value': int(np.max(img_array)),
            }
        elif image.mode in ('RGB', 'RGBA'):
            rgb_array = np.array(image.convert('RGB'))
            mean_color = np.mean(rgb_array, axis=(0, 1))
            std_color = np.std(rgb_array, axis=(0, 1))
            return {
                'mean_color_r': float(mean_color[0]),
                'mean_color_g': float(mean_color[1]),
                'mean_color_b': float(mean_color[2]),
                'contrast_r': float(std_color[0]),
                'contrast_g': float(std_color[1]),
                'contrast_b': float(std_color[2]),
                'colorfulness': self._calculate_colorfulness(rgb_array),
            }
        else:
            return {}
    
    def _calculate_colorfulness(self, img_array: np.ndarray) -> float:
        """Calculate colorfulness metric."""
        try:
            # Split into RGB channels
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Calculate rg and yb
            rg = np.abs(r - g)
            yb = np.abs(0.5 * (r + g) - b)
            
            # Compute mean and standard deviation
            rg_mean, rg_std = np.mean(rg), np.std(rg)
            yb_mean, yb_std = np.mean(yb), np.std(yb)
            
            # Combine results
            colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
            
            return round(float(colorfulness), 2)
            
        except Exception as e:
            logger.warning(f"Colorfulness calculation failed: {e}")
            return 0.0
    
    def _calculate_enhancement_metrics(self, original_stats: Dict, enhanced_stats: Dict) -> Dict[str, Any]:
        """Calculate enhancement improvement metrics."""
        metrics = {}
        
        try:
            # Brightness change
            if 'mean_brightness' in original_stats and 'mean_brightness' in enhanced_stats:
                orig_bright = original_stats['mean_brightness']
                enh_bright = enhanced_stats['mean_brightness']
                metrics['brightness_change'] = round(enh_bright - orig_bright, 2)
                metrics['brightness_change_percent'] = round(((enh_bright - orig_bright) / orig_bright) * 100, 2) if orig_bright > 0 else 0
            
            # Contrast change
            if 'contrast' in original_stats and 'contrast' in enhanced_stats:
                orig_contrast = original_stats['contrast']
                enh_contrast = enhanced_stats['contrast']
                metrics['contrast_change'] = round(enh_contrast - orig_contrast, 2)
                metrics['contrast_change_percent'] = round(((enh_contrast - orig_contrast) / orig_contrast) * 100, 2) if orig_contrast > 0 else 0
            
            # Colorfulness change
            if 'colorfulness' in original_stats and 'colorfulness' in enhanced_stats:
                orig_color = original_stats['colorfulness']
                enh_color = enhanced_stats['colorfulness']
                metrics['colorfulness_change'] = round(enh_color - orig_color, 2)
                metrics['colorfulness_change_percent'] = round(((enh_color - orig_color) / orig_color) * 100, 2) if orig_color > 0 else 0
            
            # Overall improvement score
            improvement_score = 0
            if 'contrast_change_percent' in metrics and metrics['contrast_change_percent'] > 0:
                improvement_score += 1
            if 'colorfulness_change_percent' in metrics and metrics['colorfulness_change_percent'] > 0:
                improvement_score += 1
            
            metrics['improvement_score'] = improvement_score
            metrics['improvement_rating'] = self._get_improvement_rating(improvement_score)
            
        except Exception as e:
            logger.warning(f"Enhancement metrics calculation failed: {e}")
        
        return metrics
    
    def _get_improvement_rating(self, score: int) -> str:
        """Get improvement rating based on score."""
        ratings = {
            0: 'minimal',
            1: 'moderate',
            2: 'significant'
        }
        return ratings.get(score, 'minimal')
    
    def _generate_output_path(self, input_path: str, operations: List[str]) -> str:
        """Generate output path with descriptive naming."""
        path_obj = Path(input_path)
        ops_suffix = '_'.join(operations[:3])  # Use first 3 operations in filename
        return str(path_obj.parent / f"{path_obj.stem}_enhanced_{ops_suffix}{path_obj.suffix}")
    
    def _get_save_kwargs(self, output_path: str, quality: int, original_info: dict) -> Dict[str, Any]:
        """Get save parameters based on output format."""
        kwargs = original_info.copy()  # Preserve original metadata
        
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            kwargs.update({
                'quality': quality,
                'optimize': True,
                'progressive': True,
            })
        elif output_path.lower().endswith('.png'):
            kwargs.update({
                'optimize': True,
            })
        
        return kwargs
    
    def batch_enhance(self, 
                     image_paths: List[str],
                     operations: List[str] = None,
                     output_dir: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Enhance multiple images in batch.
        
        Args:
            image_paths: List of image paths to enhance
            operations: Enhancement operations to apply
            output_dir: Directory for output images
            **kwargs: Additional enhancement parameters
            
        Returns:
            Batch enhancement results
        """
        results = {
            'total_images': len(image_paths),
            'successful_enhancements': 0,
            'failed_enhancements': 0,
            'average_improvement_score': 0,
            'enhancement_results': []
        }
        
        improvement_scores = []
        
        for image_path in image_paths:
            try:
                # Generate output path
                if output_dir:
                    output_filename = self._generate_output_path(image_path, operations or ['auto_enhance'])
                    output_path = Path(output_dir) / Path(output_filename).name
                else:
                    output_path = None
                
                # Perform enhancement
                enhance_result = self.run(
                    input_path=image_path,
                    operations=operations,
                    output_path=str(output_path) if output_path else None,
                    **kwargs
                )
                
                if enhance_result.get('success'):
                    results['successful_enhancements'] += 1
                    improvement_scores.append(enhance_result.get('enhancement_metrics', {}).get('improvement_score', 0))
                
                else:
                    results['failed_enhancements'] += 1
                
                results['enhancement_results'].append({
                    'input_file': image_path,
                    'success': enhance_result.get('success', False),
                    'result': enhance_result if enhance_result.get('success') else {'error': enhance_result.get('error')}
                })
                
            except Exception as e:
                results['failed_enhancements'] += 1
                results['enhancement_results'].append({
                    'input_file': image_path,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"Batch enhancement failed for {image_path}: {e}")
        
        if improvement_scores:
            results['average_improvement_score'] = sum(improvement_scores) / len(improvement_scores)
        
        return results
    
    def get_supported_operations(self) -> Dict[str, Any]:
        """Get information about supported enhancement operations."""
        return {
            'supported_operations': self.SUPPORTED_OPERATIONS,
            'total_operations': len(self.SUPPORTED_OPERATIONS),
            'recommended_presets': {
                'basic_enhancement': ['auto_contrast', 'sharpness'],
                'photo_optimization': ['auto_enhance'],
                'noise_reduction': ['denoise', 'smooth'],
                'artistic_effects': ['sepia', 'grayscale', 'emboss'],
                'edge_detection': ['find_edges', 'edge_enhance'],
            }
        }
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy image enhancement tool (maintains original interface)."""
    
    name = 'enhance_image'
    description = 'Enhance images'
    
    def __init__(self):
        self.enhanced_tool = ImageEnhancerTool()
    
    def run(self, input_path: str, operation: str = 'contrast', **kwargs) -> Dict[str, Any]:
        """
        Enhance image with enhanced capabilities.
        
        Args:
            input_path: Path to input image
            operation: Enhancement operation
            **kwargs: Additional enhancement parameters
            
        Returns:
            Enhancement results
        """
        return self.enhanced_tool.run(
            input_path=input_path,
            operations=[operation],
            **kwargs
        )


# Example usage
if __name__ == "__main__":
    tool = ImageEnhancerTool()
    
    # Test basic enhancement
    result = tool.run(
        input_path="sample.jpg",
        operations=['auto_enhance', 'denoise'],
        intensity=1.2
    )
    
    if result['success']:
        print(f"Enhancement successful:")
        print(f"Operations: {', '.join(result['operations_applied'])}")
        print(f"Output: {result['output_path']}")
        print(f"Improvement: {result['enhancement_metrics']['improvement_rating']}")
        print(f"Brightness change: {result['enhancement_metrics'].get('brightness_change', 0)}")
        print(f"Contrast change: {result['enhancement_metrics'].get('contrast_change', 0)}")
    else:
        print(f"Enhancement failed: {result['error']}")
    
    # Show supported operations
    operations = tool.get_supported_operations()
    print(f"\nSupported operations: {len(operations['supported_operations'])}")