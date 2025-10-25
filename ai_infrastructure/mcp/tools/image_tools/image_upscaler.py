"""
Enhanced Image Upscaling Tool with multiple algorithms and AI-powered super-resolution.
Dependencies: Pillow, numpy, opencv-python (optional), realesrgan (optional)
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageFilter
import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Install with: pip install opencv-python")

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    logging.warning("Real-ESRGAN not available. Install with: pip install realesrgan")

logger = logging.getLogger(__name__)

class ImageUpscaleTool:
    """Enhanced image upscaling tool with multiple algorithms and AI-powered super-resolution."""
    
    name = 'upscale_image'
    description = 'Upscale images using various methods including AI-powered super-resolution'
    
    # Supported upscaling methods
    SUPPORTED_METHODS = {
        'nearest': 'Nearest Neighbor (fastest, low quality)',
        'bilinear': 'Bilinear Interpolation (fast, decent quality)',
        'bicubic': 'Bicubic Interpolation (good balance)',
        'lanczos': 'Lanczos Resampling (high quality)',
        'esrgan': 'Real-ESRGAN (AI-powered, best quality)',
        'waifu2x': 'Waifu2x (AI-powered, for anime/art)',
    }
    
    # Supported scale factors by method
    SCALE_FACTORS = {
        'nearest': [2, 3, 4, 5, 6, 7, 8],
        'bilinear': [2, 3, 4, 5, 6, 7, 8],
        'bicubic': [2, 3, 4, 5, 6, 7, 8],
        'lanczos': [2, 3, 4, 5, 6, 7, 8],
        'esrgan': [2, 4],  # Real-ESRGAN supports specific scales
        'waifu2x': [2, 4, 8],
    }
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        self.esrgan_model = None
        self.waifu2x_model = None
    
    def run(self, 
            input_path: str,
            output_path: Optional[str] = None,
            scale: int = 2,
            method: str = 'lanczos',
            preserve_quality: bool = True,
            enhance_details: bool = False,
            target_format: Optional[str] = None,
            quality: int = 95) -> Dict[str, Any]:
        """
        Upscale image with advanced options and multiple algorithms.
        
        Args:
            input_path: Path to input image
            output_path: Output path for upscaled image (optional)
            scale: Scale factor (2, 3, 4, etc.)
            method: Upscaling method ('nearest', 'bilinear', 'bicubic', 'lanczos', 'esrgan')
            preserve_quality: Whether to preserve image quality during upscaling
            enhance_details: Whether to enhance details after upscaling
            target_format: Target image format (e.g., 'png', 'jpg')
            quality: Output quality for JPEG (1-100)
            
        Returns:
            Upscaling results with metadata
        """
        # Validate input
        if not os.path.exists(input_path):
            return self._error_result(f"Input image not found: {input_path}")
        
        # Validate image format
        file_ext = Path(input_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported image format: {file_ext}")
        
        # Validate method
        if method not in self.SUPPORTED_METHODS:
            return self._error_result(f"Unsupported method: {method}. Supported: {list(self.SUPPORTED_METHODS.keys())}")
        
        # Validate scale factor
        if scale not in self.SCALE_FACTORS.get(method, [2, 3, 4]):
            return self._error_result(f"Scale {scale} not supported for method {method}. Supported: {self.SCALE_FACTORS.get(method, [2, 3, 4])}")
        
        try:
            # Load original image
            original_image = Image.open(input_path)
            original_size = original_image.size
            original_format = original_image.format
            
            # Generate output path if not provided
            if output_path is None:
                output_path = self._generate_output_path(input_path, scale, method, target_format)
            
            # Ensure output directory exists
            os.makedirs(Path(output_path).parent, exist_ok=True)
            
            # Perform upscaling based on method
            if method == 'esrgan' and REALESRGAN_AVAILABLE:
                upscaled_image = self._upscale_esrgan(input_path, output_path, scale)
            elif method == 'waifu2x' and self._is_waifu2x_available():
                upscaled_image = self._upscale_waifu2x(input_path, output_path, scale)
            else:
                upscaled_image = self._upscale_pil(original_image, output_path, scale, method)
            
            # Post-processing
            if enhance_details:
                upscaled_image = self._enhance_details(upscaled_image)
            
            # Get final image info
            if hasattr(upscaled_image, 'size'):
                final_size = upscaled_image.size
            else:
                final_image = Image.open(output_path)
                final_size = final_image.size
                final_image.close()
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(original_image, input_path, output_path)
            
            results = {
                'input_file': os.path.basename(input_path),
                'output_file': os.path.basename(output_path),
                'output_path': os.path.abspath(output_path),
                'method': method,
                'method_description': self.SUPPORTED_METHODS[method],
                'scale_factor': scale,
                'original_size': f"{original_size[0]}x{original_size[1]}",
                'new_size': f"{final_size[0]}x{final_size[1]}",
                'size_increase_percent': ((final_size[0] * final_size[1]) / (original_size[0] * original_size[1]) - 1) * 100,
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"Image upscaled: {original_size} → {final_size} using {method} (scale: {scale}x)")
            return results
            
        except Exception as e:
            logger.error(f"Image upscaling failed: {e}")
            # Clean up failed output file
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            return self._error_result(f"Upscaling failed: {str(e)}")
    
    def _upscale_pil(self, image: Image.Image, output_path: str, scale: int, method: str) -> Image.Image:
        """Upscale image using PIL methods."""
        methods = {
            'nearest': Image.Resampling.NEAREST,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'lanczos': Image.Resampling.LANCZOS,
        }
        
        resample_method = methods.get(method, Image.Resampling.LANCZOS)
        new_size = (image.width * scale, image.height * scale)
        
        # Perform upscaling
        upscaled_image = image.resize(new_size, resample_method)
        
        # Save with quality settings
        save_kwargs = {}
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            save_kwargs['quality'] = 95
            save_kwargs['optimize'] = True
        elif output_path.lower().endswith('.png'):
            save_kwargs['optimize'] = True
        
        upscaled_image.save(output_path, **save_kwargs)
        return upscaled_image
    
    def _upscale_esrgan(self, input_path: str, output_path: str, scale: int) -> Image.Image:
        """Upscale image using Real-ESRGAN."""
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not available. Install with: pip install realesrgan")
        
        try:
            # Initialize ESRGAN model if not already done
            if self.esrgan_model is None:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
                self.esrgan_model = RealESRGANer(
                    scale=scale,
                    model_path=None,  # Will download automatically
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True
                )
            
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read image: {input_path}")
            
            # Handle different image modes
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Upscale
            output, _ = self.esrgan_model.enhance(img, outscale=scale)
            
            # Save result
            cv2.imwrite(output_path, output)
            return Image.open(output_path)
            
        except Exception as e:
            logger.warning(f"ESRGAN upscaling failed, falling back to PIL: {e}")
            # Fallback to PIL
            image = Image.open(input_path)
            return self._upscale_pil(image, output_path, scale, 'lanczos')
    
    def _is_waifu2x_available(self) -> bool:
        """Check if Waifu2x is available."""
        try:
            import torch
            # Waifu2x would require additional setup
            return False  # Currently not implemented
        except ImportError:
            return False
    
    def _upscale_waifu2x(self, input_path: str, output_path: str, scale: int) -> Image.Image:
        """Upscale image using Waifu2x (placeholder implementation)."""
        # This is a placeholder - Waifu2x implementation would go here
        # For now, fallback to ESRGAN or PIL
        logger.warning("Waifu2x not fully implemented, using fallback")
        if REALESRGAN_AVAILABLE:
            return self._upscale_esrgan(input_path, output_path, scale)
        else:
            image = Image.open(input_path)
            return self._upscale_pil(image, output_path, scale, 'lanczos')
    
    def _enhance_details(self, image: Image.Image) -> Image.Image:
        """Enhance image details after upscaling."""
        try:
            # Apply mild sharpening
            enhanced_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # Enhance contrast slightly
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Detail enhancement failed: {e}")
            return image
    
    def _generate_output_path(self, input_path: str, scale: int, method: str, target_format: Optional[str]) -> str:
        """Generate output path with descriptive naming."""
        path_obj = Path(input_path)
        
        if target_format:
            extension = f".{target_format.lower()}"
        else:
            extension = path_obj.suffix
        
        base_name = path_obj.stem
        method_suffix = method if method in ['esrgan', 'waifu2x'] else ''
        
        output_filename = f"{base_name}_x{scale}{method_suffix}{extension}"
        return str(path_obj.parent / output_filename)
    
    def _calculate_quality_metrics(self, original_image: Image.Image, input_path: str, output_path: str) -> Dict[str, Any]:
        """Calculate quality and performance metrics."""
        try:
            original_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            
            original_pixels = original_image.width * original_image.height
            output_image = Image.open(output_path)
            output_pixels = output_image.width * output_image.height
            output_image.close()
            
            return {
                'original_file_size': original_size,
                'output_file_size': output_size,
                'file_size_increase_percent': ((output_size - original_size) / original_size) * 100,
                'pixel_count_increase': output_pixels / original_pixels,
                'compression_ratio': output_size / output_pixels if output_pixels > 0 else 0,
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate quality metrics: {e}")
            return {}
    
    def batch_upscale(self, 
                     image_paths: List[str],
                     scale: int = 2,
                     method: str = 'lanczos',
                     output_dir: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Upscale multiple images in batch.
        
        Args:
            image_paths: List of image paths to upscale
            scale: Scale factor
            method: Upscaling method
            output_dir: Directory for output images
            **kwargs: Additional upscaling parameters
            
        Returns:
            Batch upscaling results
        """
        results = {
            'total_images': len(image_paths),
            'successful_upscales': 0,
            'failed_upscales': 0,
            'total_pixel_increase': 0,
            'total_size_increase': 0,
            'upscale_results': []
        }
        
        pixel_increases = []
        size_increases = []
        
        for image_path in image_paths:
            try:
                # Generate output path
                if output_dir:
                    output_path = Path(output_dir) / self._generate_output_path(
                        image_path, scale, method, kwargs.get('target_format')
                    )
                else:
                    output_path = None
                
                # Perform upscaling
                upscale_result = self.run(
                    input_path=image_path,
                    output_path=str(output_path) if output_path else None,
                    scale=scale,
                    method=method,
                    **kwargs
                )
                
                if upscale_result.get('success'):
                    results['successful_upscales'] += 1
                    pixel_increases.append(upscale_result.get('size_increase_percent', 0))
                    
                    quality_metrics = upscale_result.get('quality_metrics', {})
                    size_increases.append(quality_metrics.get('file_size_increase_percent', 0))
                
                else:
                    results['failed_upscales'] += 1
                
                results['upscale_results'].append({
                    'input_file': image_path,
                    'success': upscale_result.get('success', False),
                    'result': upscale_result if upscale_result.get('success') else {'error': upscale_result.get('error')}
                })
                
            except Exception as e:
                results['failed_upscales'] += 1
                results['upscale_results'].append({
                    'input_file': image_path,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"Batch upscale failed for {image_path}: {e}")
        
        if pixel_increases:
            results['total_pixel_increase'] = sum(pixel_increases) / len(pixel_increases)
        if size_increases:
            results['total_size_increase'] = sum(size_increases) / len(size_increases)
        
        return results
    
    def get_supported_methods(self) -> Dict[str, Any]:
        """Get information about supported upscaling methods."""
        return {
            'supported_methods': self.SUPPORTED_METHODS,
            'scale_factors': self.SCALE_FACTORS,
            'ai_methods_available': {
                'esrgan': REALESRGAN_AVAILABLE,
                'waifu2x': self._is_waifu2x_available()
            }
        }
    
    def compare_methods(self, 
                       input_path: str,
                       scales: List[int] = [2, 4],
                       methods: List[str] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare different upscaling methods on the same image.
        
        Args:
            input_path: Input image path
            scales: Scale factors to test
            methods: Methods to compare
            output_dir: Directory for output images
            
        Returns:
            Method comparison results
        """
        if methods is None:
            methods = ['lanczos', 'bicubic']
            if REALESRGAN_AVAILABLE:
                methods.append('esrgan')
        
        comparison = {
            'input_image': input_path,
            'compared_methods': methods,
            'compared_scales': scales,
            'results': {}
        }
        
        for method in methods:
            comparison['results'][method] = {}
            
            for scale in scales:
                try:
                    # Generate output path
                    if output_dir:
                        output_path = Path(output_dir) / self._generate_output_path(
                            input_path, scale, method, 'png'
                        )
                    else:
                        output_path = None
                    
                    # Perform upscaling
                    result = self.run(
                        input_path=input_path,
                        output_path=str(output_path) if output_path else None,
                        scale=scale,
                        method=method
                    )
                    
                    comparison['results'][method][scale] = result
                    
                except Exception as e:
                    comparison['results'][method][scale] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return comparison
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy image upscale tool (maintains original interface)."""
    
    name = 'upscale_image'
    description = 'Upscale images'
    
    def __init__(self):
        self.enhanced_tool = ImageUpscaleTool()
    
    def run(self, input_path: str, output_path: str = None, scale: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Upscale image with enhanced capabilities.
        
        Args:
            input_path: Path to input image
            output_path: Output path for upscaled image
            scale: Scale factor
            **kwargs: Additional upscaling parameters
            
        Returns:
            Upscaling results
        """
        return self.enhanced_tool.run(
            input_path=input_path,
            output_path=output_path,
            scale=scale,
            **kwargs
        )


# Example usage
if __name__ == "__main__":
    tool = ImageUpscaleTool()
    
    # Test basic upscaling
    result = tool.run(
        input_path="sample.jpg",
        scale=2,
        method='lanczos',
        enhance_details=True
    )
    
    if result['success']:
        print(f"Upscaling successful:")
        print(f"Input: {result['original_size']}")
        print(f"Output: {result['new_size']}")
        print(f"Method: {result['method_description']}")
        print(f"Output path: {result['output_path']}")
    else:
        print(f"Upscaling failed: {result['error']}")
    
    # Show supported methods
    methods = tool.get_supported_methods()
    print(f"\nSupported methods: {len(methods['supported_methods'])}")
    for method, description in methods['supported_methods'].items():
        print(f"  - {method}: {description}")