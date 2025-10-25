from PIL import Image
class Tool:
    name='get_image_info'
    description='Get basic image metadata'
    def run(self, input_path):
        im = Image.open(input_path)
        return {'size':im.size, 'mode':im.mode, 'format':im.format}
"""
Enhanced Image Information Tool with comprehensive metadata extraction and analysis.
Dependencies: Pillow, numpy, opencv-python (optional), exifread (optional)
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import math

from PIL import Image, ExifTags, ImageOps
import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Install with: pip install opencv-python")

try:
    import exifread
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False
    logging.warning("exifread not available. Install with: pip install exifread")

logger = logging.getLogger(__name__)

class ImageInfoTool:
    """Enhanced image information tool with comprehensive metadata extraction and analysis."""
    
    name = 'get_image_info'
    description = 'Get comprehensive image metadata including EXIF data, color analysis, and quality metrics'
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.gif', '.ico', '.svg'}
    
    def run(self, 
            input_path: str,
            include_exif: bool = True,
            include_color_analysis: bool = True,
            include_quality_metrics: bool = True,
            include_histogram: bool = False,
            max_histogram_bins: int = 256) -> Dict[str, Any]:
        """
        Get comprehensive image information and metadata.
        
        Args:
            input_path: Path to input image
            include_exif: Whether to include EXIF metadata
            include_color_analysis: Whether to perform color analysis
            include_quality_metrics: Whether to calculate quality metrics
            include_histogram: Whether to include color histogram data
            max_histogram_bins: Maximum number of histogram bins
            
        Returns:
            Comprehensive image information dictionary
        """
        # Validate input
        if not os.path.exists(input_path):
            return self._error_result(f"Input image not found: {input_path}")
        
        # Validate image format
        file_ext = Path(input_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported image format: {file_ext}")
        
        try:
            # Load image
            with Image.open(input_path) as img:
                # Get basic information
                basic_info = self._get_basic_info(img, input_path)
                
                # Get EXIF metadata if requested
                exif_data = {}
                if include_exif:
                    exif_data = self._get_exif_metadata(img, input_path)
                
                # Perform color analysis if requested
                color_analysis = {}
                if include_color_analysis:
                    color_analysis = self._analyze_colors(img)
                
                # Calculate quality metrics if requested
                quality_metrics = {}
                if include_quality_metrics:
                    quality_metrics = self._calculate_quality_metrics(img, input_path)
                
                # Generate histogram if requested
                histogram_data = {}
                if include_histogram:
                    histogram_data = self._generate_histogram(img, max_histogram_bins)
                
                # Combine all information
                results = {
                    **basic_info,
                    'exif_metadata': exif_data,
                    'color_analysis': color_analysis,
                    'quality_metrics': quality_metrics,
                    'histogram': histogram_data,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                logger.info(f"Image analysis completed for {input_path}: {basic_info['dimensions']}, {basic_info['file_size_human']}")
                return results
                
        except Exception as e:
            logger.error(f"Image analysis failed for {input_path}: {e}")
            return self._error_result(f"Image analysis failed: {str(e)}")
    
    def _get_basic_info(self, img: Image.Image, input_path: str) -> Dict[str, Any]:
        """Get basic image information."""
        path_obj = Path(input_path)
        file_stat = path_obj.stat()
        
        # Basic file information
        basic_info = {
            'file_path': str(path_obj.absolute()),
            'file_name': path_obj.name,
            'file_extension': path_obj.suffix.lower(),
            'file_size': file_stat.st_size,
            'file_size_human': self._format_size(file_stat.st_size),
            'created_time': file_stat.st_ctime,
            'modified_time': file_stat.st_mtime,
            'accessed_time': file_stat.st_atime,
        }
        
        # Image properties
        basic_info.update({
            'dimensions': f"{img.width}x{img.height}",
            'width': img.width,
            'height': img.height,
            'aspect_ratio': self._calculate_aspect_ratio(img.width, img.height),
            'megapixels': round((img.width * img.height) / 1_000_000, 2),
            'color_mode': img.mode,
            'format': img.format,
            'format_description': img.format_description if hasattr(img, 'format_description') else None,
            'is_animated': getattr(img, 'is_animated', False),
            'frames': getattr(img, 'n_frames', 1),
            'has_alpha': 'A' in img.mode,
            'has_transparency': img.mode in ('RGBA', 'LA', 'PA') or (img.mode == 'P' and 'transparency' in img.info),
        })
        
        # Color depth information
        basic_info['color_depth'] = self._get_color_depth(img)
        
        # DPI information
        dpi = img.info.get('dpi', (72, 72))
        basic_info['dpi'] = dpi
        basic_info['dpi_horizontal'] = dpi[0]
        basic_info['dpi_vertical'] = dpi[1]
        
        return basic_info
    
    def _get_exif_metadata(self, img: Image.Image, input_path: str) -> Dict[str, Any]:
        """Extract EXIF metadata from image."""
        exif_data = {}
        
        try:
            # Try PIL EXIF first
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = self._clean_exif_value(value)
            
            # Try exifread for more comprehensive EXIF data
            if EXIFREAD_AVAILABLE:
                with open(input_path, 'rb') as f:
                    exif_tags = exifread.process_file(f, details=False)
                    for tag, value in exif_tags.items():
                        exif_data[tag] = str(value)
            
            # Extract common photography metadata
            photography_info = self._extract_photography_info(exif_data)
            exif_data['photography'] = photography_info
            
        except Exception as e:
            logger.warning(f"EXIF extraction failed: {e}")
            exif_data['extraction_error'] = str(e)
        
        return exif_data
    
    def _extract_photography_info(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract common photography-related EXIF data."""
        photography = {}
        
        # Camera information
        camera_fields = {
            'Make': 'camera_make',
            'Model': 'camera_model',
            'LensModel': 'lens_model',
            'LensMake': 'lens_make',
        }
        
        for exif_field, output_field in camera_fields.items():
            if exif_field in exif_data:
                photography[output_field] = exif_data[exif_field]
        
        # Exposure information
        exposure_fields = {
            'ExposureTime': 'exposure_time',
            'FNumber': 'aperture',
            'ISOSpeedRatings': 'iso',
            'FocalLength': 'focal_length',
            'Flash': 'flash_fired',
        }
        
        for exif_field, output_field in exposure_fields.items():
            if exif_field in exif_data:
                photography[output_field] = exif_data[exif_field]
        
        # Date and time
        if 'DateTime' in exif_data:
            photography['capture_date'] = exif_data['DateTime']
        if 'DateTimeOriginal' in exif_data:
            photography['capture_date_original'] = exif_data['DateTimeOriginal']
        
        # GPS information
        gps_fields = ['GPSLatitude', 'GPSLongitude', 'GPSAltitude']
        gps_data = {}
        for field in gps_fields:
            if field in exif_data:
                gps_data[field.lower()] = exif_data[field]
        if gps_data:
            photography['gps'] = gps_data
        
        return photography
    
    def _analyze_colors(self, img: Image.Image) -> Dict[str, Any]:
        """Perform color analysis on image."""
        analysis = {}
        
        try:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img
            
            # Convert to numpy array for analysis
            img_array = np.array(rgb_img)
            
            # Basic color statistics
            analysis.update({
                'mean_color': np.mean(img_array, axis=(0, 1)).tolist(),
                'std_color': np.std(img_array, axis=(0, 1)).tolist(),
                'min_color': np.min(img_array, axis=(0, 1)).tolist(),
                'max_color': np.max(img_array, axis=(0, 1)).tolist(),
            })
            
            # Dominant colors (simplified)
            analysis['dominant_colors'] = self._find_dominant_colors(img_array)
            
            # Colorfulness metric
            analysis['colorfulness'] = self._calculate_colorfulness(img_array)
            
            # Brightness and contrast
            if img.mode == 'L' or 'L' in img.mode:
                gray_array = np.array(img.convert('L'))
                analysis['brightness'] = np.mean(gray_array)
                analysis['contrast'] = np.std(gray_array)
            
            # Color distribution
            analysis['color_distribution'] = self._analyze_color_distribution(img)
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def _find_dominant_colors(self, img_array: np.ndarray, n_colors: int = 5) -> List[Dict[str, Any]]:
        """Find dominant colors in image (simplified version)."""
        try:
            # Reshape and sample pixels
            pixels = img_array.reshape(-1, 3)
            sample_size = min(10000, len(pixels))
            sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
            
            # Simple clustering by quantizing colors
            quantized = (sampled_pixels // 32) * 32  # Reduce to 8 levels per channel
            unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
            
            # Get top colors
            top_indices = np.argsort(counts)[-n_colors:][::-1]
            dominant_colors = []
            
            for idx in top_indices:
                color = unique_colors[idx]
                percentage = (counts[idx] / len(sampled_pixels)) * 100
                dominant_colors.append({
                    'rgb': color.tolist(),
                    'percentage': round(percentage, 2),
                    'hex': self._rgb_to_hex(color)
                })
            
            return dominant_colors
            
        except Exception as e:
            logger.warning(f"Dominant color analysis failed: {e}")
            return []
    
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
    
    def _analyze_color_distribution(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze color distribution across image."""
        distribution = {}
        
        try:
            # Convert to HSV for better color analysis
            hsv_img = img.convert('HSV')
            h, s, v = hsv_img.split()
            
            h_array = np.array(h)
            s_array = np.array(s)
            v_array = np.array(v)
            
            distribution.update({
                'hue_mean': round(float(np.mean(h_array)), 2),
                'hue_std': round(float(np.std(h_array)), 2),
                'saturation_mean': round(float(np.mean(s_array)), 2),
                'saturation_std': round(float(np.std(s_array)), 2),
                'value_mean': round(float(np.mean(v_array)), 2),
                'value_std': round(float(np.std(v_array)), 2),
            })
            
        except Exception as e:
            logger.warning(f"Color distribution analysis failed: {e}")
        
        return distribution
    
    def _calculate_quality_metrics(self, img: Image.Image, input_path: str) -> Dict[str, Any]:
        """Calculate image quality metrics."""
        metrics = {}
        
        try:
            # File-based metrics
            file_size = os.path.getsize(input_path)
            pixel_count = img.width * img.height
            file_bpp = (file_size * 8) / pixel_count if pixel_count > 0 else 0
            
            metrics.update({
                'bits_per_pixel': round(file_bpp, 2),
                'compression_ratio': file_size / pixel_count if pixel_count > 0 else 0,
            })
            
            # Image-based metrics
            if img.mode == 'L' or 'L' in img.mode:
                gray_array = np.array(img.convert('L'))
                
                # Sharpness (using gradient magnitude)
                if CV2_AVAILABLE:
                    gy, gx = np.gradient(gray_array.astype(float))
                    sharpness = np.mean(np.sqrt(gx**2 + gy**2))
                    metrics['sharpness'] = round(float(sharpness), 2)
                
                # Noise estimation (simplified)
                blurred = cv2.GaussianBlur(gray_array.astype(float), (5, 5), 0) if CV2_AVAILABLE else gray_array.astype(float)
                noise = np.std(gray_array.astype(float) - blurred)
                metrics['noise_level'] = round(float(noise), 2)
            
            # Quality assessment
            metrics['quality_assessment'] = self._assess_quality(metrics, img)
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            metrics['calculation_error'] = str(e)
        
        return metrics
    
    def _assess_quality(self, metrics: Dict[str, Any], img: Image.Image) -> str:
        """Assess image quality based on metrics."""
        score = 0
        
        # File size quality (higher bpp generally means better quality)
        bpp = metrics.get('bits_per_pixel', 0)
        if bpp > 2.0:
            score += 2
        elif bpp > 1.0:
            score += 1
        
        # Resolution quality
        megapixels = (img.width * img.height) / 1_000_000
        if megapixels > 8:
            score += 2
        elif megapixels > 2:
            score += 1
        
        # Sharpness quality
        sharpness = metrics.get('sharpness', 0)
        if sharpness > 10:
            score += 2
        elif sharpness > 5:
            score += 1
        
        # Noise quality
        noise = metrics.get('noise_level', 0)
        if noise < 5:
            score += 2
        elif noise < 10:
            score += 1
        
        if score >= 6:
            return 'excellent'
        elif score >= 4:
            return 'good'
        elif score >= 2:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_histogram(self, img: Image.Image, max_bins: int = 256) -> Dict[str, Any]:
        """Generate color histogram data."""
        histogram = {}
        
        try:
            # Convert to RGB for consistent analysis
            rgb_img = img.convert('RGB')
            r, g, b = rgb_img.split()
            
            # Generate histograms for each channel
            histogram['red'] = np.histogram(np.array(r), bins=max_bins, range=(0, 255))[0].tolist()
            histogram['green'] = np.histogram(np.array(g), bins=max_bins, range=(0, 255))[0].tolist()
            histogram['blue'] = np.histogram(np.array(b), bins=max_bins, range=(0, 255))[0].tolist()
            
            # For grayscale images
            if img.mode == 'L':
                gray_hist = np.histogram(np.array(img), bins=max_bins, range=(0, 255))[0].tolist()
                histogram['gray'] = gray_hist
            
            # Summary statistics
            histogram['summary'] = {
                'bins': max_bins,
                'range': [0, 255],
                'total_pixels': img.width * img.height
            }
            
        except Exception as e:
            logger.warning(f"Histogram generation failed: {e}")
            histogram['generation_error'] = str(e)
        
        return histogram
    
    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate aspect ratio as string (e.g., '16:9')."""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        divisor = gcd(width, height)
        return f"{width // divisor}:{height // divisor}"
    
    def _get_color_depth(self, img: Image.Image) -> str:
        """Get color depth information."""
        mode_bits = {
            '1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32,
            'CMYK': 32, 'YCbCr': 24, 'LAB': 24, 'HSV': 24
        }
        bits = mode_bits.get(img.mode, 0)
        return f"{bits} bits per pixel"
    
    def _clean_exif_value(self, value) -> Any:
        """Clean EXIF values for JSON serialization."""
        if isinstance(value, bytes):
            return value.hex()
        elif hasattr(value, '__str__'):
            return str(value)
        return value
    
    def _rgb_to_hex(self, rgb: np.ndarray) -> str:
        """Convert RGB array to hex color code."""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def _format_size(self, size_bytes: int) -> str:
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
    
    def batch_analyze(self, 
                     image_paths: List[str],
                     **kwargs) -> Dict[str, Any]:
        """
        Analyze multiple images in batch.
        
        Args:
            image_paths: List of image paths to analyze
            **kwargs: Additional analysis parameters
            
        Returns:
            Batch analysis results
        """
        results = {
            'total_images': len(image_paths),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_megapixels': 0,
            'average_quality_score': 0,
            'analysis_results': []
        }
        
        quality_scores = []
        total_mp = 0
        
        for image_path in image_paths:
            try:
                analysis_result = self.run(image_path, **kwargs)
                
                if analysis_result.get('success'):
                    results['successful_analyses'] += 1
                    total_mp += analysis_result.get('megapixels', 0)
                    
                    # Extract quality score
                    quality = analysis_result.get('quality_metrics', {}).get('quality_assessment', 'poor')
                    quality_scores.append(self._quality_to_score(quality))
                
                else:
                    results['failed_analyses'] += 1
                
                results['analysis_results'].append({
                    'input_file': image_path,
                    'success': analysis_result.get('success', False),
                    'result': analysis_result if analysis_result.get('success') else {'error': analysis_result.get('error')}
                })
                
            except Exception as e:
                results['failed_analyses'] += 1
                results['analysis_results'].append({
                    'input_file': image_path,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"Batch analysis failed for {image_path}: {e}")
        
        if quality_scores:
            results['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        results['total_megapixels'] = total_mp
        
        return results
    
    def _quality_to_score(self, quality: str) -> int:
        """Convert quality assessment to numerical score."""
        scores = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        return scores.get(quality, 1)
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy image info tool (maintains original interface)."""
    
    name = 'get_image_info'
    description = 'Get image information'
    
    def __init__(self):
        self.enhanced_tool = ImageInfoTool()
    
    def run(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Get image information with enhanced capabilities.
        
        Args:
            input_path: Path to input image
            **kwargs: Additional analysis parameters
            
        Returns:
            Image information dictionary
        """
        return self.enhanced_tool.run(input_path=input_path, **kwargs)


# Example usage
if __name__ == "__main__":
    tool = ImageInfoTool()
    
    # Test with a sample image
    result = tool.run(
        input_path="sample.jpg",
        include_exif=True,
        include_color_analysis=True,
        include_quality_metrics=True
    )
    
    if result['success']:
        print(f"Image Analysis Results:")
        print(f"Dimensions: {result['dimensions']}")
        print(f"File Size: {result['file_size_human']}")
        print(f"Color Mode: {result['color_mode']}")
        print(f"Megapixels: {result['megapixels']}")
        print(f"Quality: {result['quality_metrics'].get('quality_assessment', 'N/A')}")
        
        if result['exif_metadata'].get('photography'):
            photo_info = result['exif_metadata']['photography']
            if 'camera_model' in photo_info:
                print(f"Camera: {photo_info['camera_model']}")
    else:
        print(f"Analysis failed: {result['error']}")