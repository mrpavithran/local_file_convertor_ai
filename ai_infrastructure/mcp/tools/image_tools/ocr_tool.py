"""
Enhanced OCR Tool with image preprocessing and advanced text extraction capabilities.
Dependencies: Pillow, pytesseract, opencv-python (optional), numpy
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Install with: pip install opencv-python")

logger = logging.getLogger(__name__)

class OCRTool:
    """Enhanced OCR tool with advanced image preprocessing and text extraction."""
    
    name = 'ocr_image'
    description = 'Perform OCR on images with advanced preprocessing, multiple languages, and detailed text extraction'
    
    # Supported languages and their codes
    SUPPORTED_LANGUAGES = {
        'eng': 'English',
        'spa': 'Spanish',
        'fra': 'French',
        'deu': 'German',
        'ita': 'Italian',
        'por': 'Portuguese',
        'rus': 'Russian',
        'chi_sim': 'Chinese Simplified',
        'chi_tra': 'Chinese Traditional',
        'jpn': 'Japanese',
        'kor': 'Korean',
        'ara': 'Arabic',
        'hin': 'Hindi',
        'ben': 'Bengali',
    }
    
    def __init__(self):
        self.default_ocr_config = r'--oem 3 --psm 6'
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.gif'}
    
    def run(self, 
            input_path: str,
            language: str = 'eng',
            preprocess: bool = True,
            preprocessing_steps: Optional[List[str]] = None,
            output_format: str = 'text',
            confidence_threshold: int = 0,
            preserve_layout: bool = False) -> Dict[str, Any]:
        """
        Perform OCR on image with advanced options.
        
        Args:
            input_path: Path to input image
            language: OCR language code (e.g., 'eng', 'spa', 'fra')
            preprocess: Whether to preprocess image for better OCR
            preprocessing_steps: Specific preprocessing steps to apply
            output_format: Output format ('text', 'detailed', 'hocr', 'pdf')
            confidence_threshold: Minimum confidence for text extraction (0-100)
            preserve_layout: Whether to preserve text layout and structure
            
        Returns:
            OCR results with text and metadata
        """
        # Validate input
        if not os.path.exists(input_path):
            return self._error_result(f"Input image not found: {input_path}")
        
        # Validate image format
        file_ext = Path(input_path).suffix.lower()
        if file_ext not in self.supported_image_formats:
            logger.warning(f"Unsupported image format: {file_ext}")
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            return self._error_result(f"Unsupported language: {language}. Supported: {list(self.SUPPORTED_LANGUAGES.keys())}")
        
        try:
            # Load image
            original_image = Image.open(input_path)
            
            # Preprocess image if requested
            if preprocess:
                processed_image = self._preprocess_image(original_image, preprocessing_steps)
            else:
                processed_image = original_image
            
            # Build OCR configuration
            ocr_config = self._build_ocr_config(preserve_layout, confidence_threshold)
            
            # Perform OCR based on output format
            results = self._perform_ocr(processed_image, language, ocr_config, output_format)
            
            # Add metadata
            results.update({
                'input_file': os.path.basename(input_path),
                'input_path': input_path,
                'language': language,
                'language_name': self.SUPPORTED_LANGUAGES[language],
                'preprocessing_applied': preprocess,
                'preprocessing_steps': preprocessing_steps if preprocess else None,
                'output_format': output_format,
                'image_dimensions': original_image.size,
                'image_mode': original_image.mode,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            # Calculate additional statistics
            self._add_ocr_statistics(results, original_image)
            
            logger.info(f"OCR completed for {input_path}: {results.get('word_count', 0)} words, confidence: {results.get('average_confidence', 0):.2f}")
            return results
            
        except Exception as e:
            logger.error(f"OCR failed for {input_path}: {e}")
            return self._error_result(f"OCR processing failed: {str(e)}")
    
    def _preprocess_image(self, image: Image.Image, steps: Optional[List[str]] = None) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: Input PIL Image
            steps: Specific preprocessing steps to apply
            
        Returns:
            Preprocessed PIL Image
        """
        if steps is None:
            steps = ['grayscale', 'denoise', 'contrast', 'sharpness']
        
        processed_image = image.copy()
        
        for step in steps:
            try:
                if step == 'grayscale':
                    if processed_image.mode != 'L':
                        processed_image = processed_image.convert('L')
                
                elif step == 'denoise':
                    processed_image = self._apply_denoising(processed_image)
                
                elif step == 'contrast':
                    enhancer = ImageEnhance.Contrast(processed_image)
                    processed_image = enhancer.enhance(2.0)
                
                elif step == 'sharpness':
                    enhancer = ImageEnhance.Sharpness(processed_image)
                    processed_image = enhancer.enhance(2.0)
                
                elif step == 'threshold':
                    processed_image = self._apply_threshold(processed_image)
                
                elif step == 'deskew':
                    processed_image = self._apply_deskewing(processed_image)
                
                elif step == 'border_removal':
                    processed_image = self._remove_borders(processed_image)
                    
            except Exception as e:
                logger.warning(f"Preprocessing step '{step}' failed: {e}")
                continue
        
        return processed_image
    
    def _apply_denoising(self, image: Image.Image) -> Image.Image:
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
    
    def _apply_threshold(self, image: Image.Image) -> Image.Image:
        """Apply binary thresholding."""
        if CV2_AVAILABLE:
            img_array = np.array(image)
            _, thresholded = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(thresholded)
        else:
            # Simple PIL-based threshold
            return image.point(lambda x: 0 if x < 128 else 255, '1')
    
    def _apply_deskewing(self, image: Image.Image) -> Image.Image:
        """Apply deskewing to correct image rotation."""
        if not CV2_AVAILABLE:
            return image
        
        try:
            img_array = np.array(image)
            
            # Convert to binary image for deskewing
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours and compute minimum area rectangle
            coords = np.column_stack(np.where(binary > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Rotate image
            (h, w) = img_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            return Image.fromarray(rotated)
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    def _remove_borders(self, image: Image.Image) -> Image.Image:
        """Remove borders from image."""
        if not CV2_AVAILABLE:
            return image
        
        try:
            img_array = np.array(image)
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours and remove border-like contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (assumed to be the main content)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Crop to the main content
                cropped = img_array[y:y+h, x:x+w]
                return Image.fromarray(cropped)
            
            return image
            
        except Exception as e:
            logger.warning(f"Border removal failed: {e}")
            return image
    
    def _build_ocr_config(self, preserve_layout: bool, confidence_threshold: int) -> str:
        """Build tesseract OCR configuration."""
        config = self.default_ocr_config
        
        if preserve_layout:
            config += ' -c preserve_interword_spaces=1'
        
        if confidence_threshold > 0:
            config += f' -c tessedit_confidence_threshold={confidence_threshold}'
        
        return config
    
    def _perform_ocr(self, image: Image.Image, language: str, config: str, output_format: str) -> Dict[str, Any]:
        """Perform OCR with specified output format."""
        results = {}
        
        if output_format == 'text':
            # Simple text extraction
            text = pytesseract.image_to_string(image, lang=language, config=config)
            results['text'] = text.strip()
            
        elif output_format == 'detailed':
            # Detailed text extraction with bounding boxes and confidence
            ocr_data = pytesseract.image_to_data(image, lang=language, config=config, output_type=pytesseract.Output.DICT)
            results.update(self._process_detailed_ocr(ocr_data))
            
        elif output_format == 'hocr':
            # hOCR formatted output
            hocr_data = pytesseract.image_to_pdf_or_hocr(image, extension='hocr', lang=language)
            results['hocr'] = hocr_data.decode('utf-8') if isinstance(hocr_data, bytes) else hocr_data
            
        elif output_format == 'pdf':
            # PDF output with embedded text
            pdf_data = pytesseract.image_to_pdf_or_hocr(image, extension='pdf', lang=language)
            results['pdf_data'] = pdf_data
            results['pdf_size'] = len(pdf_data)
        
        return results
    
    def _process_detailed_ocr(self, ocr_data: Dict) -> Dict[str, Any]:
        """Process detailed OCR data into structured format."""
        n_boxes = len(ocr_data['level'])
        
        words = []
        confidences = []
        
        for i in range(n_boxes):
            confidence = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            
            if confidence > 0 and text:  # Only confident, non-empty text
                word_info = {
                    'text': text,
                    'confidence': confidence,
                    'bounding_box': {
                        'left': int(ocr_data['left'][i]),
                        'top': int(ocr_data['top'][i]),
                        'width': int(ocr_data['width'][i]),
                        'height': int(ocr_data['height'][i])
                    },
                    'block_num': int(ocr_data['block_num'][i]),
                    'paragraph_num': int(ocr_data['par_num'][i]),
                    'line_num': int(ocr_data['line_num'][i]),
                    'word_num': int(ocr_data['word_num'][i]),
                }
                words.append(word_info)
                confidences.append(confidence)
        
        # Calculate statistics
        total_chars = sum(len(word['text']) for word in words)
        total_words = len(words)
        
        return {
            'text': ' '.join(word['text'] for word in words),
            'detailed_words': words,
            'word_count': total_words,
            'character_count': total_chars,
            'confidence_scores': confidences,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
        }
    
    def _add_ocr_statistics(self, results: Dict[str, Any], original_image: Image.Image):
        """Add additional OCR statistics to results."""
        text = results.get('text', '')
        
        # Basic text statistics
        results.update({
            'line_count': len(text.split('\n')) if text else 0,
            'non_empty_line_count': len([line for line in text.split('\n') if line.strip()]) if text else 0,
            'estimated_reading_time_minutes': round(len(text.split()) / 200, 2) if text else 0,  # 200 wpm
        })
        
        # Language detection confidence (if multiple languages were attempted)
        if 'average_confidence' in results:
            confidence = results['average_confidence']
            if confidence > 80:
                results['quality_assessment'] = 'excellent'
            elif confidence > 60:
                results['quality_assessment'] = 'good'
            elif confidence > 40:
                results['quality_assessment'] = 'fair'
            else:
                results['quality_assessment'] = 'poor'
    
    def batch_ocr(self, 
                  image_paths: List[str],
                  language: str = 'eng',
                  output_dir: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Perform OCR on multiple images.
        
        Args:
            image_paths: List of image paths
            language: OCR language
            output_dir: Directory to save text outputs
            **kwargs: Additional OCR parameters
            
        Returns:
            Batch OCR results
        """
        results = {
            'total_images': len(image_paths),
            'successful_ocr': 0,
            'failed_ocr': 0,
            'total_words': 0,
            'total_characters': 0,
            'average_confidence': 0,
            'ocr_results': []
        }
        
        confidences = []
        
        for image_path in image_paths:
            try:
                ocr_result = self.run(image_path, language=language, **kwargs)
                
                if ocr_result.get('success'):
                    results['successful_ocr'] += 1
                    results['total_words'] += ocr_result.get('word_count', 0)
                    results['total_characters'] += ocr_result.get('character_count', 0)
                    confidences.append(ocr_result.get('average_confidence', 0))
                    
                    # Save text to file if output directory specified
                    if output_dir and 'text' in ocr_result:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = Path(output_dir) / f"{Path(image_path).stem}.txt"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(ocr_result['text'])
                        ocr_result['output_file'] = str(output_path)
                
                else:
                    results['failed_ocr'] += 1
                
                results['ocr_results'].append({
                    'input_file': image_path,
                    'success': ocr_result.get('success', False),
                    'result': ocr_result if ocr_result.get('success') else {'error': ocr_result.get('error')}
                })
                
            except Exception as e:
                results['failed_ocr'] += 1
                results['ocr_results'].append({
                    'input_file': image_path,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"Batch OCR failed for {image_path}: {e}")
        
        if confidences:
            results['average_confidence'] = sum(confidences) / len(confidences)
        
        return results
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get information about supported languages."""
        return {
            'supported_languages': self.SUPPORTED_LANGUAGES,
            'total_languages': len(self.SUPPORTED_LANGUAGES),
            'default_language': 'eng'
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
    """Legacy OCR tool (maintains original interface)."""
    
    name = 'ocr_image'
    description = 'Perform OCR on images'
    
    def __init__(self):
        self.enhanced_tool = OCRTool()
    
    def run(self, input_path: str, language: str = 'eng', **kwargs) -> Dict[str, Any]:
        """
        Perform OCR with enhanced capabilities.
        
        Args:
            input_path: Path to input image
            language: OCR language code
            **kwargs: Additional OCR parameters
            
        Returns:
            OCR results
        """
        return self.enhanced_tool.run(input_path=input_path, language=language, **kwargs)


# Example usage
if __name__ == "__main__":
    tool = OCRTool()
    
    # Test with a sample image
    result = tool.run(
        input_path="sample_image.png",
        language='eng',
        preprocess=True,
        output_format='detailed',
        confidence_threshold=60
    )
    
    if result['success']:
        print(f"OCR Results:")
        print(f"Text: {result['text'][:200]}...")
        print(f"Words: {result['word_count']}")
        print(f"Confidence: {result['average_confidence']:.2f}")
        print(f"Quality: {result['quality_assessment']}")
    else:
        print(f"OCR failed: {result['error']}")
    
    # Show supported languages
    languages = tool.get_supported_languages()
    print(f"\nSupported languages: {len(languages['supported_languages'])}")