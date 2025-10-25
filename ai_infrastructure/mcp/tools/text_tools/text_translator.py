import logging
import random
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class Tool:
    name = 'translate_text'
    description = 'Enhanced text translation with multiple backend support'
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic'
        }
        
        # Mock translations for demonstration
        self.mock_translations = {
            'hello': {
                'es': 'hola',
                'fr': 'bonjour',
                'de': 'hallo',
                'it': 'ciao',
                'pt': 'olá'
            },
            'goodbye': {
                'es': 'adiós',
                'fr': 'au revoir', 
                'de': 'auf wiedersehen',
                'it': 'arrivederci',
                'pt': 'adeus'
            },
            'thank you': {
                'es': 'gracias',
                'fr': 'merci',
                'de': 'danke',
                'it': 'grazie', 
                'pt': 'obrigado'
            }
        }
    
    def run(self, 
            text: str, 
            target_lang: str = 'en',
            source_lang: Optional[str] = None,
            mock_mode: bool = True) -> Dict[str, Any]:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language code (default: 'en')
            source_lang: Source language code (auto-detected if not provided)
            mock_mode: Use mock translations for demonstration
            
        Returns:
            Translation results
        """
        try:
            # Validate target language
            if target_lang not in self.supported_languages:
                return self._error_result(f"Unsupported target language: {target_lang}")
            
            result = {
                'original_text': text,
                'translated_text': '',
                'target_language': target_lang,
                'source_language': source_lang or 'auto',
                'character_count': len(text),
                'word_count': len(text.split())
            }
            
            if mock_mode:
                translation = self._mock_translate(text, target_lang, source_lang)
                result.update({
                    'translated_text': translation,
                    'translation_engine': 'mock_demo',
                    'confidence': 0.95 if translation != text else 0.1
                })
            else:
                # Placeholder for real translation service
                result.update({
                    'translated_text': text,  # Identity fallback
                    'translation_engine': 'identity_fallback',
                    'confidence': 0.1,
                    'note': 'Real translation service not implemented'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return self._error_result(f"Translation failed: {str(e)}")
    
    def _mock_translate(self, text: str, target_lang: str, source_lang: Optional[str]) -> str:
        """Generate mock translations for demonstration."""
        text_lower = text.lower().strip()
        
        # Check for known phrases
        for phrase, translations in self.mock_translations.items():
            if phrase in text_lower:
                return translations.get(target_lang, text)
        
        # Generate mock translation for unknown text
        if target_lang != 'en':
            return f"[{target_lang.upper()}: {text}]"
        
        return text
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get list of supported languages."""
        return {
            'languages': self.supported_languages,
            'total_supported': len(self.supported_languages),
            'default_target': 'en'
        }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Simple language detection (mock)."""
        # Simple keyword-based detection for demonstration
        language_indicators = {
            'en': ['the', 'and', 'is', 'to', 'of'],
            'es': ['el', 'la', 'y', 'en', 'de'],
            'fr': ['le', 'la', 'et', 'en', 'de'],
            'de': ['der', 'die', 'das', 'und', 'in']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[lang] = score
        
        detected_lang = max(scores.items(), key=lambda x: x[1])[0] if scores else 'en'
        
        return {
            'detected_language': detected_lang,
            'confidence': min(scores[detected_lang] / 5.0, 1.0) if detected_lang in scores else 0.1,
            'scores': scores
        }
    
    def batch_translate(self, 
                       texts: List[str], 
                       target_lang: str = 'en',
                       source_lang: Optional[str] = None) -> Dict[str, Any]:
        """Translate multiple texts at once."""
        translations = []
        
        for text in texts:
            result = self.run(text, target_lang, source_lang)
            translations.append(result)
        
        return {
            'batch_results': translations,
            'total_texts': len(texts),
            'target_language': target_lang
        }
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_message,
            'success': False
        }


# Example usage
if __name__ == "__main__":
    translator = Tool()
    
    # Single translation
    result = translator.run("Hello world", "es")
    print(f"Translation: {result['translated_text']}")
    
    # Language detection
    detection = translator.detect_language("Bonjour le monde")
    print(f"Detected language: {detection['detected_language']}")
    
    # Supported languages
    languages = translator.get_supported_languages()
    print(f"Supported: {languages['total_supported']} languages")