import re
from typing import Dict, Any

class Tool:
    name = 'clean_text'
    description = 'Smart text cleaner with basic normalization'
    
    def run(self, text: str) -> Dict[str, Any]:
        """
        Clean text with smart normalization.
        Maintains your original simple interface but adds intelligent cleaning.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text with basic metrics
        """
        if not text or not text.strip():
            return {'cleaned': '', 'original_length': 0, 'changes_made': 0}
        
        original = text
        original_length = len(text)
        
        # Your original logic (preserved)
        cleaned = re.sub(r'\s+', ' ', text).strip()
        
        # Additional smart cleaning
        if len(cleaned) != len(original.strip()):
            # If we made changes beyond simple strip, do additional cleaning
            cleaned = self._smart_clean(cleaned)
        
        changes = self._count_changes(original, cleaned)
        
        return {
            'cleaned': cleaned,
            'original_length': original_length,
            'cleaned_length': len(cleaned),
            'changes_made': changes,
            'reduction_percent': round((original_length - len(cleaned)) / original_length * 100, 1)
        }
    
    def _smart_clean(self, text: str) -> str:
        """Additional smart cleaning rules."""
        # Fix common punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([(])\s+', r'\1', text)       # Remove space after opening paren
        text = re.sub(r'\s+([)])', r'\1', text)       # Remove space before closing paren
        
        # Fix multiple punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)    # Reduce multiple punctuation
        
        return text
    
    def _count_changes(self, original: str, cleaned: str) -> int:
        """Count approximately how many changes were made."""
        changes = 0
        
        # Count whitespace reductions
        original_spaces = len(re.findall(r'\s{2,}', original))
        cleaned_spaces = len(re.findall(r'\s{2,}', cleaned))
        changes += original_spaces - cleaned_spaces
        
        # Count if trimming occurred
        if original.strip() != original:
            changes += 1
        
        return max(changes, 1)  # At least 1 change if we're returning cleaned text