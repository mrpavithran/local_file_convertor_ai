import re
class Tool:
    name='analyze_text'
    description='Basic text stats'
    def run(self, text):
        words = re.findall(r"\w+", text)
        return {'words': len(words), 'characters': len(text)}
import re
from collections import Counter
from typing import Dict, Any, List, Tuple
import string

class Tool:
    name = 'analyze_text'
    description = 'Flexible text analysis with full or lightweight mode'
    
    def run(self, text: str, mode: str = 'full') -> Dict[str, Any]:
        """
        Analyze text with either full metrics or lightweight metrics.
        
        Args:
            text: Input text
            mode: 'full' for comprehensive metrics, 'light' for essential metrics
            
        Returns:
            Dictionary of analysis results
        """
        if not text or not text.strip():
            return self._empty_result(mode)
        
        words = re.findall(r'\b\w+\b', text)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if mode == 'light':
            avg_word_len = sum(len(w) for w in words)/len(words) if words else 0
            avg_sentence_len = len(words)/len(sentences) if sentences else 0
            unique_words = set(word.lower() for word in words)
            readability = self._estimate_readability(avg_word_len, avg_sentence_len)
            
            return {
                'words': len(words),
                'characters': len(text),
                'characters_no_spaces': len(text.replace(' ', '')),
                'sentences': len(sentences),
                'paragraphs': len(paragraphs),
                'unique_words': len(unique_words),
                'average_word_length': round(avg_word_len, 1),
                'average_sentence_length': round(avg_sentence_len, 1),
                'readability': readability
            }
        
        # Full mode analysis
        char_stats = self._analyze_characters(text)
        word_stats = self._analyze_words(words)
        sentence_stats = self._analyze_sentences(sentences, words)
        readability = self._calculate_readability(word_stats, sentence_stats)
        complexity = self._assess_complexity(word_stats, sentence_stats, readability)
        
        return {
            'basic_metrics': {
                'characters': len(text),
                'characters_no_spaces': char_stats['non_whitespace'],
                'words': len(words),
                'sentences': len(sentences),
                'paragraphs': len(paragraphs),
                'lines': text.count('\n') + 1
            },
            'character_analysis': char_stats,
            'word_analysis': word_stats,
            'sentence_analysis': sentence_stats,
            'readability_scores': readability,
            'complexity_assessment': complexity,
            'common_words': self._get_common_words(words, 10),
            'unique_insights': self._generate_insights(word_stats, sentence_stats, readability)
        }

    # ------------------------
    # Shared helper functions
    # ------------------------
    
    def _empty_result(self, mode: str) -> Dict[str, Any]:
        if mode == 'full':
            return {'error': 'No text provided', 'basic_metrics': {'characters':0,'words':0}}
        return {'words':0,'characters':0,'sentences':0}

    def _analyze_characters(self, text: str) -> Dict[str, Any]:
        total_chars = len(text)
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        punctuation = sum(c in string.punctuation for c in text)
        other = total_chars - (letters + digits + spaces + punctuation)
        common_chars = dict(Counter(text.lower()).most_common(10))
        return {
            'total': total_chars,
            'non_whitespace': total_chars - spaces,
            'letters': letters,
            'digits': digits,
            'spaces': spaces,
            'punctuation': punctuation,
            'other': other,
            'letter_ratio': round(letters/total_chars,3) if total_chars>0 else 0,
            'common_characters': common_chars
        }
    
    def _analyze_words(self, words: List[str]) -> Dict[str, Any]:
        if not words:
            return {'total':0,'unique':0,'lexical_diversity':0,'average_word_length':0}
        word_lengths = [len(w) for w in words]
        unique_words = set(words)
        syllable_counts = [max(1,len(re.findall(r'[aeiouy]+', w))) for w in words]
        return {
            'total': len(words),
            'unique': len(unique_words),
            'lexical_diversity': round(len(unique_words)/len(words),3),
            'average_word_length': round(sum(word_lengths)/len(words),1),
            'max_word_length': max(word_lengths),
            'min_word_length': min(word_lengths),
            'word_length_distribution': dict(Counter(word_lengths)),
            'syllable_stats': {'average_syllables_per_word': round(sum(syllable_counts)/len(syllable_counts),2),
                               'total_syllables': sum(syllable_counts)}
        }
    
    def _analyze_sentences(self, sentences: List[str], words: List[str]) -> Dict[str, Any]:
        if not sentences:
            return {'total':0,'average_sentence_length':0}
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        return {
            'total': len(sentences),
            'average_sentence_length': round(sum(sentence_lengths)/len(sentences),1),
            'max_sentence_length': max(sentence_lengths),
            'min_sentence_length': min(sentence_lengths),
            'words_per_sentence': round(len(words)/len(sentences),1)
        }
    
    def _calculate_readability(self, word_stats: Dict, sentence_stats: Dict) -> Dict[str,float]:
        avg_sentence_length = sentence_stats.get('average_sentence_length',0)
        avg_syllables = word_stats.get('syllable_stats',{}).get('average_syllables_per_word',0)
        flesch = max(0, 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables))
        fk_grade = max(0, (0.39*avg_sentence_length) + (11.8*avg_syllables) - 15.59)
        ari = max(0, (4.71*word_stats.get('average_word_length',0) + 0.5*avg_sentence_length - 21.43))
        return {
            'flesch_reading_ease': round(flesch,1),
            'flesch_kincaid_grade': round(fk_grade,1),
            'automated_readability_index': round(ari,1),
            'reading_level': self._interpret_readability(flesch)
        }
    
    def _estimate_readability(self, avg_word_len: float, avg_sentence_len: float) -> str:
        if avg_sentence_len < 15 and avg_word_len < 5:
            return "Easy"
        elif avg_sentence_len < 25:
            return "Moderate"
        return "Complex"
    
    def _interpret_readability(self, flesch_score: float) -> str:
        if flesch_score>=90: return 'Very Easy (5th grade)'
        if flesch_score>=80: return 'Easy (6th grade)'
        if flesch_score>=70: return 'Fairly Easy (7th grade)'
        if flesch_score>=60: return 'Standard (8th-9th grade)'
        if flesch_score>=50: return 'Fairly Difficult (10th-12th grade)'
        if flesch_score>=30: return 'Difficult (College)'
        return 'Very Difficult (Graduate)'
    
    def _assess_complexity(self, word_stats: Dict, sentence_stats: Dict, readability: Dict) -> Dict[str,Any]:
        ld = word_stats.get('lexical_diversity',0)
        asl = sentence_stats.get('average_sentence_length',0)
        awl = word_stats.get('average_word_length',0)
        score = min(max((1-ld)*25 + min(asl/5,25) + min(awl*3,25) + (100-readability.get('flesch_reading_ease',50))/2,0),100)
        return {
            'score': round(score,1),
            'level': 'Simple' if score<33 else 'Moderate' if score<66 else 'Complex',
            'factors': {
                'vocabulary_diversity': 'High' if ld>0.7 else 'Medium' if ld>0.5 else 'Low',
                'sentence_structure': 'Complex' if asl>20 else 'Moderate' if asl>15 else 'Simple',
                'word_complexity': 'Complex' if awl>6 else 'Moderate' if awl>5 else 'Simple'
            }
        }
    
    def _get_common_words(self, words: List[str], top_n:int=10) -> List[Tuple[str,int]]:
        common_filter = {'the','and','a','an','in','on','at','to','for','of','is','are'}
        filtered = [w for w in words if w.lower() not in common_filter and len(w)>2]
        counter = Counter(filtered if filtered else words)
        return counter.most_common(top_n)
    
    def _generate_insights(self, word_stats: Dict, sentence_stats: Dict, readability: Dict) -> List[str]:
        insights = []
        ld = word_stats.get('lexical_diversity',0)
        asl = sentence_stats.get('average_sentence_length',0)
        avg_word_len = word_stats.get('average_word_length',0)
        level = readability.get('reading_level','')
        if ld>0.7: insights.append("Rich vocabulary with high word diversity")
        elif ld<0.3: insights.append("Consider using more varied vocabulary")
        if asl>25: insights.append("Long sentences may affect readability - consider breaking them up")
        elif asl<10: insights.append("Short, punchy sentences - good for clarity")
        if 'Difficult' in level: insights.append("Text may be challenging for general audiences")
        elif 'Easy' in level: insights.append("Accessible writing style suitable for broad audiences")
        if avg_word_len>6: insights.append("Uses longer words - consider simpler alternatives")
        return insights
