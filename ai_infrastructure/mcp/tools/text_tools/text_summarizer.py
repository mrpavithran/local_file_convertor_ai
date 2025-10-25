import re
from typing import Dict, List, Any, Optional

class Tool:
    name = 'summarize_text'
    description = 'Smart adaptive text summarizer with quality metrics'

    def run(self, 
            text: str, 
            length: int = 200,
            strategy: Optional[str] = None,
            preserve_sentences: bool = True,
            min_sentences: int = 1) -> Dict[str, Any]:
        """
        Summarize text intelligently.

        Args:
            text: Text to summarize
            length: Maximum summary length in characters
            strategy: Summarization strategy ('first_n', 'key_sentences', 'extractive'), auto-selected if None
            preserve_sentences: Preserve full sentences
            min_sentences: Minimum sentences to include

        Returns:
            Summary results with metrics
        """
        if not text or not text.strip():
            return {'summary': '', 'original_length': 0, 'summary_length': 0}

        original_length = len(text)

        # Auto-detect strategy if not specified
        if strategy is None:
            strategy = self._auto_strategy(text)

        if strategy == 'first_n':
            summary = self._first_n_summary(text, length, preserve_sentences)
        elif strategy == 'key_sentences':
            summary = self._key_sentences_summary(text, length)
        elif strategy == 'extractive':
            summary = self._extractive_summary(text, length)
        else:
            summary = self._first_n_summary(text, length, preserve_sentences)

        # Ensure minimum sentence count
        summary = self._ensure_min_sentences(summary, text, min_sentences)

        # Compute quality metrics
        quality = self._compute_quality_metrics(text, summary)

        return {
            'summary': summary,
            'original_length': original_length,
            'summary_length': len(summary),
            'compression_ratio': round(len(summary) / original_length, 3),
            'strategy_used': strategy,
            'preserved_sentences': preserve_sentences,
            'min_sentences': min_sentences,
            'quality_metrics': quality
        }

    # ----------------------
    # Summarization strategies
    # ----------------------
    def _first_n_summary(self, text: str, length: int, preserve_sentences: bool) -> str:
        if not preserve_sentences or length >= len(text):
            return text[:length]

        truncated = text[:length]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        sentence_end = max(last_period, last_exclamation, last_question)

        if sentence_end > 0:
            return text[:sentence_end + 1]
        return truncated

    def _key_sentences_summary(self, text: str, length: int) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:length]

        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            if i == 0:  # First sentence is important
                score += 2
            score += min(len(sentence) / 100, 1)
            scored_sentences.append((sentence, score, len(sentence)))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        summary_parts = []
        current_length = 0
        for sentence, _, sent_len in scored_sentences:
            if current_length + sent_len <= length:
                summary_parts.append(sentence)
                current_length += sent_len
            else:
                break
        if not summary_parts:
            return text[:length]
        return ' '.join(summary_parts)

    def _extractive_summary(self, text: str, length: int) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text[:length]

        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_words = re.findall(r'\b\w+\b', sentence.lower())
            for word in sentence_words:
                if word in word_freq and len(word) > 3:
                    score += word_freq[word]
            if len(sentence_words) > 0:
                score /= len(sentence_words)
            scored_sentences.append((sentence, score, len(sentence)))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        summary_parts = []
        current_length = 0
        for sentence, _, sent_len in scored_sentences:
            if current_length + sent_len <= length:
                summary_parts.append(sentence)
                current_length += sent_len
        if not summary_parts:
            return self._first_n_summary(text, length, True)
        return ' '.join(summary_parts)

    # ----------------------
    # Utilities
    # ----------------------
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _ensure_min_sentences(self, summary: str, text: str, min_sentences: int) -> str:
        sentences = self._split_sentences(summary)
        if len(sentences) < min_sentences:
            extra = self._split_sentences(text)
            return ' '.join(extra[:min_sentences])
        return summary

    def _compute_quality_metrics(self, original: str, summary: str) -> Dict[str, float]:
        original_words = set(re.findall(r'\b\w+\b', original.lower()))
        summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
        retention = len(summary_words & original_words) / max(len(original_words), 1)

        sentences = self._split_sentences(summary)
        words = re.findall(r'\b\w+\b', summary)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        readability = max(0, 1 - abs(avg_sentence_length - 15)/15)  # crude estimate

        return {
            'information_retention': round(retention, 3),
            'readability': round(readability, 3)
        }

    def _auto_strategy(self, text: str) -> str:
        word_count = len(text.split())
        if word_count < 50:
            return 'first_n'
        elif any(word in text.lower() for word in ['study', 'research', 'data', 'analysis']):
            return 'extractive'
        else:
            return 'key_sentences'

    # ----------------------
    # Batch summarization
    # ----------------------
    def batch_summarize(self, texts: List[str], length: int = 200, strategy: Optional[str] = None) -> Dict[str, Any]:
        summaries = []
        for text in texts:
            result = self.run(text, length, strategy)
            summaries.append(result)
        avg_compression = round(
            sum(s.get('compression_ratio', 0) for s in summaries) / len(summaries), 3
        ) if summaries else 0
        return {'summaries': summaries, 'total_texts': len(texts), 'average_compression_ratio': avg_compression}

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    summarizer = Tool()
    sample_text = """
    Artificial intelligence is transforming many industries. Machine learning algorithms can now 
    perform tasks that were once thought to require human intelligence. Natural language processing 
    enables computers to understand and generate human language. Computer vision allows machines to 
    interpret visual information. These technologies are being applied in healthcare, finance, 
    transportation, and many other fields. The rapid advancement of AI raises important ethical 
    questions about privacy, bias, and job displacement. Researchers are working on developing 
    responsible AI systems that benefit society while minimizing potential harms.
    """

    result = summarizer.run(sample_text, 150)
    print("Summary:", result['summary'])
    print("Metrics:", result['quality_metrics'])
