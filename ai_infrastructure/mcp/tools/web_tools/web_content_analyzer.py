import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import re

class Tool:
    name = 'analyze_web'
    description = 'Web content quality analyzer with quick and full modes'

    def run(self, url: str, analysis_type: str = 'full') -> Dict[str, Any]:
        """
        Analyze web content for quality, SEO, technical, and UX.

        Args:
            url: URL to analyze
            analysis_type: 'full' for detailed analysis, 'quick' for basic metrics

        Returns:
            Analysis results with score, notes, and recommendations
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; WebAnalyzer/1.0)'}
            response = requests.get(url, timeout=10, headers=headers)
            if response.status_code != 200:
                return {'error': f'HTTP {response.status_code}', 'url': url}

            soup = BeautifulSoup(response.text, 'html.parser')

            if analysis_type == 'quick':
                return self._quick_analysis(soup, url, response)
            else:
                return self._full_analysis(soup, url, response)

        except Exception as e:
            return {'error': str(e), 'url': url}

    # ------------------ Quick Analysis ------------------
    def _quick_analysis(self, soup: BeautifulSoup, url: str, response) -> Dict[str, Any]:
        title = soup.title.string if soup.title else ''
        meta_desc = self._get_meta_description(soup)
        text_content = self._extract_text_content(soup)
        word_count = len(text_content.split())

        # Quick scoring
        score = self._calculate_quick_score(title, meta_desc, word_count)

        # Notes
        notes = []
        if not title:
            notes.append("❌ Missing page title")
        elif not (30 <= len(title) <= 60):
            notes.append(f"⚠️ Title length ({len(title)}) should be 30-60 characters")
        else:
            notes.append("✅ Title is well-formatted")

        if not meta_desc:
            notes.append("❌ Missing meta description")
        else:
            notes.append("✅ Meta description present")

        if word_count < 200:
            notes.append("⚠️ Content appears thin")
        else:
            notes.append("✅ Substantial content")

        return {
            'url': url,
            'score': round(score, 2),
            'notes': notes,
            'metrics': {
                'title_present': bool(title),
                'meta_description_present': bool(meta_desc),
                'content_length_words': word_count,
                'response_code': response.status_code,
                'content_type': response.headers.get('content-type', '')
            }
        }

    # ------------------ Full Analysis ------------------
    def _full_analysis(self, soup: BeautifulSoup, url: str, response) -> Dict[str, Any]:
        # Content Quality
        content_quality = self._analyze_content_quality(soup)
        # SEO
        seo_analysis = self._analyze_seo(soup)
        # Technical
        technical_analysis = self._analyze_technical(soup, response)
        # UX
        ux_analysis = self._analyze_user_experience(soup)

        overall_score = (
            content_quality['score'] * 0.4 +
            seo_analysis['score'] * 0.3 +
            technical_analysis['score'] * 0.2 +
            ux_analysis['score'] * 0.1
        )

        recommendations = self._generate_recommendations(content_quality, seo_analysis, technical_analysis, ux_analysis)

        return {
            'url': url,
            'overall_score': round(overall_score, 2),
            'grade': self._score_to_grade(overall_score),
            'category_scores': {
                'content_quality': content_quality['score'],
                'seo': seo_analysis['score'],
                'technical': technical_analysis['score'],
                'user_experience': ux_analysis['score']
            },
            'content_quality': content_quality,
            'seo_analysis': seo_analysis,
            'technical_analysis': technical_analysis,
            'user_experience': ux_analysis,
            'recommendations': recommendations
        }

    # ------------------ Helper Methods ------------------
    def _get_meta_description(self, soup: BeautifulSoup) -> str:
        meta = soup.find('meta', attrs={'name': 'description'})
        return meta.get('content', '') if meta else ''

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        for script in soup(["script", "style"]):
            script.decompose()
        return " ".join(line.strip() for line in soup.get_text().splitlines() if line.strip())

    def _calculate_quick_score(self, title: str, meta_desc: str, word_count: int) -> float:
        score = 0.5  # base
        if title: score += 0.2
        if meta_desc: score += 0.2
        if word_count > 100: score += 0.1
        return min(score, 1.0)

    def _analyze_content_quality(self, soup: BeautifulSoup) -> Dict[str, Any]:
        text = self._extract_text_content(soup)
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        word_count = len(words)
        avg_sentence_length = word_count / len(sentences) if sentences else 0
        avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
        headers = self._analyze_headings(soup)
        score = 0
        if word_count > 300: score += 2
        elif word_count > 100: score += 1
        if 15 <= avg_sentence_length <= 25: score += 2
        if 4 <= avg_word_length <= 6: score += 1
        score += min(headers['score'], 2)
        return {
            'score': round(score / 7, 2),
            'word_count': word_count,
            'average_sentence_length': round(avg_sentence_length, 1),
            'average_word_length': round(avg_word_length, 1),
            'heading_structure': headers
        }

    def _analyze_seo(self, soup: BeautifulSoup) -> Dict[str, Any]:
        title = soup.title.string if soup.title else ''
        meta_desc = self._get_meta_description(soup)
        images = soup.find_all('img')
        images_with_alt = len([img for img in images if img.get('alt')])
        score = 0
        issues = []
        if title:
            score += 1 if 30 <= len(title) <= 60 else 0
            if not (30 <= len(title) <= 60): issues.append(f"Title length {len(title)} not ideal")
        else:
            issues.append("Missing title")
        if meta_desc:
            score += 1 if 120 <= len(meta_desc) <= 160 else 0
            if not (120 <= len(meta_desc) <= 160): issues.append(f"Meta description length {len(meta_desc)} not ideal")
        else:
            issues.append("Missing meta description")
        if images:
            score += 1 if images_with_alt / len(images) > 0.8 else 0
            if images_with_alt / len(images) <= 0.8: issues.append(f"Only {int(images_with_alt/len(images)*100)}% of images have alt")
        return {'score': round(score / 4, 2), 'issues': issues, 'images_with_alt': images_with_alt, 'total_images': len(images)}

    def _analyze_technical(self, soup: BeautifulSoup, response) -> Dict[str, Any]:
        viewport = bool(soup.find('meta', attrs={'name':'viewport'}))
        charset = bool(soup.find('meta', attrs={'charset': True}) or soup.find('meta', attrs={'http-equiv':'content-type'}))
        lang = bool(soup.find('html') and soup.find('html').get('lang'))
        score = sum([viewport, charset, lang, response.status_code==200])
        return {'score': round(score/4,2), 'has_viewport': viewport, 'has_charset': charset, 'has_lang': lang, 'response_code': response.status_code}

    def _analyze_user_experience(self, soup: BeautifulSoup) -> Dict[str, Any]:
        nav_links = len(soup.find_all('a', href=True))
        images = len(soup.find_all('img'))
        lists = len(soup.find_all(['ul','ol']))
        forms = len(soup.find_all('form'))
        score = sum([nav_links>5, images>0, lists>0, forms>0])
        return {'score': round(score/4,2), 'navigation_links': nav_links, 'images': images, 'lists': lists, 'forms': forms}

    def _analyze_headings(self, soup: BeautifulSoup) -> Dict[str, Any]:
        headers = {f'h{i}': len(soup.find_all(f'h{i}')) for i in range(1,7)}
        h1_count = headers['h1']
        score = 1 if h1_count>0 else 0
        score += 1 if h1_count==1 else 0
        return {'score': round(score/2,2), 'has_h1': h1_count>0, 'h1_count': h1_count, 'headings': headers}

    def _generate_recommendations(self, content: Dict, seo: Dict, technical: Dict, ux: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        # Content recommendations
        if content['word_count'] < 300:
            recommendations.append("📝 Increase content to at least 300 words for better depth")
        
        if not content['heading_structure']['has_h1']:
            recommendations.append("🔤 Add an H1 heading for better structure and SEO")
        elif content['heading_structure']['h1_count'] > 1:
            recommendations.append("🔤 Use only one H1 heading per page")
        
        # SEO recommendations
        recommendations.extend(seo.get('issues', []))
        
        # Technical recommendations
        if not technical['has_viewport']:
            recommendations.append("📱 Add viewport meta tag for mobile responsiveness")
        
        if not technical['has_lang']:
            recommendations.append("🌐 Add lang attribute to HTML tag for accessibility")
        
        # UX recommendations
        if ux['navigation_links'] < 5:
            recommendations.append("🧭 Add more navigation links for better user experience")
        
        if ux['images'] == 0:
            recommendations.append("🖼️ Consider adding images to improve visual appeal")
        
        return recommendations

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.9: return 'A+'
        elif score >= 0.8: return 'A'
        elif score >= 0.7: return 'B'
        elif score >= 0.6: return 'C'
        elif score >= 0.5: return 'D'
        else: return 'F'