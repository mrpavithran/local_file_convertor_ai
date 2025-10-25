import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
import re
import time
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)

class Tool:
    name = 'web_scraper'
    description = 'Full-featured web scraper with quick summary output and detailed extraction'
    
    def run(self, 
            url: str,
            extract: List[str] = None,
            timeout: int = 10,
            user_agent: str = None) -> Dict[str, Any]:
        """
        Enhanced web scraping with multiple content extraction options and quick summary.
        """
        if extract is None:
            extract = ['title', 'meta', 'links']
        
        headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (compatible; WebScraperTool/1.0)'
        }
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout, headers=headers)
            response_time = time.time() - start_time
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = {
                'url': url,
                'status_code': response.status_code,
                'response_time_seconds': round(response_time, 2),
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.content),
                'encoding': response.encoding,
            }
            
            # Extract requested content
            if 'title' in extract:
                results['title'] = self._extract_title(soup)
            if 'meta' in extract:
                results['meta_tags'] = self._extract_meta(soup)
            if 'headers' in extract:
                results['headers'] = self._extract_headers(soup)
            if 'links' in extract:
                results['links'] = self._extract_links(soup, url)
            if 'text' in extract:
                results['text_content'] = self._extract_text(soup)
            if 'images' in extract:
                results['images'] = self._extract_images(soup, url)
            
            # Basic page info
            results['page_info'] = self._get_page_info(soup, response)
            
            # Quick summary for pipelines (similar to Version 3)
            results['summary'] = {
                'title': results['title']['text'] if 'title' in results else '',
                'meta_description': results['meta_tags'].get('description', '') if 'meta_tags' in results else '',
                'word_count': results['text_content']['word_count'] if 'text_content' in results else 0,
                'link_count': len(results['links']['all']) if 'links' in results else 0,
                'images': len(results['images']) if 'images' in results else 0
            }
            
            logger.info(f"Successfully scraped {url}")
            return results
            
        except requests.exceptions.Timeout:
            return {'error': f'Request timeout after {timeout} seconds', 'url': url}
        except requests.exceptions.RequestException as e:
            return {'error': f'Request failed: {str(e)}', 'url': url}
        except Exception as e:
            return {'error': f'Scraping failed: {str(e)}', 'url': url}
    
    # ---------- Extraction Helpers ----------
    
    def _extract_title(self, soup: BeautifulSoup) -> Dict[str, Any]:
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text().strip()
            return {'text': title_text, 'length': len(title_text), 'has_title': True}
        return {'text': '', 'length': 0, 'has_title': False}
    
    def _extract_meta(self, soup: BeautifulSoup) -> Dict[str, str]:
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                meta_tags[name] = content.strip()
        return meta_tags
    
    def _extract_headers(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        headers = {}
        for i in range(1, 7):
            tag_name = f'h{i}'
            elements = soup.find_all(tag_name)
            headers[tag_name] = [elem.get_text().strip() for elem in elements if elem.get_text().strip()]
        return headers
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, List[Dict[str, Any]]]:
        links = {'internal': [], 'external': [], 'all': []}
        base_domain = urlparse(base_url).netloc
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().strip()
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            link_info = {'url': absolute_url, 'text': text, 'is_relative': href.startswith('/'), 'domain': parsed_url.netloc}
            links['all'].append(link_info)
            if parsed_url.netloc == base_domain or not parsed_url.netloc:
                links['internal'].append(link_info)
            else:
                links['external'].append(link_info)
        return links
    
    def _extract_text(self, soup: BeautifulSoup) -> Dict[str, Any]:
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        words = re.findall(r'\b\w+\b', clean_text)
        return {
            'full_text': clean_text,
            'character_count': len(clean_text),
            'word_count': len(words),
            'line_count': len(lines),
            'average_word_length': round(sum(len(word) for word in words)/len(words),1) if words else 0
        }
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            alt = img.get('alt','')
            if src:
                images.append({
                    'src': urljoin(base_url, src),
                    'alt': alt,
                    'title': img.get('title',''),
                    'width': img.get('width'),
                    'height': img.get('height')
                })
        return images
    
    def _get_page_info(self, soup: BeautifulSoup, response: requests.Response) -> Dict[str, Any]:
        lang = soup.find('html').get('lang','') if soup.find('html') else ''
        charset = soup.find('meta', charset=True)
        return {
            'language': lang,
            'charset': charset.get('charset','') if charset else '',
            'doctype': 'HTML5' if soup.find('!doctype') else 'Unknown',
            'has_forms': len(soup.find_all('form'))>0,
            'has_tables': len(soup.find_all('table'))>0,
            'has_lists': len(soup.find_all(['ul','ol']))>0
        }
    
    # ---------- Batch Scraping ----------
    
    def batch_scrape(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        results = []
        successful = 0
        failed = 0
        for url in urls:
            result = self.run(url, **kwargs)
            results.append(result)
            if 'error' not in result:
                successful += 1
            else:
                failed += 1
            time.sleep(1)
        return {
            'results': results,
            'summary': {
                'total_urls': len(urls),
                'successful': successful,
                'failed': failed,
                'success_rate': round(successful/len(urls)*100,1) if urls else 0
            }
        }
