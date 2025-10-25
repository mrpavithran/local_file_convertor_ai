import requests
from typing import Dict, Any, List
import time
from urllib.parse import urlparse

class Tool:
    name = 'http_status'
    description = 'Enhanced HTTP status checker with redirect tracing and performance metrics'

    def run(self, url: str) -> Dict[str, Any]:
        """
        Your original HTTP status checker enhanced with critical additional metrics.
        Maintains exact same interface while adding valuable insights.
        
        Args:
            url: URL to check
            
        Returns:
            HTTP status information with performance and redirect details
        """
        try:
            start_time = time.time()
            
            # Use HEAD first for efficiency, fallback to GET if needed
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                # If HEAD returns 405 Method Not Allowed, use GET
                if response.status_code == 405:
                    response = requests.get(url, timeout=10, allow_redirects=True)
            except requests.exceptions.ConnectionError:
                # If HEAD fails with connection error, try GET
                response = requests.get(url, timeout=10, allow_redirects=True)
            
            response_time = time.time() - start_time
            
            # Your original data (preserved)
            status_code = response.status_code
            
            # Enhanced metrics
            redirect_chain = self._get_redirect_chain(response)
            final_url = response.url
            content_type = response.headers.get('content-type', 'Unknown')
            server = response.headers.get('server', 'Unknown')
            is_https = urlparse(final_url).scheme == 'https'
            
            # Status category
            status_category = self._get_status_category(status_code)
            
            return {
                'status_code': status_code,  # Your original field
                'status_category': status_category,  # Useful addition
                'response_time_seconds': round(response_time, 2),  # Performance insight
                'final_url': final_url,  # Redirect insight
                'redirect_count': len(redirect_chain) - 1,  # Redirect insight
                'content_type': content_type,  # Content insight
                'server': server,  # Server insight
                'is_https': is_https,  # Security insight
                'url_checked': url  # Reference
            }
            
        except requests.exceptions.Timeout:
            return {
                'status_code': 0,
                'error': 'Request timeout after 10 seconds',
                'url_checked': url
            }
        except requests.exceptions.ConnectionError:
            return {
                'status_code': 0,
                'error': 'Connection failed - site may be down',
                'url_checked': url
            }
        except requests.exceptions.RequestException as e:
            return {
                'status_code': 0,
                'error': f'Request failed: {str(e)}',
                'url_checked': url
            }
        except Exception as e:
            return {
                'status_code': 0,
                'error': f'Unexpected error: {str(e)}',
                'url_checked': url
            }

    def _get_redirect_chain(self, response: requests.Response) -> List[str]:
        """Extract redirect chain from response history."""
        chain = [response.url]  # Start with final URL
        for resp in response.history:
            chain.insert(0, resp.url)  # Add redirects in order
        return chain

    def _get_status_category(self, status_code: int) -> str:
        """Categorize HTTP status codes."""
        if 100 <= status_code < 200:
            return 'Informational'
        elif 200 <= status_code < 300:
            return 'Success'
        elif 300 <= status_code < 400:
            return 'Redirection'
        elif 400 <= status_code < 500:
            return 'Client Error'
        elif 500 <= status_code < 600:
            return 'Server Error'
        else:
            return 'Unknown'