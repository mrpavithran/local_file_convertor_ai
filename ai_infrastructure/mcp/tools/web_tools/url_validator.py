import re
from urllib.parse import urlparse
from typing import Dict, Any, List
import ipaddress

class Tool:
    name = 'validate_url'
    description = 'URL validator with syntax, domain, HTTPS, and security checks'

    def run(self, url: str) -> Dict[str, Any]:
        """
        Validate a URL with key checks:
        - Syntax
        - Domain structure
        - HTTPS usage
        - Suspicious characters
        """
        result = {
            'url': url,
            'valid': False,
            'issues': [],
            'uses_https': False,
            'has_domain': False,
            'is_suspicious': False
        }

        # Basic syntax check
        pattern = r'^https?://\S+$'
        if not re.match(pattern, url, re.IGNORECASE):
            result['issues'].append("Invalid URL syntax (should start with http:// or https://)")
            return result

        # Parse URL
        parsed = urlparse(url)
        result['uses_https'] = parsed.scheme == 'https'
        result['has_domain'] = bool(parsed.netloc)
        result['is_suspicious'] = bool(re.search(r'[<>]', url))

        # Domain & TLD check
        domain_valid = self._check_domain(parsed.netloc)
        if not domain_valid['valid']:
            result['issues'].extend(domain_valid['issues'])

        # Security suggestions
        if not result['uses_https']:
            result['issues'].append("Consider using HTTPS for secure connections")
        if result['is_suspicious']:
            result['issues'].append("URL contains suspicious characters (< or >)")

        # Determine overall validity
        result['valid'] = result['has_domain'] and domain_valid['valid'] and not result['is_suspicious']

        return result

    def _check_domain(self, netloc: str) -> Dict[str, Any]:
        """Check if domain is valid and has a TLD"""
        if not netloc:
            return {'valid': False, 'issues': ['Missing domain']}

        # Remove port if exists
        domain_part = netloc.split(':')[0]
        parts = domain_part.split('.')
        issues: List[str] = []

        if len(parts) < 2:
            issues.append("Missing top-level domain (TLD)")
        elif len(parts[-1]) < 2 or not parts[-1].isalpha():
            issues.append(f"Invalid TLD: {parts[-1]}")

        # Check if domain is IP address
        try:
            ipaddress.ip_address(parts[0])
            issues.append("Domain appears to be an IP address")
        except ValueError:
            pass  # Not an IP, OK

        return {'valid': len(issues) == 0, 'issues': issues}
