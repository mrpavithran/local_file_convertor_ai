"""
Conversion Tools Package for MCP
Provides file format conversion tools including batch operations.
"""

import logging
from typing import List, Dict, Any, Optional
import importlib

logger = logging.getLogger(__name__)

# Import individual conversion tools
from .convert_csv_to_xlsx import Tool as CSVToXLSXConverterTool
from .convert_docx_to_pdf import Tool as DOCXToPDFConverterTool
from .convert_pdf_to_docx import Tool as PDFToDOCXConverterTool
from .batch_converter import Tool as BatchConverterTool

# Optional tools with fallback
try:
    from .convert_txt_to_pdf import Tool as TXTToPDFConverterTool
    TXT_TO_PDF_AVAILABLE = True
except ImportError:
    TXT_TO_PDF_AVAILABLE = False
    logger.warning("TXT to PDF converter not available")

try:
    from .image_converter import Tool as ImageConverterTool
    IMAGE_CONVERSION_AVAILABLE = True
except ImportError:
    IMAGE_CONVERSION_AVAILABLE = False
    logger.warning("Image conversion tools not available")

__all__ = [
    "CSVToXLSXConverterTool",
    "DOCXToPDFConverterTool", 
    "PDFToDOCXConverterTool",
    "BatchConverterTool",
    "get_all_tools",
    "get_tool_by_name",
    "get_supported_conversions",
    "validate_conversion_path"
]

class ConversionToolsManager:
    """Manager for conversion tools with additional utilities."""
    
    def __init__(self):
        self.tools = self._discover_tools()
        self.conversion_matrix = self._build_conversion_matrix()
    
    def _discover_tools(self) -> Dict[str, Any]:
        """Discover and initialize all available conversion tools."""
        tools = {
            'convert_csv_to_xlsx': CSVToXLSXConverterTool(),
            'convert_docx_to_pdf': DOCXToPDFConverterTool(),
            'convert_pdf_to_docx': PDFToDOCXConverterTool(),
            'batch_convert': BatchConverterTool(),
        }
        
        # Add optional tools if available
        if TXT_TO_PDF_AVAILABLE:
            tools['convert_txt_to_pdf'] = TXTToPDFConverterTool()
        
        if IMAGE_CONVERSION_AVAILABLE:
            tools['convert_image'] = ImageConverterTool()
        
        return tools
    
    def _build_conversion_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Build a matrix of supported conversions."""
        return {
            'csv→xlsx': {
                'tool': 'convert_csv_to_xlsx',
                'input_ext': ['.csv'],
                'output_ext': '.xlsx',
                'description': 'Convert CSV to Excel XLSX'
            },
            'docx→pdf': {
                'tool': 'convert_docx_to_pdf', 
                'input_ext': ['.docx', '.doc'],
                'output_ext': '.pdf',
                'description': 'Convert DOCX to PDF'
            },
            'pdf→docx': {
                'tool': 'convert_pdf_to_docx',
                'input_ext': ['.pdf'],
                'output_ext': '.docx', 
                'description': 'Convert PDF to DOCX (text extraction)'
            },
            'txt→pdf': {
                'tool': 'convert_txt_to_pdf',
                'input_ext': ['.txt'],
                'output_ext': '.pdf',
                'description': 'Convert text to PDF',
                'available': TXT_TO_PDF_AVAILABLE
            }
        }
    
    def get_available_conversions(self) -> Dict[str, Any]:
        """Get all available conversion paths."""
        available = {}
        for conversion, info in self.conversion_matrix.items():
            if info.get('available', True):
                available[conversion] = info
        return available
    
    def find_conversion_tool(self, input_ext: str, output_ext: str) -> Optional[str]:
        """Find the appropriate tool for a conversion path."""
        for conversion, info in self.conversion_matrix.items():
            if (input_ext.lower() in info['input_ext'] and 
                output_ext.lower() == info['output_ext'] and
                info.get('available', True)):
                return info['tool']
        return None

# Global manager instance
_manager = ConversionToolsManager()

def get_all_tools() -> List[Any]:
    """
    Get instances of all available conversion tools.
    
    Returns:
        List of initialized tool instances
    """
    return list(_manager.tools.values())

def get_tool_by_name(name: str) -> Optional[Any]:
    """
    Get a specific conversion tool by name.
    
    Args:
        name: Tool name (e.g., 'convert_csv_to_xlsx')
        
    Returns:
        Tool instance or None if not found
    """
    return _manager.tools.get(name)

def get_supported_conversions() -> Dict[str, Any]:
    """
    Get all supported conversion paths with details.
    
    Returns:
        Dictionary of conversion paths and their details
    """
    return _manager.get_available_conversions()

def validate_conversion_path(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Validate if a conversion path is supported.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        
    Returns:
        Validation result with tool information
    """
    from pathlib import Path
    
    input_ext = Path(input_path).suffix.lower()
    output_ext = Path(output_path).suffix.lower()
    
    tool_name = _manager.find_conversion_tool(input_ext, output_ext)
    
    if tool_name:
        return {
            'supported': True,
            'tool_name': tool_name,
            'tool': _manager.tools[tool_name],
            'input_extension': input_ext,
            'output_extension': output_ext
        }
    else:
        return {
            'supported': False,
            'input_extension': input_ext,
            'output_extension': output_ext,
            'available_conversions': _manager.get_available_conversions()
        }

def batch_convert_files(file_pairs: List[Dict[str, str]], 
                       overwrite: bool = False) -> Dict[str, Any]:
    """
    Convert multiple files using appropriate tools.
    
    Args:
        file_pairs: List of dicts with 'input' and 'output' paths
        overwrite: Whether to overwrite existing files
        
    Returns:
        Batch conversion results
    """
    results = {
        'total_files': len(file_pairs),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'conversions': []
    }
    
    batch_tool = get_tool_by_name('batch_convert')
    if not batch_tool:
        return {
            'error': 'Batch conversion tool not available',
            **results
        }
    
    # Group files by conversion type for efficient processing
    conversion_groups = {}
    for pair in file_pairs:
        validation = validate_conversion_path(pair['input'], pair['output'])
        if validation['supported']:
            conv_type = validation['tool_name']
            if conv_type not in conversion_groups:
                conversion_groups[conv_type] = []
            conversion_groups[conv_type].append(pair)
        else:
            results['conversions'].append({
                'input': pair['input'],
                'output': pair['output'],
                'status': 'failed',
                'error': f"Unsupported conversion: {validation['input_extension']} → {validation['output_extension']}"
            })
            results['failed'] += 1
    
    # Process each conversion group
    for conv_type, pairs in conversion_groups.items():
        tool = get_tool_by_name(conv_type)
        if tool:
            for pair in pairs:
                try:
                    # Check if output exists
                    if not overwrite and Path(pair['output']).exists():
                        results['conversions'].append({
                            'input': pair['input'],
                            'output': pair['output'],
                            'status': 'skipped',
                            'reason': 'Output file exists'
                        })
                        results['skipped'] += 1
                        continue
                    
                    # Perform conversion
                    conversion_result = tool.run(
                        input_path=pair['input'],
                        output_path=pair['output']
                    )
                    
                    results['conversions'].append({
                        'input': pair['input'],
                        'output': pair['output'],
                        'status': 'success',
                        'result': conversion_result
                    })
                    results['successful'] += 1
                    
                except Exception as e:
                    logger.error(f"Conversion failed for {pair['input']}: {e}")
                    results['conversions'].append({
                        'input': pair['input'],
                        'output': pair['output'],
                        'status': 'failed',
                        'error': str(e)
                    })
                    results['failed'] += 1
    
    return results

def get_conversion_statistics() -> Dict[str, Any]:
    """
    Get statistics about available conversion tools.
    
    Returns:
        Conversion tools statistics
    """
    tools = get_all_tools()
    conversions = get_supported_conversions()
    
    return {
        'total_tools': len(tools),
        'supported_conversions': len(conversions),
        'conversion_paths': list(conversions.keys()),
        'available_tools': list(_manager.tools.keys()),
        'optional_tools': {
            'txt_to_pdf': TXT_TO_PDF_AVAILABLE,
            'image_conversion': IMAGE_CONVERSION_AVAILABLE
        }
    }

# Convenience function for quick conversions
def quick_convert(input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
    """
    Quick conversion with automatic tool selection.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        **kwargs: Additional tool-specific parameters
        
    Returns:
        Conversion results
    """
    validation = validate_conversion_path(input_path, output_path)
    
    if not validation['supported']:
        return {
            'success': False,
            'error': f"Unsupported conversion: {validation['input_extension']} → {validation['output_extension']}",
            'available_conversions': validation['available_conversions']
        }
    
    try:
        tool = validation['tool']
        result = tool.run(input_path=input_path, output_path=output_path, **kwargs)
        return {
            'success': True,
            'tool_used': validation['tool_name'],
            'result': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'tool_used': validation['tool_name']
        }

# Example usage and testing
if __name__ == "__main__":
    # Print available tools and conversions
    stats = get_conversion_statistics()
    print("Conversion Tools Statistics:")
    print(f"Total tools: {stats['total_tools']}")
    print(f"Supported conversions: {stats['supported_conversions']}")
    print("Available conversion paths:")
    for path in stats['conversion_paths']:
        print(f"  - {path}")
    
    # Test conversion validation
    test_cases = [
        ("data.csv", "data.xlsx"),
        ("document.docx", "document.pdf"), 
        ("file.txt", "file.pdf"),
        ("image.png", "image.jpg")
    ]
    
    print("\nConversion Validation Tests:")
    for input_file, output_file in test_cases:
        validation = validate_conversion_path(input_file, output_file)
        status = "✓" if validation['supported'] else "✗"
        print(f"  {status} {input_file} → {output_file}: {validation['supported']}")