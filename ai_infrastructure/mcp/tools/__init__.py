"""
MCP Tools Package
Fixed imports to resolve ModuleNotFoundError
"""

# Remove the problematic import that's causing the error
# from .tool_registry import ToolRegistry  # ← THIS LINE IS CAUSING THE ERROR

try:
    from .conversion_tools.convert_pdf_to_docx import PDFToDOCXConverter
except ImportError as e:
    print(f"Warning: PDFToDOCXConverter not available: {e}")
    # Create a placeholder class
    class PDFToDOCXConverter:
        def __init__(self):
            pass
        def convert(self, input_path, output_path=None):
            return {"success": False, "error": "Converter not available"}

# Import other conversion tools if they exist
try:
    from .conversion_tools.convert_docx_to_pdf import DOCXToPDFConverter
except ImportError:
    class DOCXToPDFConverter:
        def __init__(self): pass

try:
    from .conversion_tools.convert_image_format import ImageFormatConverter
except ImportError:
    class ImageFormatConverter:
        def __init__(self): pass

try:
    from .conversion_tools.convert_txt_to_pdf import TXTToPDFConverter
except ImportError:
    class TXTToPDFConverter:
        def __init__(self): pass

try:
    from .conversion_tools.convert_html_to_pdf import HTMLToPDFConverter
except ImportError:
    class HTMLToPDFConverter:
        def __init__(self): pass

__all__ = [
    'PDFToDOCXConverter',
    'DOCXToPDFConverter', 
    'ImageFormatConverter',
    'TXTToPDFConverter',
    'HTMLToPDFConverter'
]