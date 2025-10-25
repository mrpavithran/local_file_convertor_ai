"""
Enhanced PDF to DOCX converter with improved text extraction and formatting.
Dependencies: PyPDF2, python-docx, pdfminer.six (for better text extraction)
Note: For complex layouts, consider using specialized PDF conversion tools.
"""
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import re

# Try to import PyPDF2 (fallback to pypdf)
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Please install PyPDF2 or pypdf: pip install pypdf")

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

# Optional: pdfminer for better text extraction
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pdfminer.six not available. Install with: pip install pdfminer.six for better text extraction")

logger = logging.getLogger(__name__)

class PDFToDOCXConverter:
    """Enhanced PDF to DOCX converter with improved text extraction and formatting."""

    def __init__(self):
        self.default_font_size = 11
        self.default_font_name = "Calibri"
        self.line_spacing = 1.15
        self.margin_inches = 1.0

    def convert(self,
                input_path: str,
                output_path: Optional[str] = None,
                include_metadata: bool = True,
                preserve_formatting: bool = True,
                use_pdfminer: bool = True) -> Dict[str, any]:
        """
        Convert PDF file to DOCX with enhanced text extraction.
        
        Args:
            input_path: Path to input PDF file
            output_path: Path for output DOCX file (optional)
            include_metadata: Include basic metadata at the top
            preserve_formatting: Attempt to preserve basic formatting
            use_pdfminer: Use pdfminer for better text extraction (if available)
            
        Returns:
            Dictionary with conversion results and statistics
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if not input_path.lower().endswith('.pdf'):
            logger.warning(f"Input file '{input_path}' does not have .pdf extension")
        
        if output_path is None:
            output_path = self._generate_output_path(input_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        try:
            # Initialize DOCX document
            doc = Document()
            self._setup_document_styles(doc)
            
            # Read PDF
            reader = PdfReader(input_path)
            total_pages = len(reader.pages)
            
            if include_metadata:
                self._add_metadata(doc, input_path, reader)
            
            # Choose extraction method
            if use_pdfminer and PDFMINER_AVAILABLE:
                text_content = self._extract_text_with_pdfminer(input_path)
            else:
                text_content = self._extract_text_with_pypdf(reader)
            
            # Add content to document
            conversion_stats = self._add_content_to_doc(doc, text_content, preserve_formatting)
            
            # Save document
            doc.save(output_path)
            
            # Generate statistics
            stats = self._get_conversion_stats(input_path, output_path, total_pages, conversion_stats)
            
            logger.info(f"Successfully converted '{input_path}' to '{output_path}'")
            return stats
        
        except Exception as e:
            # Clean up failed output file
            if os.path.exists(output_path):
                os.remove(output_path)
            raise Exception(f"Conversion failed: {str(e)}") from e

    def _setup_document_styles(self, doc: Document):
        """Setup default document styles and margins."""
        # Set margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(self.margin_inches)
            section.bottom_margin = Inches(self.margin_inches)
            section.left_margin = Inches(self.margin_inches)
            section.right_margin = Inches(self.margin_inches)

    def _extract_text_with_pypdf(self, reader: PdfReader) -> List[Dict[str, any]]:
        """Extract text using PyPDF2 with basic structure."""
        content = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                # Basic structure detection
                lines = text.split('\n')
                for line_num, line in enumerate(lines, start=1):
                    if line.strip():
                        # Simple heading detection (all caps, ends with no period)
                        is_heading = (line.isupper() and 
                                    len(line) < 100 and 
                                    not line.endswith('.') and
                                    not line.endswith(','))
                        
                        content.append({
                            'page': page_num,
                            'line': line_num,
                            'text': line.strip(),
                            'is_heading': is_heading,
                            'font_size': self.default_font_size,  # PyPDF2 doesn't provide font info
                            'is_bold': False
                        })
        
        return content

    def _extract_text_with_pdfminer(self, input_path: str) -> List[Dict[str, any]]:
        """Extract text with formatting using pdfminer (if available)."""
        if not PDFMINER_AVAILABLE:
            return self._extract_text_with_pypdf(PdfReader(input_path))
        
        content = []
        
        try:
            for page_num, page_layout in enumerate(extract_pages(input_path), start=1):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        # Extract text with formatting
                        text = element.get_text().strip()
                        if text:
                            # Analyze text properties
                            font_size = self.default_font_size
                            is_bold = False
                            
                            # Try to get font information from characters
                            if hasattr(element, '__iter__'):
                                for text_line in element:
                                    if hasattr(text_line, '__iter__'):
                                        for character in text_line:
                                            if isinstance(character, LTChar):
                                                font_size = max(font_size, round(character.size, 1))
                                                font_name = character.fontname.lower()
                                                if 'bold' in font_name:
                                                    is_bold = True
                                                break
                                    break
                            
                            # Detect headings based on font size and text characteristics
                            is_heading = (font_size > self.default_font_size + 2 or 
                                        (len(text) < 100 and 
                                         text.isupper() and 
                                         not text.endswith('.') and
                                         not text.endswith(',')))
                            
                            content.append({
                                'page': page_num,
                                'text': text,
                                'font_size': font_size,
                                'is_bold': is_bold,
                                'is_heading': is_heading
                            })
        
        except Exception as e:
            logger.warning(f"pdfminer extraction failed, falling back to PyPDF2: {e}")
            # Fallback to PyPDF2
            reader = PdfReader(input_path)
            return self._extract_text_with_pypdf(reader)
        
        return content

    def _add_content_to_doc(self, doc: Document, content: List[Dict], preserve_formatting: bool) -> Dict[str, int]:
        """Add extracted content to DOCX document with formatting."""
        stats = {
            'paragraphs': 0,
            'headings': 0,
            'pages': len(set(item['page'] for item in content)) if content else 0
        }
        
        current_page = 0
        
        for item in content:
            # Add page break if needed
            if preserve_formatting and item['page'] != current_page and current_page > 0:
                doc.add_page_break()
                stats['pages'] += 1
            current_page = item['page']
            
            text = item['text']
            if not text.strip():
                continue
            
            # Create paragraph with appropriate style
            if preserve_formatting and item.get('is_heading', False):
                para = doc.add_paragraph(text, style='Heading 1')
                stats['headings'] += 1
            else:
                para = doc.add_paragraph(text)
                stats['paragraphs'] += 1
            
            # Apply formatting if available and requested
            if preserve_formatting:
                font_size = item.get('font_size', self.default_font_size)
                is_bold = item.get('is_bold', False)
                
                # Adjust font size
                if font_size != self.default_font_size:
                    for run in para.runs:
                        run.font.size = Pt(min(font_size, 36))  # Cap at 36pt
                
                # Apply bold formatting
                if is_bold:
                    for run in para.runs:
                        run.bold = True
            
            # Set line spacing
            para.paragraph_format.line_spacing = self.line_spacing
        
        return stats

    def _add_metadata(self, doc: Document, input_path: str, reader: PdfReader):
        """Add comprehensive metadata to DOCX document."""
        title = doc.add_paragraph()
        title_run = title.add_run("PDF Conversion Report")
        title_run.bold = True
        title_run.font.size = Pt(14)
        
        # Basic file info
        doc.add_paragraph(f"Source PDF: {os.path.basename(input_path)}")
        doc.add_paragraph(f"File size: {self._format_file_size(os.path.getsize(input_path))}")
        doc.add_paragraph(f"Conversion date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # PDF metadata
        if reader.metadata:
            meta_para = doc.add_paragraph("PDF Metadata:")
            for key, value in reader.metadata.items():
                if value and str(value).strip():
                    doc.add_paragraph(f"  {key}: {value}")
        
        doc.add_paragraph("")  # Add spacing
        doc.add_paragraph("Converted Content:")
        doc.add_paragraph("")  # Add spacing

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _generate_output_path(self, input_path: str) -> str:
        """Generate output path with conflict resolution."""
        base_name = os.path.splitext(input_path)[0]
        counter = 1
        output_path = f"{base_name}.docx"
        
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.docx"
            counter += 1
        
        return output_path

    def _get_conversion_stats(self, input_path: str, output_path: str, 
                            total_pages: int, conversion_stats: Dict) -> Dict[str, any]:
        """Generate comprehensive conversion statistics."""
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        return {
            "output_path": os.path.abspath(output_path),
            "input_file": os.path.basename(input_path),
            "output_file": os.path.basename(output_path),
            "pages_converted": total_pages,
            "paragraphs_created": conversion_stats.get('paragraphs', 0),
            "headings_detected": conversion_stats.get('headings', 0),
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "input_size_human": self._format_file_size(input_size),
            "output_size_human": self._format_file_size(output_size),
            "size_change_percent": round(((output_size - input_size) / input_size) * 100, 2) if input_size > 0 else 0,
            "conversion_success": True,
            "timestamp": datetime.now().isoformat(),
            "pdfminer_used": PDFMINER_AVAILABLE
        }

    def batch_convert(self,
                      input_directory: str,
                      output_directory: Optional[str] = None,
                      recursive: bool = False,
                      **kwargs) -> Dict[str, any]:
        """Convert all PDFs in a directory with enhanced options."""
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory not found: {input_directory}")
        
        output_directory = output_directory or input_directory
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(Path(input_directory).glob(pattern))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {input_directory}")
        
        results = {
            "total_files": len(pdf_files),
            "successful_conversions": 0,
            "failed_conversions": 0,
            "conversions": [],
            "total_pages": 0,
            "total_paragraphs": 0
        }
        
        for pdf_file in pdf_files:
            try:
                relative_path = pdf_file.relative_to(input_directory)
                output_path = Path(output_directory) / relative_path.with_suffix('.docx')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                result = self.convert(str(pdf_file), str(output_path), **kwargs)
                results["successful_conversions"] += 1
                results["total_pages"] += result.get("pages_converted", 0)
                results["total_paragraphs"] += result.get("paragraphs_created", 0)
                
                results["conversions"].append({
                    "input_file": str(pdf_file),
                    "output_file": str(output_path),
                    "success": True,
                    "stats": result
                })
                
            except Exception as e:
                results["failed_conversions"] += 1
                results["conversions"].append({
                    "input_file": str(pdf_file),
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"Failed to convert '{pdf_file}': {e}")
        
        return results


# MCP Tool class
class Tool:
    """MCP-compatible PDF to DOCX conversion tool."""
    
    name = "convert_pdf_to_docx"
    description = "Convert PDF files to DOCX format with enhanced text extraction"
    
    def __init__(self):
        self.converter = PDFToDOCXConverter()
    
    def run(self, 
            input_path: str, 
            output_path: str = None,
            preserve_formatting: bool = True,
            use_advanced_extraction: bool = True) -> Dict[str, any]:
        """
        Convert PDF to DOCX with enhanced options.
        
        Args:
            input_path: Path to input PDF file
            output_path: Output DOCX path (optional)
            preserve_formatting: Attempt to preserve formatting
            use_advanced_extraction: Use pdfminer if available
            
        Returns:
            Conversion results and statistics
        """
        return self.converter.convert(
            input_path=input_path,
            output_path=output_path,
            preserve_formatting=preserve_formatting,
            use_pdfminer=use_advanced_extraction
        )


# Command-line interface
def main():
    """Enhanced command-line interface."""
    if len(sys.argv) < 2 or '--help' in sys.argv:
        print("""
Enhanced PDF to DOCX Converter

Usage:
  python pdf_to_docx.py <input.pdf> [<output.docx>] [options]
  python pdf_to_docx.py --batch <directory> [options]
  
Options:
  --no-formatting          Disable formatting preservation
  --simple-extraction      Use simple text extraction (faster)
  --batch DIR              Convert all PDFs in directory
  --output-dir DIR         Output directory for batch conversion
  --recursive              Include subdirectories in batch mode
  
Examples:
  python pdf_to_docx.py document.pdf
  python pdf_to_docx.py document.pdf output.docx --no-formatting
  python pdf_to_docx.py --batch ./pdfs --output-dir ./docxs --recursive
        """)
        sys.exit(1)
    
    converter = PDFToDOCXConverter()
    
    if '--batch' in sys.argv:
        # Batch conversion mode
        input_dir = sys.argv[sys.argv.index('--batch') + 1]
        output_dir = None
        recursive = '--recursive' in sys.argv
        preserve_formatting = '--no-formatting' not in sys.argv
        use_advanced = '--simple-extraction' not in sys.argv
        
        if '--output-dir' in sys.argv:
            output_dir = sys.argv[sys.argv.index('--output-dir') + 1]
        
        results = converter.batch_convert(
            input_dir, output_dir, recursive,
            preserve_formatting=preserve_formatting,
            use_pdfminer=use_advanced
        )
        
        print(f"Batch conversion complete:")
        print(f"  Successful: {results['successful_conversions']}/{results['total_files']}")
        print(f"  Total pages: {results['total_pages']}")
        print(f"  Total paragraphs: {results['total_paragraphs']}")
        
    else:
        # Single file conversion
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        preserve_formatting = '--no-formatting' not in sys.argv
        use_advanced = '--simple-extraction' not in sys.argv
        
        result = converter.convert(
            input_path=input_path,
            output_path=output_path,
            preserve_formatting=preserve_formatting,
            use_pdfminer=use_advanced
        )
        
        print(f"Conversion successful: {result['output_path']}")
        print(f"Pages: {result['pages_converted']}")
        print(f"Paragraphs: {result['paragraphs_created']}")
        print(f"Headings: {result['headings_detected']}")
        print(f"Size: {result['input_size_human']} → {result['output_size_human']}")


if __name__ == "__main__":
    main()