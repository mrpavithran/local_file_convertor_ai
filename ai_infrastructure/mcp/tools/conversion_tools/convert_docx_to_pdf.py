"""
Enhanced DOCX to PDF converter with improved formatting preservation.
Dependencies: python-docx, reportlab
Note: For complex formatting, consider using LibreOffice in headless mode or cloud services.
"""
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Set up logging
logger = logging.getLogger(__name__)

class DOCXToPDFConverter:
    """Enhanced DOCX to PDF converter with formatting preservation."""
    
    def __init__(self):
        self.supported_page_sizes = {
            'letter': letter,
            'a4': A4
        }
        self._register_fonts()
    
    def _register_fonts(self):
        """Register common fonts for better text rendering."""
        try:
            # Try to register common fonts - these might need to be installed on the system
            font_mappings = [
                ('Helvetica', 'Helvetica'),
                ('Times-Roman', 'Times-Roman'),
                ('Courier', 'Courier'),
            ]
            
            for font_name, font_path in font_mappings:
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                except:
                    pass  # Font might not be available
        except Exception as e:
            logger.warning(f"Font registration failed: {e}")

    def convert(self, 
                input_path: str, 
                output_path: Optional[str] = None,
                page_size: str = 'letter',
                preserve_formatting: bool = True,
                include_metadata: bool = True) -> Dict[str, any]:
        """
        Convert DOCX file to PDF with enhanced formatting.
        
        Args:
            input_path: Path to input DOCX file
            output_path: Path for output PDF file (optional)
            page_size: Page size ('letter' or 'a4')
            preserve_formatting: Whether to preserve basic formatting
            include_metadata: Whether to include document metadata
            
        Returns:
            Dictionary with conversion results and statistics
        """
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if not input_path.lower().endswith('.docx'):
            logger.warning(f"Input file '{input_path}' does not have .docx extension")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_output_path(input_path)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Read DOCX document
            doc = Document(input_path)
            
            # Get page size
            pdf_page_size = self.supported_page_sizes.get(page_size.lower(), letter)
            
            # Create PDF document
            pdf_doc = SimpleDocTemplate(
                output_path,
                pagesize=pdf_page_size,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build story (content)
            story = []
            
            # Add metadata if requested
            if include_metadata:
                self._add_metadata(story, doc, input_path)
            
            # Convert content
            conversion_stats = self._convert_content(doc, story, preserve_formatting)
            
            # Build PDF
            pdf_doc.build(story)
            
            # Generate result statistics
            stats = self._get_conversion_stats(input_path, output_path, conversion_stats)
            
            logger.info(f"Successfully converted '{input_path}' to '{output_path}'")
            return stats
            
        except Exception as e:
            # Clean up failed output file
            if os.path.exists(output_path):
                os.remove(output_path)
            raise Exception(f"Conversion failed: {str(e)}") from e

    def _convert_content(self, doc: Document, story: List, preserve_formatting: bool) -> Dict[str, int]:
        """Convert DOCX content to PDF elements."""
        stats = {
            'paragraphs': 0,
            'tables': 0,
            'images': 0  # Note: Basic implementation doesn't handle images
        }
        
        styles = getSampleStyleSheet()
        
        for element in doc.element.body:
            if element.tag.endswith('p'):  # Paragraph
                paragraph = self._convert_paragraph(element, styles, preserve_formatting)
                if paragraph:
                    story.append(paragraph)
                    story.append(Spacer(1, 12))  # Add spacing between paragraphs
                    stats['paragraphs'] += 1
            
            elif element.tag.endswith('tbl'):  # Table
                table = self._convert_table(element, preserve_formatting)
                if table:
                    story.append(table)
                    story.append(Spacer(1, 12))
                    stats['tables'] += 1
        
        return stats

    def _convert_paragraph(self, element, styles, preserve_formatting: bool) -> Optional[Paragraph]:
        """Convert a paragraph element to PDF paragraph."""
        try:
            text = ''
            runs = element.xpath('.//w:r', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
            
            for run in runs:
                run_text = ''.join(node.text for node in run.xpath('.//w:t', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}) if node.text)
                
                if preserve_formatting:
                    # Basic formatting preservation
                    if run.xpath('.//w:b', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                        run_text = f"<b>{run_text}</b>"
                    if run.xpath('.//w:i', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                        run_text = f"<i>{run_text}</i>"
                    if run.xpath('.//w:u', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                        run_text = f"<u>{run_text}</u>"
                
                text += run_text
            
            if text.strip():
                # Determine paragraph style based on formatting
                style = styles['Normal']
                
                # Check for headings
                style_elem = element.xpath('.//w:pStyle', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
                if style_elem:
                    style_val = style_elem[0].get(qn('w:val'))
                    if style_val == 'Heading1':
                        style = styles['Heading1']
                    elif style_val == 'Heading2':
                        style = styles['Heading2']
                    elif style_val == 'Heading3':
                        style = styles['Heading3']
                
                return Paragraph(text, style)
            
        except Exception as e:
            logger.warning(f"Failed to convert paragraph: {e}")
        
        return None

    def _convert_table(self, element, preserve_formatting: bool) -> Optional[Table]:
        """Convert a table element to PDF table."""
        try:
            rows = element.xpath('.//w:tr', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
            table_data = []
            
            for row in rows:
                row_data = []
                cells = row.xpath('.//w:tc', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
                
                for cell in cells:
                    cell_text = ''
                    texts = cell.xpath('.//w:t', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
                    for text_elem in texts:
                        if text_elem.text:
                            cell_text += text_elem.text
                    row_data.append(cell_text.strip())
                
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                return table
        
        except Exception as e:
            logger.warning(f"Failed to convert table: {e}")
        
        return None

    def _add_metadata(self, story: List, doc: Document, input_path: str):
        """Add document metadata to PDF."""
        styles = getSampleStyleSheet()
        
        # Title
        if doc.core_properties.title:
            story.append(Paragraph(f"<b>Title:</b> {doc.core_properties.title}", styles['Normal']))
        
        # Author
        if doc.core_properties.author:
            story.append(Paragraph(f"<b>Author:</b> {doc.core_properties.author}", styles['Normal']))
        
        # Source file
        story.append(Paragraph(f"<b>Source:</b> {os.path.basename(input_path)}", styles['Normal']))
        
        # Conversion timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"<b>Converted:</b> {timestamp}", styles['Normal']))
        
        story.append(Spacer(1, 24))  # Add space before content

    def _generate_output_path(self, input_path: str) -> str:
        """Generate output path from input path."""
        base_name = os.path.splitext(input_path)[0]
        counter = 1
        output_path = f"{base_name}.pdf"
        
        # Avoid overwriting existing files
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.pdf"
            counter += 1
        
        return output_path

    def _get_conversion_stats(self, input_path: str, output_path: str, conversion_stats: Dict) -> Dict[str, any]:
        """Generate conversion statistics."""
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        return {
            "output_path": os.path.abspath(output_path),
            "input_file": os.path.basename(input_path),
            "output_file": os.path.basename(output_path),
            "conversion_stats": conversion_stats,
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "size_change_percent": round(((output_size - input_size) / input_size) * 100, 2) if input_size > 0 else 0,
            "conversion_success": True,
            "timestamp": datetime.now().isoformat()
        }

    def batch_convert(self, 
                     input_directory: str, 
                     output_directory: Optional[str] = None,
                     recursive: bool = False,
                     **kwargs) -> Dict[str, any]:
        """
        Convert multiple DOCX files in a directory.
        
        Args:
            input_directory: Directory containing DOCX files
            output_directory: Output directory (optional)
            recursive: Whether to search subdirectories
            **kwargs: Additional conversion arguments
            
        Returns:
            Dictionary with batch conversion results
        """
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory not found: {input_directory}")
        
        output_directory = output_directory or input_directory
        
        # Find DOCX files
        pattern = "**/*.docx" if recursive else "*.docx"
        docx_files = list(Path(input_directory).glob(pattern))
        
        if not docx_files:
            raise FileNotFoundError(f"No DOCX files found in: {input_directory}")
        
        results = {
            "total_files": len(docx_files),
            "successful_conversions": 0,
            "failed_conversions": 0,
            "conversions": []
        }
        
        for docx_file in docx_files:
            try:
                relative_path = docx_file.relative_to(input_directory)
                output_path = Path(output_directory) / relative_path.with_suffix('.pdf')
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert file
                result = self.convert(
                    input_path=str(docx_file),
                    output_path=str(output_path),
                    **kwargs
                )
                
                results["successful_conversions"] += 1
                results["conversions"].append({
                    "input_file": str(docx_file),
                    "output_file": str(output_path),
                    "success": True,
                    "stats": result
                })
                
            except Exception as e:
                results["failed_conversions"] += 1
                results["conversions"].append({
                    "input_file": str(docx_file),
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"Failed to convert '{docx_file}': {e}")
        
        return results


# MCP Tool class
class Tool:
    """MCP-compatible tool for DOCX to PDF conversion."""
    
    name = "convert_docx_to_pdf"
    description = "Convert DOCX files to PDF format with basic formatting preservation"
    
    def __init__(self):
        self.converter = DOCXToPDFConverter()
    
    def run(self, 
            input_path: str, 
            output_path: str = None,
            page_size: str = 'letter',
            preserve_formatting: bool = True) -> Dict[str, any]:
        """
        Convert DOCX file to PDF format.
        
        Args:
            input_path: Path to the input DOCX file
            output_path: Path for the output PDF file (optional)
            page_size: Page size ('letter' or 'a4')
            preserve_formatting: Whether to preserve basic formatting
            
        Returns:
            Dictionary with conversion results
        """
        return self.converter.convert(
            input_path=input_path,
            output_path=output_path,
            page_size=page_size,
            preserve_formatting=preserve_formatting
        )


# Legacy function for backward compatibility
def convert_docx_to_pdf(input_path: str, output_path: str):
    """Legacy conversion function (basic version)."""
    converter = DOCXToPDFConverter()
    return converter.convert(input_path, output_path, preserve_formatting=False)


# Command-line interface
def main():
    """Command-line interface for DOCX to PDF conversion."""
    if len(sys.argv) < 2:
        print("""
DOCX to PDF Converter

Usage:
  python docx_to_pdf.py <input.docx> [<output.pdf>] [options]
  
Options:
  --page-size SIZE      Page size (letter or a4, default: letter)
  --no-formatting       Disable formatting preservation
  --batch DIR           Convert all DOCX files in directory
  --recursive           Include subdirectories in batch mode
  --output-dir DIR      Output directory for batch mode
  
Examples:
  python docx_to_pdf.py document.docx
  python docx_to_pdf.py document.docx output.pdf --page-size a4
  python docx_to_pdf.py --batch ./documents --output-dir ./pdfs --recursive
        """)
        sys.exit(1)
    
    converter = DOCXToPDFConverter()
    
    if '--batch' in sys.argv:
        # Batch conversion mode
        input_dir = sys.argv[sys.argv.index('--batch') + 1]
        output_dir = None
        recursive = '--recursive' in sys.argv
        
        if '--output-dir' in sys.argv:
            output_dir = sys.argv[sys.argv.index('--output-dir') + 1]
        
        results = converter.batch_convert(input_dir, output_dir, recursive)
        print(f"Batch conversion complete: {results['successful_conversions']}/{results['total_files']} successful")
        
    else:
        # Single file conversion
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Parse options
        page_size = 'letter'
        preserve_formatting = '--no-formatting' not in sys.argv
        
        if '--page-size' in sys.argv:
            page_size = sys.argv[sys.argv.index('--page-size') + 1]
        
        result = converter.convert(
            input_path=input_path,
            output_path=output_path,
            page_size=page_size,
            preserve_formatting=preserve_formatting
        )
        
        print(f"Conversion successful: {result['output_path']}")
        print(f"Statistics: {result['conversion_stats']}")


if __name__ == '__main__':
    main()