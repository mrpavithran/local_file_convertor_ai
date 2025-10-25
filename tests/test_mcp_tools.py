"""
Tests for MCP tools across all categories.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MCP tools from all categories
try:
    from ai_infrastructure.mcp.tools.file_tools.file_detector import FileDetector
    from ai_infrastructure.mcp.tools.file_tools.file_info import FileInfoTool
    from ai_infrastructure.mcp.tools.file_tools.directory_scanner import DirectoryScanner
    
    from ai_infrastructure.mcp.tools.conversion_tools.convert_docx_to_pdf import DocxToPdfConverter
    from ai_infrastructure.mcp.tools.conversion_tools.convert_csv_to_xlsx import CsvToXlsxConverter
    from ai_infrastructure.mcp.tools.conversion_tools.batch_converter import BatchConverter
    
    from ai_infrastructure.mcp.tools.image_tools.ocr_tool import OCRTool
    from ai_infrastructure.mcp.tools.image_tools.image_info import ImageInfoTool
    from ai_infrastructure.mcp.tools.image_tools.image_upscaler import ImageUpscaler
    
    from ai_infrastructure.mcp.tools.text_tools.text_summarizer import TextSummarizer
    from ai_infrastructure.mcp.tools.text_tools.text_translator import TextTranslator
    from ai_infrastructure.mcp.tools.text_tools.text_analyzer import TextAnalyzer
    
    from ai_infrastructure.mcp.tools.web_tools.web_scraper import WebScraper
    from ai_infrastructure.mcp.tools.web_tools.url_validator import URLValidator
    from ai_infrastructure.mcp.tools.web_tools.http_status_checker import HTTPStatusChecker
    
    from ai_infrastructure.mcp.tools.system_tools.system_info import SystemInfoTool
    from ai_infrastructure.mcp.tools.system_tools.disk_usage import DiskUsageTool
    from ai_infrastructure.mcp.tools.system_tools.process_checker import ProcessChecker
    
    MCP_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some MCP tools not available for testing: {e}")
    MCP_TOOLS_AVAILABLE = False


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestFileTools:
    """Test file system tools."""
    
    @pytest.fixture
    def file_detector(self):
        """Create a FileDetector instance."""
        return FileDetector()
    
    @pytest.fixture
    def file_info_tool(self):
        """Create a FileInfoTool instance."""
        return FileInfoTool()
    
    @pytest.fixture
    def directory_scanner(self):
        """Create a DirectoryScanner instance."""
        return DirectoryScanner()
    
    @pytest.fixture
    def sample_files(self):
        """Create sample files for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create various file types
        files = {
            'document.txt': 'Text document content',
            'code.py': 'import os\nprint("test")',
            'data.csv': 'name,age\nJohn,30\nJane,25',
            'empty.txt': '',
        }
        
        for filename, content in files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding='utf-8')
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_file_detection(self, file_detector, sample_files):
        """Test file type detection tool."""
        test_file = Path(sample_files) / 'document.txt'
        
        result = file_detector.execute({'file_path': str(test_file)})
        
        assert 'file_type' in result
        assert 'category' in result
        assert 'mime_type' in result
        assert result['file_type'] == 'text'
    
    def test_file_info_extraction(self, file_info_tool, sample_files):
        """Test file information extraction tool."""
        test_file = Path(sample_files) / 'document.txt'
        
        result = file_info_tool.execute({'file_path': str(test_file)})
        
        assert 'name' in result
        assert 'size_bytes' in result
        assert 'size_human' in result
        assert 'created' in result
        assert 'modified' in result
        assert result['name'] == 'document.txt'
    
    def test_directory_scanning(self, directory_scanner, sample_files):
        """Test directory scanning tool."""
        result = directory_scanner.execute({
            'directory_path': sample_files,
            'recursive': False
        })
        
        assert 'files_found' in result
        assert 'categories' in result
        assert 'total_size' in result
        assert result['files_found'] > 0


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestConversionTools:
    """Test file conversion tools."""
    
    @pytest.fixture
    def docx_to_pdf_converter(self):
        """Create a DocxToPdfConverter instance."""
        return DocxToPdfConverter()
    
    @pytest.fixture
    def csv_to_xlsx_converter(self):
        """Create a CsvToXlsxConverter instance."""
        return CsvToXlsxConverter()
    
    @pytest.fixture
    def batch_converter(self):
        """Create a BatchConverter instance."""
        return BatchConverter()
    
    @pytest.fixture
    def sample_docx_file(self):
        """Create a sample DOCX file for testing."""
        # For testing, we'll create a simple text file and mock DOCX conversion
        temp_dir = tempfile.mkdtemp()
        docx_path = Path(temp_dir) / 'test.docx'
        docx_path.write_text('Test DOCX content')
        
        yield str(docx_path)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / 'test.csv'
        csv_content = """name,age,city
John,30,New York
Jane,25,London
Bob,35,Tokyo"""
        csv_path.write_text(csv_content)
        
        yield str(csv_path)
        shutil.rmtree(temp_dir)
    
    @patch('ai_infrastructure.mcp.tools.conversion_tools.convert_docx_to_pdf.docx2pdf')
    def test_docx_to_pdf_conversion(self, mock_docx2pdf, docx_to_pdf_converter, sample_docx_file):
        """Test DOCX to PDF conversion tool."""
        # Mock successful conversion
        mock_docx2pdf.return_value = None
        
        result = docx_to_pdf_converter.execute({
            'input_file': sample_docx_file,
            'output_file': sample_docx_file.replace('.docx', '.pdf')
        })
        
        assert 'success' in result
        assert 'output_file' in result
        assert result['success'] is True
    
    def test_csv_to_xlsx_conversion(self, csv_to_xlsx_converter, sample_csv_file):
        """Test CSV to XLSX conversion tool."""
        output_file = sample_csv_file.replace('.csv', '.xlsx')
        
        result = csv_to_xlsx_converter.execute({
            'input_file': sample_csv_file,
            'output_file': output_file
        })
        
        # Check if conversion was successful
        assert 'success' in result
        assert 'output_file' in result
        
        # Cleanup
        if Path(output_file).exists():
            Path(output_file).unlink()
    
    def test_batch_conversion(self, batch_converter, sample_csv_file):
        """Test batch conversion tool."""
        temp_dir = Path(sample_csv_file).parent
        
        # Create multiple CSV files
        csv_files = []
        for i in range(3):
            csv_file = temp_dir / f'test_{i}.csv'
            csv_file.write_text('name,value\ntest,123')
            csv_files.append(str(csv_file))
        
        result = batch_converter.execute({
            'input_files': csv_files,
            'output_format': 'xlsx',
            'output_directory': str(temp_dir / 'converted')
        })
        
        assert 'processed' in result
        assert 'successful' in result
        assert 'failed' in result
        assert result['processed'] == len(csv_files)


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestImageTools:
    """Test image processing tools."""
    
    @pytest.fixture
    def ocr_tool(self):
        """Create an OCRTool instance."""
        return OCRTool()
    
    @pytest.fixture
    def image_info_tool(self):
        """Create an ImageInfoTool instance."""
        return ImageInfoTool()
    
    @pytest.fixture
    def image_upscaler(self):
        """Create an ImageUpscaler instance."""
        return ImageUpscaler()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image file for testing."""
        from PIL import Image
        import io
        
        temp_dir = tempfile.mkdtemp()
        image_path = Path(temp_dir) / 'test.png'
        
        # Create a simple image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
        
        yield str(image_path)
        shutil.rmtree(temp_dir)
    
    @patch('pytesseract.image_to_string')
    def test_ocr_functionality(self, mock_ocr, ocr_tool, sample_image):
        """Test OCR tool functionality."""
        # Mock OCR result
        mock_ocr.return_value = "Extracted text from image"
        
        result = ocr_tool.execute({
            'image_path': sample_image,
            'language': 'eng'
        })
        
        assert 'extracted_text' in result
        assert 'confidence' in result
        assert result['extracted_text'] == "Extracted text from image"
    
    def test_image_info_extraction(self, image_info_tool, sample_image):
        """Test image information extraction."""
        result = image_info_tool.execute({
            'image_path': sample_image
        })
        
        assert 'format' in result
        assert 'dimensions' in result
        assert 'mode' in result
        assert 'size_bytes' in result
        assert 'width' in result
        assert 'height' in result
    
    @patch('ai_infrastructure.mcp.tools.image_tools.image_upscaler.ISR')
    def test_image_upscaling(self, mock_isr, image_upscaler, sample_image):
        """Test image upscaling tool."""
        # Mock ISR upscaling
        mock_model = MagicMock()
        mock_isr.return_value = mock_model
        
        mock_image = MagicMock()
        mock_model.upscale.return_value = mock_image
        
        result = image_upscaler.execute({
            'image_path': sample_image,
            'scale_factor': 2,
            'output_path': sample_image.replace('.png', '_upscaled.png')
        })
        
        assert 'success' in result
        assert 'original_dimensions' in result
        assert 'upscaled_dimensions' in result
        assert 'scale_factor' in result


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestTextTools:
    """Test text processing tools."""
    
    @pytest.fixture
    def text_summarizer(self):
        """Create a TextSummarizer instance."""
        return TextSummarizer()
    
    @pytest.fixture
    def text_translator(self):
        """Create a TextTranslator instance."""
        return TextTranslator()
    
    @pytest.fixture
    def text_analyzer(self):
        """Create a TextAnalyzer instance."""
        return TextAnalyzer()
    
    @pytest.fixture
    def sample_text(self):
        """Create sample text for testing."""
        return """
        Artificial intelligence is transforming many industries. 
        Machine learning algorithms can now process vast amounts of data 
        and make predictions with remarkable accuracy. 
        Natural language processing enables computers to understand 
        and generate human language. These technologies are creating 
        new opportunities for innovation and efficiency.
        """
    
    @patch('ai_infrastructure.mcp.tools.text_tools.text_summarizer.PromptExecutor')
    def test_text_summarization(self, mock_prompt_executor, text_summarizer, sample_text):
        """Test text summarization tool."""
        # Mock AI response
        mock_executor = MagicMock()
        mock_prompt_executor.return_value = mock_executor
        mock_executor.generate_completion.return_value = {
            'response': 'This is a summary of the text about AI technologies.'
        }
        
        result = text_summarizer.execute({
            'text': sample_text,
            'max_length': 100
        })
        
        assert 'summary' in result
        assert 'original_length' in result
        assert 'summary_length' in result
        assert 'compression_ratio' in result
    
    @patch('ai_infrastructure.mcp.tools.text_tools.text_translator.PromptExecutor')
    def test_text_translation(self, mock_prompt_executor, text_translator, sample_text):
        """Test text translation tool."""
        # Mock AI response
        mock_executor = MagicMock()
        mock_prompt_executor.return_value = mock_executor
        mock_executor.generate_completion.return_value = {
            'response': 'Translated text in Spanish'
        }
        
        result = text_translator.execute({
            'text': sample_text,
            'target_language': 'Spanish'
        })
        
        assert 'translated_text' in result
        assert 'source_language' in result
        assert 'target_language' in result
        assert result['target_language'] == 'Spanish'
    
    def test_text_analysis(self, text_analyzer, sample_text):
        """Test text analysis tool."""
        result = text_analyzer.execute({
            'text': sample_text
        })
        
        assert 'word_count' in result
        assert 'sentence_count' in result
        assert 'character_count' in result
        assert 'reading_time' in result
        assert 'keyword_density' in result
        assert 'sentiment' in result


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestWebTools:
    """Test web-related tools."""
    
    @pytest.fixture
    def web_scraper(self):
        """Create a WebScraper instance."""
        return WebScraper()
    
    @pytest.fixture
    def url_validator(self):
        """Create a URLValidator instance."""
        return URLValidator()
    
    @pytest.fixture
    def http_status_checker(self):
        """Create an HTTPStatusChecker instance."""
        return HTTPStatusChecker()
    
    @patch('requests.get')
    def test_web_scraping(self, mock_get, web_scraper):
        """Test web scraping tool."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '''
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Welcome to Test Page</h1>
                <p>This is a test paragraph.</p>
            </body>
        </html>
        '''
        mock_get.return_value = mock_response
        
        result = web_scraper.execute({
            'url': 'https://example.com',
            'extract': ['title', 'headings', 'paragraphs']
        })
        
        assert 'title' in result
        assert 'headings' in result
        assert 'paragraphs' in result
        assert 'status_code' in result
        assert result['status_code'] == 200
    
    def test_url_validation(self, url_validator):
        """Test URL validation tool."""
        test_cases = [
            ('https://example.com', True),
            ('http://localhost:8000', True),
            ('not-a-url', False),
            ('ftp://example.com', True),  # FTP is valid URL
        ]
        
        for url, expected_valid in test_cases:
            result = url_validator.execute({'url': url})
            assert result['is_valid'] == expected_valid
            assert 'protocol' in result
            assert 'domain' in result
    
    @patch('requests.head')
    def test_http_status_checking(self, mock_head, http_status_checker):
        """Test HTTP status checking tool."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response
        
        result = http_status_checker.execute({
            'url': 'https://example.com'
        })
        
        assert 'status_code' in result
        assert 'status_message' in result
        assert 'is_success' in result
        assert 'response_time' in result
        assert result['status_code'] == 200
        assert result['is_success'] is True


@pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
class TestSystemTools:
    """Test system utility tools."""
    
    @pytest.fixture
    def system_info_tool(self):
        """Create a SystemInfoTool instance."""
        return SystemInfoTool()
    
    @pytest.fixture
    def disk_usage_tool(self):
        """Create a DiskUsageTool instance."""
        return DiskUsageTool()
    
    @pytest.fixture
    def process_checker(self):
        """Create a ProcessChecker instance."""
        return ProcessChecker()
    
    def test_system_info(self, system_info_tool):
        """Test system information tool."""
        result = system_info_tool.execute({})
        
        assert 'platform' in result
        assert 'python_version' in result
        assert 'hostname' in result
        assert 'cpu_count' in result
        assert 'total_memory' in result
        assert 'available_memory' in result
    
    def test_disk_usage(self, disk_usage_tool):
        """Test disk usage analysis tool."""
        result = disk_usage_tool.execute({
            'path': '/'
        })
        
        assert 'total' in result
        assert 'used' in result
        assert 'free' in result
        assert 'percent_used' in result
        assert all(isinstance(value, (int, float)) for value in [
            result['total'], result['used'], result['free'], result['percent_used']
        ])
    
    def test_process_checking(self, process_checker):
        """Test process monitoring tool."""
        result = process_checker.execute({
            'process_name': 'python'
        })
        
        assert 'processes_found' in result
        assert 'processes' in result
        assert isinstance(result['processes'], list)


class TestToolIntegration:
    """Integration tests for MCP tools."""
    
    @pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
    def test_tool_chaining(self):
        """Test chaining multiple tools together."""
        # This would test executing a sequence of tools
        # e.g., file scan -> document conversion -> text analysis
        pass
    
    @pytest.mark.skipif(not MCP_TOOLS_AVAILABLE, reason="MCP tools not available")
    def test_error_handling_across_tools(self):
        """Test error handling when tools fail."""
        # Test how the system handles tool execution errors
        # and continues processing with other tools
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])