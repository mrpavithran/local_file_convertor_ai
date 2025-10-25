"""
Tests for file operations including management, type detection, and batch processing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from file_operations.file_manager import FileManager
from file_operations.type_detector import TypeDetector
from file_operations.batch_processor import BatchProcessor
from file_operations.output_manager import OutputManager
from file_operations.error_handler import ErrorHandler


class TestFileManager:
    """Test file management operations."""
    
    @pytest.fixture
    def file_manager(self):
        """Create a FileManager instance for testing."""
        return FileManager()
    
    @pytest.fixture
    def test_directory(self):
        """Create a temporary test directory with sample files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample files
        sample_files = {
            'test.txt': 'This is a text file for testing.',
            'test.py': 'print("Hello World")\n# Python test file',
            'test.jpg': b'fake_jpg_data',  # Will create as binary
            'test.pdf': b'%PDF-1.4 fake pdf data',
            'test.docx': b'fake_docx_data',
        }
        
        for filename, content in sample_files.items():
            file_path = Path(temp_dir) / filename
            if isinstance(content, str):
                file_path.write_text(content, encoding='utf-8')
            else:
                file_path.write_bytes(content)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_directory_scanning(self, file_manager, test_directory):
        """Test directory scanning functionality."""
        scan_results = file_manager.scan_directory(test_directory, recursive=False)
        
        assert isinstance(scan_results, dict)
        assert 'documents' in scan_results
        assert 'code' in scan_results
        assert 'images' in scan_results
        assert 'other' in scan_results
        
        # Should find our test files
        found_files = []
        for category in scan_results.values():
            found_files.extend([f.name for f in category])
        
        assert 'test.txt' in found_files
        assert 'test.py' in found_files
    
    def test_file_info_extraction(self, file_manager, test_directory):
        """Test file information extraction."""
        test_file = Path(test_directory) / 'test.txt'
        
        file_info = file_manager.get_file_info(test_file)
        
        assert file_info['name'] == 'test.txt'
        assert file_info['path'] == str(test_file)
        assert file_info['size_bytes'] > 0
        assert 'size_human' in file_info
        assert file_info['file_type'] == 'text'
        assert 'created' in file_info
        assert 'modified' in file_info
    
    def test_file_organization(self, file_manager, test_directory):
        """Test file organization by category."""
        target_dir = Path(test_directory) / 'organized'
        
        results = file_manager.organize_files(
            source_dir=test_directory,
            target_dir=target_dir,
            organization_strategy='category'
        )
        
        assert isinstance(results, dict)
        assert 'moved' in results
        assert 'skipped' in results
        assert 'errors' in results
        
        # Check if organized directories were created
        if target_dir.exists():
            category_dirs = [d.name for d in target_dir.iterdir() if d.is_dir()]
            assert len(category_dirs) > 0
    
    def test_batch_renaming(self, file_manager, test_directory):
        """Test batch file renaming."""
        operations = file_manager.batch_rename(
            directory=test_directory,
            pattern='test',
            replacement='renamed',
            dry_run=True  # Don't actually rename files
        )
        
        assert isinstance(operations, list)
        
        # Should find operations for files matching pattern
        matching_files = [f for f in Path(test_directory).iterdir() 
                         if f.is_file() and 'test' in f.name]
        
        if matching_files:
            assert len(operations) > 0
            for op in operations:
                assert 'old_name' in op
                assert 'new_name' in op
                assert 'status' in op


class TestTypeDetector:
    """Test file type detection."""
    
    @pytest.fixture
    def type_detector(self):
        """Create a TypeDetector instance for testing."""
        return TypeDetector()
    
    @pytest.fixture
    def sample_files(self):
        """Create sample files for type detection testing."""
        temp_dir = tempfile.mkdtemp()
        
        files = {
            'text.txt': 'Plain text file content',
            'python.py': 'import os\nprint("test")',
            'javascript.js': 'console.log("test");',
            'html.html': '<html><body>Test</body></html>',
            'markdown.md': '# Test Markdown',
        }
        
        for filename, content in files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding='utf-8')
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_extension_detection(self, type_detector, sample_files):
        """Test file type detection by extension."""
        test_cases = [
            ('text.txt', 'text'),
            ('python.py', 'code'),
            ('javascript.js', 'code'),
            ('html.html', 'code'),
            ('markdown.md', 'document'),
        ]
        
        for filename, expected_type in test_cases:
            file_path = Path(sample_files) / filename
            detected_type = type_detector._detect_by_extension(file_path)
            assert detected_type == expected_type
    
    def test_file_categorization(self, type_detector, sample_files):
        """Test file categorization."""
        test_cases = [
            ('text.txt', 'documents'),
            ('python.py', 'code'),
            ('javascript.js', 'code'),
            ('html.html', 'code'),
        ]
        
        for filename, expected_category in test_cases:
            file_path = Path(sample_files) / filename
            category = type_detector.categorize_file(file_path)
            assert category == expected_category
    
    def test_mime_type_detection(self, type_detector, sample_files):
        """Test MIME type detection."""
        text_file = Path(sample_files) / 'text.txt'
        mime_type = type_detector.get_mime_type(text_file)
        
        # Should detect text MIME type
        assert mime_type.startswith('text/') or 'text' in mime_type
    
    def test_text_file_detection(self, type_detector, sample_files):
        """Test text file detection."""
        text_file = Path(sample_files) / 'text.txt'
        python_file = Path(sample_files) / 'python.py'
        
        assert type_detector.is_text_file(text_file) is True
        assert type_detector.is_text_file(python_file) is True


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    @pytest.fixture
    def batch_processor(self):
        """Create a BatchProcessor instance for testing."""
        return BatchProcessor(max_workers=2)
    
    @pytest.fixture
    def sample_files(self):
        """Create sample files for batch processing."""
        temp_dir = tempfile.mkdtemp()
        
        for i in range(5):
            file_path = Path(temp_dir) / f'test_{i}.txt'
            file_path.write_text(f'Content of test file {i}')
        
        yield [Path(temp_dir) / f'test_{i}.txt' for i in range(5)]
        shutil.rmtree(temp_dir)
    
    def test_parallel_processing(self, batch_processor, sample_files):
        """Test parallel batch processing."""
        def mock_processing_function(file_path):
            """Mock processing function that simulates work."""
            return {'file': str(file_path), 'processed': True, 'content_length': len(file_path.read_text())}
        
        results = batch_processor.process_batch(
            file_paths=sample_files,
            process_function=mock_processing_function,
            mode='parallel'
        )
        
        assert results['processed'] == len(sample_files)
        assert results['successful'] == len(sample_files)
        assert results['failed'] == 0
        assert len(results['details']) == len(sample_files)
    
    def test_serial_processing(self, batch_processor, sample_files):
        """Test serial batch processing."""
        def mock_processing_function(file_path):
            return {'file': str(file_path), 'status': 'success'}
        
        results = batch_processor.process_batch(
            file_paths=sample_files,
            process_function=mock_processing_function,
            mode='serial'
        )
        
        assert results['processed'] == len(sample_files)
        assert results['successful'] == len(sample_files)
    
    def test_chunking_files(self, batch_processor, sample_files):
        """Test file chunking for batch processing."""
        chunks = batch_processor.chunk_files(sample_files, chunk_size=2)
        
        assert len(chunks) == 3  # 5 files with chunk size 2 = 3 chunks
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1
    
    def test_time_estimation(self, batch_processor, sample_files):
        """Test processing time estimation."""
        estimates = batch_processor.estimate_processing_time(
            file_paths=sample_files,
            avg_time_per_file=0.1
        )
        
        assert estimates['total_files'] == len(sample_files)
        assert estimates['avg_time_per_file'] == 0.1
        assert estimates['estimated_serial_time'] == len(sample_files) * 0.1
        assert 'estimated_parallel_time' in estimates
        assert 'recommended_mode' in estimates


class TestOutputManager:
    """Test output management and directory structures."""
    
    @pytest.fixture
    def output_manager(self):
        """Create an OutputManager instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            yield manager
    
    def test_output_structure_setup(self, output_manager):
        """Test output directory structure setup."""
        output_dir = output_manager.setup_output_structure(
            structure_type='maintained'
        )
        
        assert output_dir.exists()
        assert 'maintained_structure' in str(output_dir)
        
        # Test different structure types
        flattened_dir = output_manager.setup_output_structure(
            structure_type='flattened'
        )
        assert 'flattened_structure' in str(flattened_dir)
        
        categorized_dir = output_manager.setup_output_structure(
            structure_type='categorized',
            categories=['documents', 'images', 'code']
        )
        assert 'categorized_structure' in str(categorized_dir)
    
    def test_structure_maintenance(self, output_manager):
        """Test maintaining directory structure."""
        with tempfile.TemporaryDirectory() as source_dir:
            # Create nested structure
            nested_file = Path(source_dir) / 'subdir1' / 'subdir2' / 'test.txt'
            nested_file.parent.mkdir(parents=True, exist_ok=True)
            nested_file.write_text('test content')
            
            source_root = Path(source_dir)
            output_dir = Path(output_manager.base_output_dir) / 'test_output'
            
            output_path = output_manager.maintain_structure(
                source_file=nested_file,
                source_root=source_root,
                output_dir=output_dir
            )
            
            # Should maintain the relative path
            assert 'subdir1/subdir2/test.txt' in str(output_path)
            assert output_path.parent.exists()
    
    def test_structure_flattening(self, output_manager):
        """Test flattening directory structure."""
        with tempfile.TemporaryDirectory() as source_dir:
            # Create file with same name in different directories
            file1 = Path(source_dir) / 'dir1' / 'test.txt'
            file1.parent.mkdir(parents=True, exist_ok=True)
            file1.write_text('content1')
            
            file2 = Path(source_dir) / 'dir2' / 'test.txt' 
            file2.parent.mkdir(parents=True, exist_ok=True)
            file2.write_text('content2')
            
            output_dir = Path(output_manager.base_output_dir) / 'test_output'
            output_dir.mkdir(exist_ok=True)
            
            # Test flattening first file
            output1 = output_manager.flatten_structure(file1, output_dir)
            assert output1.parent == output_dir
            
            # Test flattening second file (should get unique name)
            output2 = output_manager.flatten_structure(file2, output_dir)
            assert output2.parent == output_dir
            assert output1 != output2


class TestErrorHandler:
    """Test error handling and recovery."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler instance for testing."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_file = f.name
        
        handler = ErrorHandler(log_file=Path(log_file))
        yield handler
        
        # Cleanup
        Path(log_file).unlink()
    
    def test_error_handling(self, error_handler):
        """Test basic error handling."""
        test_error = ValueError("Test error message")
        context = "Testing error handler"
        
        result = error_handler.handle_error(
            error=test_error,
            context=context
        )
        
        # Should log the error and return False (error not resolved)
        assert result is False
    
    def test_retry_operation(self, error_handler):
        """Test operation retry logic."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "Success"
        
        result = error_handler.retry_operation(
            operation=failing_operation,
            max_retries=3,
            retry_delay=0.1  # Short delay for testing
        )
        
        assert result == "Success"
        assert call_count == 3
    
    def test_error_report_generation(self, error_handler):
        """Test error report generation."""
        # Log some test errors
        test_errors = [
            (ValueError("First error"), "Context 1"),
            (TypeError("Second error"), "Context 2"),
            (ValueError("Third error"), "Context 3"),
        ]
        
        for error, context in test_errors:
            error_handler.handle_error(error, context)
        
        report = error_handler.generate_error_report()
        
        assert 'summary' in report
        assert 'errors_by_type' in report
        assert report['summary']['total_errors'] == len(test_errors)
        assert report['summary']['error_types'] > 0


class TestIntegration:
    """Integration tests for file operations."""
    
    def test_full_file_processing_pipeline(self):
        """Test complete file processing pipeline."""
        # This would test the integration of all file operation components
        # from scanning to processing to output management
        pass
    
    def test_error_recovery_in_pipeline(self):
        """Test error recovery in processing pipeline."""
        # Test how the system handles and recovers from errors
        # during file processing operations
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])