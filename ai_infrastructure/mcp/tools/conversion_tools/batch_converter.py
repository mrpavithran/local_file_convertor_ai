"""
Enhanced batch conversion tool with support for multiple file formats and comprehensive error handling.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class BatchConvertTool:
    """Batch file conversion tool supporting multiple formats with progress tracking and error handling."""
    
    name = 'batch_convert'
    description = 'Batch convert files between supported formats (CSV→XLSX, DOCX→PDF, PDF→DOCX)'
    
    # Supported operations and their configurations
    SUPPORTED_OPERATIONS = {
        'csv2xlsx': {
            'input_ext': ['.csv'],
            'output_ext': '.xlsx',
            'description': 'Convert CSV files to Excel XLSX format',
            'tool_name': 'convert_csv_to_xlsx'
        },
        'docx2pdf': {
            'input_ext': ['.docx', '.doc'],
            'output_ext': '.pdf', 
            'description': 'Convert DOCX files to PDF format',
            'tool_name': 'convert_docx_to_pdf'
        },
        'pdf2docx': {
            'input_ext': ['.pdf'],
            'output_ext': '.docx',
            'description': 'Convert PDF files to DOCX format',
            'tool_name': 'convert_pdf_to_docx'
        },
        'txt2pdf': {
            'input_ext': ['.txt'],
            'output_ext': '.pdf',
            'description': 'Convert text files to PDF format',
            'tool_name': 'convert_txt_to_pdf'
        }
    }
    
    def __init__(self):
        self.conversion_tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize conversion tools with error handling."""
        try:
            from ..conversion_tools import (
                CSVToXLSXConverterTool,
                DOCXToPDFConverterTool, 
                PDFToDOCXConverterTool
            )
            self.conversion_tools = {
                'convert_csv_to_xlsx': CSVToXLSXConverterTool(),
                'convert_docx_to_pdf': DOCXToPDFConverterTool(),
                'convert_pdf_to_docx': PDFToDOCXConverterTool(),
            }
        except ImportError as e:
            logger.warning(f"Some conversion tools not available: {e}")
    
    def run(self, 
            files: List[str], 
            operation: str,
            output_dir: Optional[str] = None,
            overwrite: bool = False,
            preserve_structure: bool = True) -> Dict[str, Any]:
        """
        Batch convert files with comprehensive options.
        
        Args:
            files: List of file paths to convert
            operation: Conversion operation (csv2xlsx, docx2pdf, pdf2docx)
            output_dir: Custom output directory (optional)
            overwrite: Whether to overwrite existing files
            preserve_structure: Whether to preserve directory structure
            
        Returns:
            Batch conversion results with detailed statistics
        """
        # Validate operation
        if operation not in self.SUPPORTED_OPERATIONS:
            return self._error_result(f"Unsupported operation: {operation}. Supported: {list(self.SUPPORTED_OPERATIONS.keys())}")
        
        operation_config = self.SUPPORTED_OPERATIONS[operation]
        
        # Validate files
        if not files:
            return self._error_result("No files provided for conversion")
        
        valid_files = self._validate_files(files, operation_config['input_ext'])
        if not valid_files['valid_files']:
            return self._error_result(f"No valid {operation_config['input_ext']} files found")
        
        # Setup output directory
        output_dir = self._setup_output_directory(output_dir, operation)
        
        # Process files
        results = self._process_batch_conversion(
            valid_files['valid_files'], 
            operation, 
            operation_config,
            output_dir,
            overwrite,
            preserve_structure
        )
        
        return self._compile_results(results, valid_files, operation_config)
    
    def _validate_files(self, files: List[str], supported_extensions: List[str]) -> Dict[str, Any]:
        """Validate input files and filter by supported extensions."""
        valid_files = []
        invalid_files = []
        missing_files = []
        
        for file_path in files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                continue
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext in supported_extensions:
                valid_files.append(file_path)
            else:
                invalid_files.append({
                    'file': file_path,
                    'reason': f'Unsupported extension: {file_ext}. Expected: {supported_extensions}'
                })
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'missing_files': missing_files,
            'total_checked': len(files)
        }
    
    def _setup_output_directory(self, output_dir: Optional[str], operation: str) -> str:
        """Setup and create output directory if needed."""
        if output_dir is None:
            # Create operation-specific default directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./converted_{operation}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _process_batch_conversion(self, 
                                files: List[str], 
                                operation: str,
                                operation_config: Dict,
                                output_dir: str,
                                overwrite: bool,
                                preserve_structure: bool) -> List[Dict[str, Any]]:
        """Process batch conversion for all valid files."""
        results = []
        
        for file_path in files:
            try:
                result = self._convert_single_file(
                    file_path, 
                    operation, 
                    operation_config,
                    output_dir,
                    overwrite,
                    preserve_structure
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to convert {file_path}: {e}")
                results.append({
                    'input_file': file_path,
                    'status': 'failed',
                    'error': str(e),
                    'output_file': None
                })
        
        return results
    
    def _convert_single_file(self, 
                           file_path: str, 
                           operation: str,
                           operation_config: Dict,
                           output_dir: str,
                           overwrite: bool,
                           preserve_structure: bool) -> Dict[str, Any]:
        """Convert a single file with proper path handling."""
        # Generate output path
        output_path = self._generate_output_path(
            file_path, 
            operation_config['output_ext'],
            output_dir,
            preserve_structure
        )
        
        # Check if output exists
        if os.path.exists(output_path) and not overwrite:
            return {
                'input_file': file_path,
                'status': 'skipped',
                'reason': 'Output file exists and overwrite=False',
                'output_file': output_path
            }
        
        # Perform conversion
        tool_name = operation_config['tool_name']
        if tool_name not in self.conversion_tools:
            return {
                'input_file': file_path,
                'status': 'failed',
                'error': f'Conversion tool not available: {tool_name}',
                'output_file': None
            }
        
        try:
            # Execute conversion
            conversion_result = self.conversion_tools[tool_name].run(
                input_path=file_path,
                output_path=output_path
            )
            
            return {
                'input_file': file_path,
                'status': 'success',
                'output_file': output_path,
                'conversion_stats': conversion_result,
                'file_size_input': os.path.getsize(file_path),
                'file_size_output': os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }
            
        except Exception as e:
            # Clean up failed output file
            if os.path.exists(output_path):
                os.remove(output_path)
            raise e
    
    def _generate_output_path(self, 
                            input_path: str, 
                            output_ext: str,
                            output_dir: str,
                            preserve_structure: bool) -> str:
        """Generate output path preserving or flattening structure."""
        input_path_obj = Path(input_path)
        
        if preserve_structure:
            # Preserve relative directory structure
            relative_path = input_path_obj
            if relative_path.is_absolute():
                # Create a flattened version for absolute paths
                safe_name = input_path_obj.name.replace(' ', '_').replace('/', '_')
                output_filename = input_path_obj.stem + output_ext
                return str(Path(output_dir) / safe_name)
            else:
                output_filename = input_path_obj.stem + output_ext
                return str(Path(output_dir) / output_filename)
        else:
            # Flatten structure - all files in output directory
            output_filename = input_path_obj.stem + output_ext
            return str(Path(output_dir) / output_filename)
    
    def _compile_results(self, 
                       results: List[Dict], 
                       file_validation: Dict,
                       operation_config: Dict) -> Dict[str, Any]:
        """Compile comprehensive batch conversion results."""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        skipped = [r for r in results if r['status'] == 'skipped']
        
        # Calculate statistics
        total_size_input = sum(r.get('file_size_input', 0) for r in successful)
        total_size_output = sum(r.get('file_size_output', 0) for r in successful)
        
        return {
            'operation': operation_config['description'],
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files_processed': len(results),
                'successful_conversions': len(successful),
                'failed_conversions': len(failed),
                'skipped_files': len(skipped),
                'success_rate': len(successful) / len(results) if results else 0,
                'total_size_reduction': total_size_output - total_size_input,
                'total_size_reduction_percent': ((total_size_output - total_size_input) / total_size_input * 100) if total_size_input > 0 else 0
            },
            'file_validation': {
                'total_files_checked': file_validation['total_checked'],
                'valid_files': len(file_validation['valid_files']),
                'invalid_files': file_validation['invalid_files'],
                'missing_files': file_validation['missing_files']
            },
            'conversion_results': results,
            'settings': {
                'output_directory': results[0]['output_file'] if results else None,
                'preserve_structure': True  # This would come from input in real implementation
            }
        }
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'operation': 'batch_convert',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': error_message,
            'summary': {
                'total_files_processed': 0,
                'successful_conversions': 0,
                'failed_conversions': 0,
                'skipped_files': 0,
                'success_rate': 0
            },
            'conversion_results': []
        }
    
    def get_supported_operations(self) -> Dict[str, Any]:
        """Get information about supported batch operations."""
        return {
            'supported_operations': self.SUPPORTED_OPERATIONS,
            'total_operations': len(self.SUPPORTED_OPERATIONS),
            'available_tools': list(self.conversion_tools.keys())
        }


# Legacy tool class for backward compatibility
class Tool:
    """Legacy batch conversion tool (maintains original interface)."""
    
    name = 'batch_convert'
    description = 'Batch convert files between formats'
    
    def __init__(self):
        self.converter = BatchConvertTool()
    
    def run(self, 
            files: List[str], 
            operation: str,
            output_dir: Optional[str] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Batch convert files with enhanced capabilities.
        
        Args:
            files: List of file paths to convert
            operation: Conversion operation (csv2xlsx, docx2pdf, pdf2docx)
            output_dir: Custom output directory
            **kwargs: Additional options (overwrite, preserve_structure)
            
        Returns:
            Batch conversion results
        """
        return self.converter.run(
            files=files,
            operation=operation,
            output_dir=output_dir,
            **kwargs
        )


# Example usage
if __name__ == "__main__":
    tool = BatchConvertTool()
    
    # Example batch conversion
    files = [
        "document1.docx",
        "document2.docx", 
        "data.csv"
    ]
    
    result = tool.run(files, "docx2pdf", output_dir="./converted_pdfs")
    print(f"Batch conversion complete: {result['summary']['successful_conversions']}/{result['summary']['total_files_processed']} successful")