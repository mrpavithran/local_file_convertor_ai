"""
Enhanced CSV to XLSX converter with formatting options and validation.
FIXED VERSION - Functional with proper error handling and system integration
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class CSVToXLSXConverter:
    """A tool for converting CSV files to XLSX format with advanced options."""
    
    def __init__(self):
        self.supported_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
    
    def convert(self, 
                input_path: str, 
                output_path: Optional[str] = None,
                sheet_name: str = "Data",
                preserve_headers: bool = True,
                auto_adjust_columns: bool = True,
                encoding: str = "utf-8",
                delimiter: str = ",") -> Dict[str, Any]:
        """
        Convert CSV file to XLSX format.
        
        Args:
            input_path: Path to the input CSV file
            output_path: Path for the output XLSX file (optional)
            sheet_name: Name of the Excel sheet
            preserve_headers: Whether to preserve CSV headers
            auto_adjust_columns: Whether to auto-adjust column widths in Excel
            encoding: File encoding for reading CSV
            delimiter: CSV delimiter character
            
        Returns:
            Dictionary with conversion results and metadata
        """
        # Validate input file
        if not os.path.exists(input_path):
            return {
                "success": False,
                "error": f"Input file not found: {input_path}"
            }
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_output_path(input_path)
        else:
            # Ensure output path has .xlsx extension
            if not output_path.lower().endswith('.xlsx'):
                output_path = os.path.splitext(output_path)[0] + '.xlsx'
        
        # Check if output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        try:
            # Read CSV file with error handling
            df = self._read_csv_with_fallback(
                input_path, 
                encoding=encoding,
                delimiter=delimiter,
                preserve_headers=preserve_headers
            )
            
            if df.empty:
                logger.warning(f"CSV file '{input_path}' is empty or contains no data")
            
            # Write to Excel
            self._write_to_excel(
                df, 
                output_path, 
                sheet_name=sheet_name,
                auto_adjust_columns=auto_adjust_columns
            )
            
            # Collect conversion statistics
            stats = self._get_conversion_stats(df, input_path, output_path)
            
            logger.info(f"Successfully converted '{input_path}' to '{output_path}'")
            return stats
            
        except Exception as e:
            # Clean up partially written file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return {
                "success": False,
                "error": f"Conversion failed: {str(e)}"
            }
    
    def _read_csv_with_fallback(self, 
                              input_path: str, 
                              encoding: str = "utf-8",
                              delimiter: str = ",",
                              preserve_headers: bool = True) -> Any:
        """
        Read CSV file with encoding fallback and error handling.
        """
        headers = 0 if preserve_headers else None
        
        # Try to import pandas
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV reading. Install with: pip install pandas")
        
        # Try specified encoding first
        try:
            df = pd.read_csv(
                input_path,
                encoding=encoding,
                delimiter=delimiter,
                header=headers,
                skip_blank_lines=False,
                on_bad_lines='skip'  # Skip problematic lines
            )
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            logger.warning(f"Encoding '{encoding}' failed for '{input_path}', trying fallback encodings")
        except Exception as e:
            logger.warning(f"CSV read failed with encoding {encoding}: {e}")
        
        # Try fallback encodings
        for fallback_encoding in self.supported_encodings:
            if fallback_encoding == encoding:
                continue
            try:
                df = pd.read_csv(
                    input_path,
                    encoding=fallback_encoding,
                    delimiter=delimiter,
                    header=headers,
                    skip_blank_lines=False,
                    on_bad_lines='skip'
                )
                logger.info(f"Successfully read '{input_path}' with encoding: {fallback_encoding}")
                return df
            except (UnicodeDecodeError, Exception) as e:
                logger.debug(f"Encoding {fallback_encoding} failed: {e}")
                continue
        
        # If all encodings fail, try without specifying encoding
        try:
            df = pd.read_csv(
                input_path,
                delimiter=delimiter,
                header=headers,
                skip_blank_lines=False,
                on_bad_lines='skip',
                encoding_errors='ignore'
            )
            logger.warning(f"Read '{input_path}' with encoding errors ignored")
            return df
        except Exception as e:
            raise Exception(f"Failed to read CSV file '{input_path}': {str(e)}") from e
    
    def _write_to_excel(self, 
                       df: Any, 
                       output_path: str, 
                       sheet_name: str = "Data",
                       auto_adjust_columns: bool = True):
        """
        Write DataFrame to Excel with formatting options.
        """
        try:
            # Try to import openpyxl
            try:
                import openpyxl
            except ImportError:
                raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
            
            # Write to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(
                    writer, 
                    sheet_name=sheet_name, 
                    index=False,
                    header=True if not df.empty else False
                )
                
                # Auto-adjust column widths if requested
                if auto_adjust_columns and not df.empty:
                    worksheet = writer.sheets[sheet_name]
                    self._auto_adjust_column_widths(worksheet, df)
                    
        except Exception as e:
            raise Exception(f"Excel writing failed: {str(e)}") from e
    
    def _auto_adjust_column_widths(self, worksheet, df: Any):
        """
        Auto-adjust column widths in Excel worksheet.
        """
        try:
            from openpyxl.utils import get_column_letter
            
            for idx, col in enumerate(df.columns, 1):
                # Handle both string and numeric column names
                col_name = str(col)
                max_length = 0
                
                # Check column name length
                max_length = max(max_length, len(col_name))
                
                # Check data length in this column
                if not df.empty:
                    col_data_length = df[col].astype(str).str.len().max()
                    max_length = max(max_length, col_data_length)
                
                # Set column width (add some padding)
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                column_letter = get_column_letter(idx)
                worksheet.column_dimensions[column_letter].width = adjusted_width
                
        except Exception as e:
            logger.warning(f"Could not auto-adjust column widths: {str(e)}")
    
    def _generate_output_path(self, input_path: str) -> str:
        """
        Generate output path from input path.
        """
        base_name = os.path.splitext(input_path)[0]
        counter = 1
        output_path = f"{base_name}.xlsx"
        
        # Avoid overwriting existing files
        while os.path.exists(output_path):
            output_path = f"{base_name}_{counter}.xlsx"
            counter += 1
        
        return output_path
    
    def _get_conversion_stats(self, df: Any, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Generate conversion statistics.
        """
        import pandas as pd
        
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        return {
            "success": True,
            "output_path": os.path.abspath(output_path),
            "input_file": os.path.basename(input_path),
            "output_file": os.path.basename(output_path),
            "rows": len(df),
            "columns": len(df.columns) if not df.empty else 0,
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "input_size_human": self._format_file_size(input_size),
            "output_size_human": self._format_file_size(output_size),
            "size_change_percent": round(((output_size - input_size) / input_size) * 100, 2) if input_size > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def batch_convert(self, 
                     input_directory: str, 
                     output_directory: Optional[str] = None,
                     recursive: bool = False,
                     **kwargs) -> Dict[str, Any]:
        """
        Convert multiple CSV files in a directory.
        
        Args:
            input_directory: Directory containing CSV files
            output_directory: Output directory (optional)
            recursive: Whether to search subdirectories
            **kwargs: Additional arguments for conversion
            
        Returns:
            Dictionary with batch conversion results
        """
        if not os.path.exists(input_directory):
            return {
                "success": False,
                "error": f"Input directory not found: {input_directory}"
            }
        
        output_directory = output_directory or input_directory
        
        # Find CSV files
        pattern = "**/*.csv" if recursive else "*.csv"
        csv_files = list(Path(input_directory).glob(pattern))
        
        if not csv_files:
            return {
                "success": False,
                "error": f"No CSV files found in: {input_directory}"
            }
        
        results = {
            "success": True,
            "total_files": len(csv_files),
            "successful_conversions": 0,
            "failed_conversions": 0,
            "conversions": []
        }
        
        for csv_file in csv_files:
            try:
                relative_path = csv_file.relative_to(input_directory)
                output_path = Path(output_directory) / relative_path.with_suffix('.xlsx')
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert file
                result = self.convert(
                    input_path=str(csv_file),
                    output_path=str(output_path),
                    **kwargs
                )
                
                if result.get('success'):
                    results["successful_conversions"] += 1
                else:
                    results["failed_conversions"] += 1
                
                results["conversions"].append({
                    "input_file": str(csv_file),
                    "output_file": str(output_path),
                    "success": result.get('success', False),
                    "result": result
                })
                
            except Exception as e:
                results["failed_conversions"] += 1
                results["conversions"].append({
                    "input_file": str(csv_file),
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"Failed to convert '{csv_file}': {e}")
        
        return results


# MCP Tool class
class Tool:
    """MCP-compatible tool wrapper for CSV to XLSX conversion."""
    
    name = "convert_csv_to_xlsx"
    description = "Convert CSV files to Excel XLSX format with various options"
    
    def __init__(self):
        self.converter = CSVToXLSXConverter()
    
    def run(self, 
            input_path: str, 
            output_path: str = None,
            sheet_name: str = "Data",
            preserve_headers: bool = True,
            auto_adjust_columns: bool = True) -> Dict[str, Any]:
        """
        Convert CSV file to XLSX format.
        
        Args:
            input_path: Path to the input CSV file
            output_path: Path for the output XLSX file (optional)
            sheet_name: Name of the Excel sheet
            preserve_headers: Whether to preserve CSV headers
            auto_adjust_columns: Whether to auto-adjust column widths
            
        Returns:
            Dictionary with conversion results
        """
        return self.converter.convert(
            input_path=input_path,
            output_path=output_path,
            sheet_name=sheet_name,
            preserve_headers=preserve_headers,
            auto_adjust_columns=auto_adjust_columns
        )


# Convenience function for simple conversions
def convert_csv_to_xlsx(input_path: str, output_path: str = None, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for converting CSV to XLSX.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path for the output XLSX file (optional)
        **kwargs: Additional conversion options
        
    Returns:
        Dictionary with conversion results
    """
    converter = CSVToXLSXConverter()
    return converter.convert(input_path, output_path, **kwargs)


# Test function
def test_conversion():
    """Test the CSV to XLSX conversion."""
    # Create a test CSV file
    test_csv = "test.csv"
    test_data = """Name,Age,City
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago"""
    
    try:
        with open(test_csv, 'w', encoding='utf-8') as f:
            f.write(test_data)
        
        result = convert_csv_to_xlsx(test_csv, "test_output.xlsx")
        print("Test conversion result:", result)
        
        # Clean up
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if os.path.exists("test_output.xlsx"):
            os.remove("test_output.xlsx")
            
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test
    test_conversion()