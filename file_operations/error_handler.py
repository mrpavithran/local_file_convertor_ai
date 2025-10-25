"""
Error handling with retry logic and comprehensive logging.
"""

import time
import logging
from typing import Callable, Optional, Any, Dict
from pathlib import Path
from rich.console import Console
from rich.table import Table


class ErrorHandler:
    """Handle errors with retry logic and comprehensive reporting."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.console = Console()
        self.errors_log: Dict[str, List[str]] = {}
        
        # Setup logging
        self.logger = logging.getLogger('file_operations')
        self.logger.setLevel(logging.ERROR)
        
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "",
                    retry_count: int = 3,
                    retry_delay: float = 1.0) -> bool:
        """
        Handle error with optional retry logic.
        
        Args:
            error: The exception that occurred
            context: Context information about where error occurred
            retry_count: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if error was handled successfully after retries
        """
        error_key = f"{context}_{type(error).__name__}"
        
        if error_key not in self.errors_log:
            self.errors_log[error_key] = []
        
        error_message = f"{context}: {str(error)}"
        self.errors_log[error_key].append(error_message)
        
        # Log the error
        self.logger.error(error_message, exc_info=True)
        self.console.print(f"[red]Error: {error_message}[/red]")
        
        return False
    
    def retry_operation(self,
                       operation: Callable,
                       operation_args: tuple = (),
                       operation_kwargs: Optional[Dict] = None,
                       max_retries: int = 3,
                       retry_delay: float = 1.0,
                       backoff_factor: float = 2.0) -> Any:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            operation_args: Positional arguments for operation
            operation_kwargs: Keyword arguments for operation
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries
            backoff_factor: Multiplier for delay after each retry
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        operation_kwargs = operation_kwargs or {}
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation(*operation_args, **operation_kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    current_delay = retry_delay * (backoff_factor ** attempt)
                    self.console.print(
                        f"[yellow]Attempt {attempt + 1} failed. "
                        f"Retrying in {current_delay:.1f}s...[/yellow]"
                    )
                    time.sleep(current_delay)
                else:
                    self.console.print(
                        f"[red]All {max_retries} attempts failed. "
                        f"Last error: {str(e)}[/red]"
                    )
                    raise last_exception
        
        raise last_exception  # This should never be reached
    
    def should_skip_file(self, 
                        file_path: Path,
                        error: Exception,
                        skip_patterns: Optional[list] = None) -> bool:
        """
        Determine if a file should be skipped based on error and patterns.
        
        Args:
            file_path: Path to the file
            error: The exception that occurred
            skip_patterns: List of error patterns to skip
            
        Returns:
            True if file should be skipped
        """
        skip_patterns = skip_patterns or [
            "Permission denied",
            "File not found",
            "Unsupported format",
            "Corrupted file"
        ]
        
        error_str = str(error).lower()
        
        for pattern in skip_patterns:
            if pattern.lower() in error_str:
                self.console.print(
                    f"[yellow]Skipping file {file_path} due to: {pattern}[/yellow]"
                )
                return True
        
        return False
    
    def generate_error_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive error report.
        
        Returns:
            Dictionary with error statistics and details
        """
        total_errors = sum(len(errors) for errors in self.errors_log.values())
        
        report = {
            'summary': {
                'total_errors': total_errors,
                'error_types': len(self.errors_log),
                'most_common_error': max(self.errors_log.items(), 
                                       key=lambda x: len(x[1]))[0] if self.errors_log else None
            },
            'errors_by_type': {
                error_type: {
                    'count': len(errors),
                    'examples': errors[:5]  # Show first 5 examples
                }
                for error_type, errors in self.errors_log.items()
            }
        }
        
        return report
    
    def display_error_summary(self):
        """Display error summary in a formatted table."""
        report = self.generate_error_report()
        
        if not report['errors_by_type']:
            self.console.print("[green]No errors encountered! 🎉[/green]")
            return
        
        table = Table(title="Error Summary", show_header=True, header_style="bold red")
        table.add_column("Error Type", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Examples", style="dim")
        
        for error_type, info in report['errors_by_type'].items():
            examples = "\n".join(info['examples'][:2])  # Show first 2 examples
            table.add_row(error_type, str(info['count']), examples)
        
        self.console.print(table)
        
        # Display summary
        summary = report['summary']
        self.console.print(
            f"\n[bold]Total Errors:[/bold] {summary['total_errors']} | "
            f"[bold]Error Types:[/bold] {summary['error_types']}"
        )
        
        if summary['most_common_error']:
            self.console.print(f"[bold]Most Common Error:[/bold] {summary['most_common_error']}")