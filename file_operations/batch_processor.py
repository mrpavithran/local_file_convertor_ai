"""
Batch processing with parallel execution and progress tracking.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Callable, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .error_handler import ErrorHandler


class BatchProcessor:
    """Handle batch processing of files with parallel execution."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.console = Console()
        self.error_handler = ErrorHandler()
        self.max_workers = max_workers or multiprocessing.cpu_count()
    
    def process_batch(self, 
                     file_paths: List[Path],
                     process_function: Callable,
                     mode: str = 'parallel',
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a batch of files.
        
        Args:
            file_paths: List of file paths to process
            process_function: Function to process each file
            mode: 'parallel', 'serial', or 'threaded'
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'details': []
        }
        
        if not file_paths:
            return results
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Processing batch...", total=len(file_paths))
            
            if mode == 'parallel':
                results = self._process_parallel(file_paths, process_function, progress, task)
            elif mode == 'threaded':
                results = self._process_threaded(file_paths, process_function, progress, task)
            else:  # serial
                results = self._process_serial(file_paths, process_function, progress, task)
        
        return results
    
    def _process_parallel(self, file_paths: List[Path], 
                         process_function: Callable,
                         progress: Progress,
                         task) -> Dict[str, Any]:
        """Process files using parallel processes."""
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'details': []
        }
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_function, file_path): file_path 
                for file_path in file_paths
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                results['processed'] += 1
                
                try:
                    result = future.result()
                    results['successful'] += 1
                    results['details'].append({
                        'file': str(file_path),
                        'status': 'success',
                        'result': result
                    })
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"Failed to process {file_path}: {e}"
                    results['errors'].append(error_msg)
                    results['details'].append({
                        'file': str(file_path),
                        'status': 'failed',
                        'error': str(e)
                    })
                    self.error_handler.handle_error(e, f"Processing {file_path}")
                
                progress.update(task, advance=1)
        
        return results
    
    def _process_threaded(self, file_paths: List[Path],
                         process_function: Callable,
                         progress: Progress,
                         task) -> Dict[str, Any]:
        """Process files using threads."""
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'details': []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(process_function, file_path): file_path 
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                results['processed'] += 1
                
                try:
                    result = future.result()
                    results['successful'] += 1
                    results['details'].append({
                        'file': str(file_path),
                        'status': 'success',
                        'result': result
                    })
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"Failed to process {file_path}: {e}"
                    results['errors'].append(error_msg)
                    results['details'].append({
                        'file': str(file_path),
                        'status': 'failed',
                        'error': str(e)
                    })
                    self.error_handler.handle_error(e, f"Processing {file_path}")
                
                progress.update(task, advance=1)
        
        return results
    
    def _process_serial(self, file_paths: List[Path],
                       process_function: Callable,
                       progress: Progress,
                       task) -> Dict[str, Any]:
        """Process files serially."""
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'details': []
        }
        
        for file_path in file_paths:
            results['processed'] += 1
            
            try:
                result = process_function(file_path)
                results['successful'] += 1
                results['details'].append({
                    'file': str(file_path),
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Failed to process {file_path}: {e}"
                results['errors'].append(error_msg)
                results['details'].append({
                    'file': str(file_path),
                    'status': 'failed',
                    'error': str(e)
                })
                self.error_handler.handle_error(e, f"Processing {file_path}")
            
            progress.update(task, advance=1)
        
        return results
    
    def chunk_files(self, file_paths: List[Path], chunk_size: int) -> List[List[Path]]:
        """
        Split files into chunks for batch processing.
        
        Args:
            file_paths: List of file paths
            chunk_size: Size of each chunk
            
        Returns:
            List of file chunks
        """
        return [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]
    
    def estimate_processing_time(self, file_paths: List[Path], 
                                avg_time_per_file: float) -> Dict[str, Any]:
        """
        Estimate processing time for batch.
        
        Args:
            file_paths: List of file paths
            avg_time_per_file: Average processing time per file in seconds
            
        Returns:
            Dictionary with time estimates
        """
        total_files = len(file_paths)
        total_time_serial = total_files * avg_time_per_file
        
        estimates = {
            'total_files': total_files,
            'avg_time_per_file': avg_time_per_file,
            'estimated_serial_time': total_time_serial,
            'estimated_parallel_time': total_time_serial / min(self.max_workers, total_files),
            'estimated_threaded_time': total_time_serial / min(self.max_workers, total_files),
            'recommended_mode': 'parallel' if total_files > 10 else 'serial'
        }
        
        return estimates