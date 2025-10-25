"""
System Orchestrator
Main coordinator for all system operations and workflows.
"""

import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

class SystemOrchestrator:
    """Orchestrates all system operations and manages workflow execution."""
    
    def __init__(self, config_manager, progress_tracker):
        self.config_manager = config_manager
        self.progress_tracker = progress_tracker
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        self.active_operations: Dict[str, Any] = {}
        self.operation_callbacks: Dict[str, List[Callable]] = {}
        self.system_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_operation_time': 0,
            'system_uptime': 0
        }
        self.start_time = datetime.now()
    
    def execute_operation(self, operation_type: str, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a system operation with full orchestration.
        
        Args:
            operation_type: Type of operation to execute
            operation_data: Operation parameters and data
            
        Returns:
            Operation results
        """
        operation_id = f"{operation_type}_{int(time.time())}_{len(self.active_operations)}"
        
        operation_result = {
            'operation_id': operation_id,
            'operation_type': operation_type,
            'status': 'started',
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
        
        try:
            # Track operation start
            self.active_operations[operation_id] = {
                'result': operation_result,
                'start_time': datetime.now()
            }
            
            self.progress_tracker.start_operation(operation_id, operation_type, operation_data)
            
            # Execute based on operation type
            if operation_type == 'file_conversion':
                result = self._execute_file_conversion(operation_id, operation_data)
            elif operation_type == 'batch_processing':
                result = self._execute_batch_processing(operation_id, operation_data)
            elif operation_type == 'content_enhancement':
                result = self._execute_content_enhancement(operation_id, operation_data)
            elif operation_type == 'workflow_execution':
                result = self._execute_workflow(operation_id, operation_data)
            else:
                result = {'success': False, 'error': f"Unknown operation type: {operation_type}"}
            
            # Update operation result
            operation_result.update(result)
            operation_result['status'] = 'completed' if result.get('success') else 'failed'
            operation_result['end_time'] = datetime.now().isoformat()
            
            # Update metrics
            self._update_system_metrics(operation_result)
            
            # Execute callbacks
            self._execute_callbacks(operation_id, operation_result)
            
            logger.info(f"Operation completed: {operation_id} - {operation_result['status']}")
            
        except Exception as e:
            operation_result.update({
                'status': 'failed',
                'error': f"Operation execution failed: {e}",
                'end_time': datetime.now().isoformat()
            })
            logger.error(f"Operation {operation_id} failed: {e}")
        
        finally:
            # Clean up
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
            self.progress_tracker.end_operation(operation_id)
        
        return operation_result
    
    def _execute_file_conversion(self, operation_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file conversion operation."""
        # This would integrate with file_operations module
        logger.info(f"Executing file conversion: {data.get('source_file')} -> {data.get('target_format')}")
        
        # Simulate conversion steps
        steps = [
            {'name': 'validate_input', 'status': 'completed'},
            {'name': 'read_source_file', 'status': 'completed'},
            {'name': 'convert_content', 'status': 'completed'},
            {'name': 'write_output_file', 'status': 'completed'}
        ]
        
        return {
            'success': True,
            'converted_file': data.get('source_file', '').replace('.txt', f'.{data.get("target_format", "md")}'),
            'steps': steps,
            'message': 'File conversion completed successfully'
        }
    
    def _execute_batch_processing(self, operation_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute batch processing operation."""
        files = data.get('files', [])
        operation = data.get('operation', 'convert')
        
        logger.info(f"Executing batch processing: {len(files)} files, operation: {operation}")
        
        results = []
        for i, file_path in enumerate(files):
            file_result = {
                'file': file_path,
                'status': 'completed',
                'result': f"Processed {file_path} with {operation}"
            }
            results.append(file_result)
            
            # Update progress
            progress = (i + 1) / len(files) * 100
            self.progress_tracker.update_operation_progress(operation_id, progress)
        
        return {
            'success': True,
            'processed_files': len(files),
            'results': results,
            'message': f'Batch processing completed: {len(files)} files'
        }
    
    def _execute_content_enhancement(self, operation_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content enhancement operation."""
        content = data.get('content', '')
        enhancement_type = data.get('enhancement_type', 'general')
        
        logger.info(f"Executing content enhancement: {enhancement_type}, content length: {len(content)}")
        
        # Simulate enhancement
        enhanced_content = f"ENHANCED: {content}"
        
        return {
            'success': True,
            'original_length': len(content),
            'enhanced_length': len(enhanced_content),
            'enhanced_content': enhanced_content,
            'message': 'Content enhancement completed'
        }
    
    def _execute_workflow(self, operation_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow operation."""
        workflow_name = data.get('workflow_name')
        parameters = data.get('parameters', {})
        
        logger.info(f"Executing workflow: {workflow_name}")
        
        # This would integrate with prompt_system workflow orchestrator
        return {
            'success': True,
            'workflow_name': workflow_name,
            'execution_time': '2.5s',
            'steps_completed': 3,
            'message': f'Workflow {workflow_name} executed successfully'
        }
    
    def _update_system_metrics(self, operation_result: Dict[str, Any]):
        """Update system performance metrics."""
        self.system_metrics['total_operations'] += 1
        
        if operation_result['status'] == 'completed':
            self.system_metrics['successful_operations'] += 1
        else:
            self.system_metrics['failed_operations'] += 1
        
        # Calculate average operation time
        if 'start_time' in operation_result and 'end_time' in operation_result:
            start = datetime.fromisoformat(operation_result['start_time'])
            end = datetime.fromisoformat(operation_result['end_time'])
            duration = (end - start).total_seconds()
            
            total_ops = self.system_metrics['total_operations']
            current_avg = self.system_metrics['average_operation_time']
            new_avg = (current_avg * (total_ops - 1) + duration) / total_ops
            self.system_metrics['average_operation_time'] = new_avg
        
        # Update uptime
        self.system_metrics['system_uptime'] = (datetime.now() - self.start_time).total_seconds()
    
    def _execute_callbacks(self, operation_id: str, result: Dict[str, Any]):
        """Execute registered callbacks for operation completion."""
        if operation_id in self.operation_callbacks:
            for callback in self.operation_callbacks[operation_id]:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback execution failed: {e}")
    
    def register_callback(self, operation_id: str, callback: Callable):
        """Register a callback for operation completion."""
        if operation_id not in self.operation_callbacks:
            self.operation_callbacks[operation_id] = []
        self.operation_callbacks[operation_id].append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        return {
            'status': 'running',
            'uptime_seconds': self.system_metrics['system_uptime'],
            'active_operations': len(self.active_operations),
            'metrics': self.system_metrics,
            'memory_usage': self._get_memory_usage(),
            'thread_pool_status': {
                'active_threads': self.thread_pool._max_workers,
                'pending_tasks': len([f for f in self.thread_pool._threads if not f.done()])
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def shutdown(self):
        """Shutdown the system orchestrator gracefully."""
        logger.info("Shutting down system orchestrator...")
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("System orchestrator shutdown complete")