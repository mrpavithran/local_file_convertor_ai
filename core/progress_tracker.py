"""
Progress Tracking System
Tracks operation progress and provides real-time status updates.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from threading import Lock
import json

logger = logging.getLogger(__name__)

class OperationProgress:
    """Tracks progress for a single operation."""
    
    def __init__(self, operation_id: str, operation_type: str, data: Dict[str, Any]):
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.progress = 0.0  # 0-100
        self.status = 'running'  # running, completed, failed, cancelled
        self.current_step = ''
        self.steps: List[Dict[str, Any]] = []
        self.metadata = data
        self.estimated_completion: Optional[datetime] = None
        self.last_update = datetime.now()
    
    def update_progress(self, progress: float, step: str = ''):
        """Update operation progress."""
        self.progress = max(0, min(100, progress))
        if step:
            self.current_step = step
        self.last_update = datetime.now()
        
        # Estimate completion time
        if progress > 0:
            elapsed = (self.last_update - self.start_time).total_seconds()
            total_estimated = elapsed / (progress / 100)
            self.estimated_completion = self.start_time + timedelta(seconds=total_estimated)
    
    def add_step(self, step_name: str, status: str = 'started', details: str = ''):
        """Add a step to the operation."""
        step = {
            'name': step_name,
            'status': status,
            'start_time': datetime.now().isoformat(),
            'details': details
        }
        self.steps.append(step)
        self.current_step = step_name
    
    def complete_step(self, step_name: str, details: str = ''):
        """Mark a step as completed."""
        for step in self.steps:
            if step['name'] == step_name:
                step['status'] = 'completed'
                step['end_time'] = datetime.now().isoformat()
                step['details'] = details
                break
    
    def complete_operation(self, status: str = 'completed'):
        """Mark operation as completed."""
        self.status = status
        self.progress = 100.0
        self.end_time = datetime.now()
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'progress': self.progress,
            'status': self.status,
            'current_step': self.current_step,
            'steps': self.steps,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'last_update': self.last_update.isoformat(),
            'elapsed_seconds': (self.last_update - self.start_time).total_seconds() if self.last_update else 0
        }

class ProgressTracker:
    """Tracks progress for all system operations."""
    
    def __init__(self, history_size: int = 1000):
        self.active_operations: Dict[str, OperationProgress] = {}
        self.completed_operations: List[Dict[str, Any]] = []
        self.history_size = history_size
        self.lock = Lock()
        self.metrics = {
            'total_operations': 0,
            'operations_completed': 0,
            'operations_failed': 0,
            'total_processing_time': 0,
            'average_operation_time': 0
        }
    
    def start_operation(self, operation_id: str, operation_type: str, data: Dict[str, Any]) -> bool:
        """Start tracking a new operation."""
        with self.lock:
            if operation_id in self.active_operations:
                logger.warning(f"Operation {operation_id} already being tracked")
                return False
            
            progress = OperationProgress(operation_id, operation_type, data)
            self.active_operations[operation_id] = progress
            self.metrics['total_operations'] += 1
            
            logger.info(f"Started tracking operation: {operation_id} ({operation_type})")
            return True
    
    def update_operation_progress(self, operation_id: str, progress: float, step: str = '') -> bool:
        """Update progress for an operation."""
        with self.lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found for progress update")
                return False
            
            self.active_operations[operation_id].update_progress(progress, step)
            return True
    
    def add_operation_step(self, operation_id: str, step_name: str, details: str = '') -> bool:
        """Add a step to an operation."""
        with self.lock:
            if operation_id not in self.active_operations:
                return False
            
            self.active_operations[operation_id].add_step(step_name, 'started', details)
            return True
    
    def complete_operation_step(self, operation_id: str, step_name: str, details: str = '') -> bool:
        """Complete a step in an operation."""
        with self.lock:
            if operation_id not in self.active_operations:
                return False
            
            self.active_operations[operation_id].complete_step(step_name, details)
            return True
    
    def end_operation(self, operation_id: str, status: str = 'completed') -> bool:
        """End an operation and move to completed history."""
        with self.lock:
            if operation_id not in self.active_operations:
                return False
            
            operation = self.active_operations[operation_id]
            operation.complete_operation(status)
            
            # Move to completed operations
            operation_dict = operation.to_dict()
            self.completed_operations.append(operation_dict)
            
            # Update metrics
            if status == 'completed':
                self.metrics['operations_completed'] += 1
            else:
                self.metrics['operations_failed'] += 1
            
            # Calculate average operation time
            if operation.end_time:
                duration = (operation.end_time - operation.start_time).total_seconds()
                self.metrics['total_processing_time'] += duration
                total_ops = self.metrics['operations_completed'] + self.metrics['operations_failed']
                if total_ops > 0:
                    self.metrics['average_operation_time'] = self.metrics['total_processing_time'] / total_ops
            
            # Remove from active operations
            del self.active_operations[operation_id]
            
            # Maintain history size
            if len(self.completed_operations) > self.history_size:
                self.completed_operations = self.completed_operations[-self.history_size:]
            
            logger.info(f"Completed operation: {operation_id} - {status}")
            return True
    
    def get_operation_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get progress for a specific operation."""
        with self.lock:
            if operation_id in self.active_operations:
                return self.active_operations[operation_id].to_dict()
            else:
                # Check completed operations
                for op in self.completed_operations:
                    if op['operation_id'] == operation_id:
                        return op
                return None
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get all active operations."""
        with self.lock:
            return [op.to_dict() for op in self.active_operations.values()]
    
    def get_recent_operations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent operations (both active and completed)."""
        with self.lock:
            active_ops = [op.to_dict() for op in self.active_operations.values()]
            recent_completed = self.completed_operations[-limit:] if self.completed_operations else []
            return active_ops + recent_completed
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about operations."""
        with self.lock:
            now = datetime.now()
            active_ops = list(self.active_operations.values())
            
            # Calculate current throughput
            recent_ops = [op for op in self.completed_operations 
                         if datetime.fromisoformat(op['end_time']) > now - timedelta(hours=1)]
            hourly_throughput = len(recent_ops)
            
            return {
                'metrics': self.metrics.copy(),
                'current_active': len(active_ops),
                'hourly_throughput': hourly_throughput,
                'oldest_active': min([op.start_time for op in active_ops]).isoformat() if active_ops else None,
                'success_rate': (self.metrics['operations_completed'] / self.metrics['total_operations'] * 100 
                               if self.metrics['total_operations'] > 0 else 0)
            }
    
    def save_progress_snapshot(self, filepath: str) -> bool:
        """Save current progress state to file."""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'active_operations': [op.to_dict() for op in self.active_operations.values()],
                'completed_operations': self.completed_operations,
                'metrics': self.metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            logger.info(f"Progress snapshot saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save progress snapshot: {e}")
            return False