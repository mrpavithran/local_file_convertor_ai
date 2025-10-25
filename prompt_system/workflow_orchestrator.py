"""
Workflow Orchestration
Orchestrates complex prompt workflows and multi-step operations.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class WorkflowStep:
    """Represents a single step in a workflow."""
    
    def __init__(self, name: str, template: str, parameters: Dict[str, Any], 
                 depends_on: List[str] = None):
        self.name = name
        self.template = template
        self.parameters = parameters
        self.depends_on = depends_on or []
        self.status = 'pending'  # pending, running, completed, failed
        self.result: Optional[Dict[str, Any]] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

class WorkflowOrchestrator:
    """Orchestrates multi-step prompt workflows."""
    
    def __init__(self, prompt_templates, parameter_parser):
        self.prompt_templates = prompt_templates
        self.parameter_parser = parameter_parser
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def create_workflow(self, name: str, steps: List[Dict[str, Any]]) -> bool:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            steps: List of step definitions
            
        Returns:
            Success status
        """
        try:
            workflow_steps = []
            for step_def in steps:
                step = WorkflowStep(
                    name=step_def['name'],
                    template=step_def['template'],
                    parameters=step_def.get('parameters', {}),
                    depends_on=step_def.get('depends_on', [])
                )
                workflow_steps.append(step)
            
            self.workflows[name] = workflow_steps
            logger.info(f"Created workflow '{name}' with {len(steps)} steps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workflow '{name}': {e}")
            return False
    
    def execute_workflow(self, workflow_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_name: Name of the workflow to execute
            context: Initial context for parameter resolution
            
        Returns:
            Execution results
        """
        if workflow_name not in self.workflows:
            return {
                'success': False,
                'error': f"Workflow '{workflow_name}' not found"
            }
        
        workflow = self.workflows[workflow_name]
        context = context or {}
        execution_id = f"{workflow_name}_{int(time.time())}"
        
        logger.info(f"Starting workflow execution: {execution_id}")
        
        execution_result = {
            'execution_id': execution_id,
            'workflow_name': workflow_name,
            'started_at': datetime.now().isoformat(),
            'steps': [],
            'overall_status': 'running'
        }
        
        try:
            # Execute steps in dependency order
            completed_steps = {}
            remaining_steps = workflow.copy()
            
            while remaining_steps:
                executable_steps = [
                    step for step in remaining_steps 
                    if all(dep in completed_steps for dep in step.depends_on)
                ]
                
                if not executable_steps:
                    # Circular dependency or missing dependency
                    execution_result['overall_status'] = 'failed'
                    execution_result['error'] = 'Cannot resolve step dependencies'
                    break
                
                for step in executable_steps:
                    step_result = self._execute_step(step, context, completed_steps)
                    execution_result['steps'].append(step_result)
                    
                    if step_result['status'] == 'completed':
                        completed_steps[step.name] = step_result
                        remaining_steps.remove(step)
                    else:
                        # Step failed, stop workflow
                        execution_result['overall_status'] = 'failed'
                        execution_result['error'] = f"Step '{step.name}' failed"
                        break
                
                if execution_result['overall_status'] == 'failed':
                    break
            
            if execution_result['overall_status'] == 'running':
                execution_result['overall_status'] = 'completed'
                execution_result['success'] = True
            
            execution_result['completed_at'] = datetime.now().isoformat()
            execution_result['total_steps'] = len(workflow)
            execution_result['completed_steps'] = len(completed_steps)
            
            # Store in history
            self.execution_history.append(execution_result)
            
            logger.info(f"Workflow completed: {execution_id} - {execution_result['overall_status']}")
            
        except Exception as e:
            execution_result['overall_status'] = 'failed'
            execution_result['error'] = f"Workflow execution failed: {e}"
            execution_result['completed_at'] = datetime.now().isoformat()
            logger.error(f"Workflow execution failed: {e}")
        
        return execution_result
    
    def _execute_step(self, step: WorkflowStep, context: Dict[str, Any], 
                     completed_steps: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step.started_at = datetime.now()
        step_result = {
            'step_name': step.name,
            'template': step.template,
            'status': 'running',
            'started_at': step.started_at.isoformat()
        }
        
        try:
            # Resolve parameters using context and previous steps
            resolved_parameters = self._resolve_step_parameters(step, context, completed_steps)
            
            # Format the prompt
            prompt_result = self.prompt_templates.format_prompt(
                step.template, 
                **resolved_parameters
            )
            
            if not prompt_result['success']:
                step_result.update({
                    'status': 'failed',
                    'error': prompt_result['error'],
                    'completed_at': datetime.now().isoformat()
                })
                return step_result
            
            # Store prompt result (in real implementation, this would execute the prompt)
            step_result.update({
                'status': 'completed',
                'prompt_result': prompt_result,
                'parameters_used': resolved_parameters,
                'completed_at': datetime.now().isoformat()
            })
            
            logger.info(f"Step completed: {step.name}")
            
        except Exception as e:
            step_result.update({
                'status': 'failed',
                'error': f"Step execution failed: {e}",
                'completed_at': datetime.now().isoformat()
            })
            logger.error(f"Step '{step.name}' failed: {e}")
        
        return step_result
    
    def _resolve_step_parameters(self, step: WorkflowStep, context: Dict[str, Any],
                                completed_steps: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve step parameters from context and previous steps."""
        resolved = {}
        
        # Add context parameters
        resolved.update(context)
        
        # Add parameters from dependent steps
        for dep_name in step.depends_on:
            if dep_name in completed_steps:
                dep_result = completed_steps[dep_name]
                if 'prompt_result' in dep_result:
                    # Use output from previous step
                    resolved[f"{dep_name}_output"] = dep_result['prompt_result'].get('prompt', '')
        
        # Add step-specific parameters
        resolved.update(step.parameters)
        
        # Parse and validate all parameters
        return self.parameter_parser.parse_parameters(step.template, resolved)
    
    def get_workflow_status(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow."""
        if workflow_name not in self.workflows:
            return None
        
        # Find latest execution for this workflow
        recent_executions = [
            exec_data for exec_data in self.execution_history
            if exec_data['workflow_name'] == workflow_name
        ]
        
        if not recent_executions:
            return {'status': 'not_executed'}
        
        latest = recent_executions[-1]
        return {
            'status': latest['overall_status'],
            'last_execution': latest['started_at'],
            'total_steps': latest['total_steps'],
            'completed_steps': latest.get('completed_steps', 0)
        }
    
    def list_workflows(self) -> Dict[str, Any]:
        """List all available workflows."""
        workflow_info = {}
        for name, steps in self.workflows.items():
            workflow_info[name] = {
                'step_count': len(steps),
                'steps': [step.name for step in steps],
                'status': self.get_workflow_status(name)
            }
        
        return {
            'workflows': workflow_info,
            'total_workflows': len(self.workflows)
        }