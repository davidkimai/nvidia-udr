"""
Strategy Executor - Sandboxed execution of generated research strategy code.
Handles Phase 2 of UDR operation with security constraints and state management.
"""

import ast
import sys
import traceback
import signal
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional
from datetime import datetime


class ExecutionTimeout(Exception):
    """Raised when strategy execution exceeds timeout limit."""
    pass


class StrategyExecutor:
    """
    Executes generated research strategy code in controlled environment 
    with timeout protection and security constraints.
    """
    
    def __init__(self, timeout: int = 3600):
        """
        Initialize strategy executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.restricted_modules = {
            'subprocess', 'os', 'sys', 'shutil', 'socket', 'urllib',
            'http', 'ftplib', 'smtplib', 'telnetlib', 'exec', 'eval'
        }
        
    def execute(self, 
                code: str, 
                context: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Execute strategy code with provided context.
        
        Args:
            code: Python code to execute
            context: Execution context with tools and parameters
            
        Yields:
            Dict: Progress notifications from strategy execution
            
        Raises:
            ExecutionTimeout: If execution exceeds timeout
            ValueError: If code execution fails
        """
        # Validate code before execution
        validation = self.validate_code(code)
        if not validation['valid']:
            raise ValueError(f"Code validation failed: {validation['details']}")
        
        # Prepare secure execution environment
        exec_globals = self._build_execution_globals(context)
        exec_locals = {}
        
        try:
            with self._timeout_context(self.timeout):
                # Execute code to define the strategy function
                exec(code, exec_globals, exec_locals)
                
                # Get the strategy function
                if 'execute_strategy' not in exec_locals:
                    raise ValueError("Code must define 'execute_strategy' function")
                
                strategy_func = exec_locals['execute_strategy']
                
                # Execute strategy and yield notifications
                for notification in strategy_func(**context):
                    # Validate notification format
                    if not self._validate_notification(notification):
                        continue
                    yield notification
                    
        except ExecutionTimeout:
            yield {
                "type": "execution_timeout",
                "description": f"Strategy execution exceeded {self.timeout} seconds",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            yield {
                "type": "execution_error",
                "description": f"Strategy execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error_details": traceback.format_exc()
            }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code for security and structural requirements.
        
        Args:
            code: Python code to validate
            
        Returns:
            Dict: Validation results with 'valid' bool and 'details' str
        """
        try:
            # Parse code into AST for analysis
            tree = ast.parse(code)
            
            # Check for security violations
            security_check = self._check_security_violations(tree)
            if not security_check['safe']:
                return {
                    'valid': False,
                    'details': f"Security violation: {security_check['reason']}"
                }
            
            # Check for required structure
            structure_check = self._check_code_structure(tree)
            if not structure_check['valid']:
                return {
                    'valid': False,
                    'details': structure_check['reason']
                }
            
            # Basic syntax validation
            compile(code, '<strategy>', 'exec')
            
            return {'valid': True, 'details': 'Code validation passed'}
            
        except SyntaxError as e:
            return {
                'valid': False,
                'details': f"Syntax error: {str(e)}"
            }
        except Exception as e:
            return {
                'valid': False,
                'details': f"Validation error: {str(e)}"
            }
    
    def _build_execution_globals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build restricted global namespace for code execution.
        
        Args:
            context: Execution context with tools and parameters
            
        Returns:
            Dict: Safe global namespace for code execution
        """
        # Restricted builtins - remove dangerous functions
        safe_builtins = {
            name: getattr(__builtins__, name) 
            for name in dir(__builtins__)
            if not name.startswith('_') and name not in {
                'exec', 'eval', 'compile', 'open', 'input', 
                'raw_input', 'file', 'execfile', 'reload'
            }
        }
        
        # Add safe modules and context
        exec_globals = {
            '__builtins__': safe_builtins,
            'datetime': __import__('datetime'),
            'json': __import__('json'),
            're': __import__('re'),
            'math': __import__('math'),
            **context  # Include search_tool, llm_interface, prompt, etc.
        }
        
        return exec_globals
    
    def _check_security_violations(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Check AST for security violations.
        
        Args:
            tree: Parsed AST of code
            
        Returns:
            Dict: Security check results
        """
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, restricted_modules):
                self.violations = []
                self.restricted_modules = restricted_modules
            
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name in self.restricted_modules:
                        self.violations.append(f"Import of restricted module: {alias.name}")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module and node.module in self.restricted_modules:
                    self.violations.append(f"Import from restricted module: {node.module}")
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'exec', 'eval', 'compile'}:
                        self.violations.append(f"Dangerous function call: {node.func.id}")
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self.restricted_modules)
        visitor.visit(tree)
        
        return {
            'safe': len(visitor.violations) == 0,
            'reason': '; '.join(visitor.violations) if visitor.violations else None
        }
    
    def _check_code_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Check code for required structural elements.
        
        Args:
            tree: Parsed AST of code
            
        Returns:
            Dict: Structure validation results
        """
        class StructureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_execute_strategy = False
                self.has_yield = False
                self.function_params = []
            
            def visit_FunctionDef(self, node):
                if node.name == 'execute_strategy':
                    self.has_execute_strategy = True
                    self.function_params = [arg.arg for arg in node.args.args]
                    
                    # Check for yield statements in function
                    for child in ast.walk(node):
                        if isinstance(child, ast.Yield):
                            self.has_yield = True
                            break
                
                self.generic_visit(node)
        
        visitor = StructureVisitor()
        visitor.visit(tree)
        
        if not visitor.has_execute_strategy:
            return {'valid': False, 'reason': 'Missing execute_strategy function'}
        
        if not visitor.has_yield:
            return {'valid': False, 'reason': 'execute_strategy must use yield statements'}
        
        required_params = {'prompt', 'search_tool', 'llm_interface'}
        if not required_params.issubset(set(visitor.function_params)):
            return {
                'valid': False, 
                'reason': f'execute_strategy must accept parameters: {required_params}'
            }
        
        return {'valid': True}
    
    def _validate_notification(self, notification: Any) -> bool:
        """
        Validate notification format and content.
        
        Args:
            notification: Notification object from strategy
            
        Returns:
            bool: True if notification is valid
        """
        if not isinstance(notification, dict):
            return False
        
        required_fields = {'type', 'description', 'timestamp'}
        if not required_fields.issubset(notification.keys()):
            return False
        
        # Additional validation for specific notification types
        if notification.get('type') == 'final_report':
            if 'report' not in notification:
                return False
        
        return True
    
    @contextmanager
    def _timeout_context(self, seconds: int):
        """
        Context manager for execution timeout.
        
        Args:
            seconds: Timeout duration
        """
        def timeout_handler(signum, frame):
            raise ExecutionTimeout(f"Execution exceeded {seconds} seconds")
        
        # Set up timeout signal
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows fallback - no timeout (or implement threading-based timeout)
            yield


class CodeAnalyzer:
    """Utility class for analyzing generated strategy code."""
    
    @staticmethod
    def extract_function_info(code: str) -> Dict[str, Any]:
        """
        Extract information about the execute_strategy function.
        
        Args:
            code: Python code containing strategy function
            
        Returns:
            Dict: Function analysis results
        """
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'execute_strategy':
                    return {
                        'parameters': [arg.arg for arg in node.args.args],
                        'has_docstring': ast.get_docstring(node) is not None,
                        'line_count': node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0,
                        'yield_count': sum(1 for child in ast.walk(node) if isinstance(child, ast.Yield))
                    }
            
            return {'error': 'execute_strategy function not found'}
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    @staticmethod
    def estimate_complexity(code: str) -> str:
        """
        Estimate complexity of generated strategy code.
        
        Args:
            code: Strategy code to analyze
            
        Returns:
            str: Complexity level (low, medium, high)
        """
        try:
            tree = ast.parse(code)
            
            # Count various complexity indicators
            loop_count = sum(1 for node in ast.walk(tree) 
                           if isinstance(node, (ast.For, ast.While)))
            
            condition_count = sum(1 for node in ast.walk(tree) 
                                if isinstance(node, (ast.If, ast.IfExp)))
            
            call_count = sum(1 for node in ast.walk(tree) 
                           if isinstance(node, ast.Call))
            
            total_score = loop_count * 3 + condition_count * 2 + call_count
            
            if total_score < 10:
                return 'low'
            elif total_score < 25:
                return 'medium'
            else:
                return 'high'
                
        except Exception:
            return 'unknown'
