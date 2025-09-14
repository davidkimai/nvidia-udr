"""
Strategy Executor - Sandboxed execution with tiered security model.
Provides development-friendly defaults with clear paths to production-grade isolation.

SECURITY MODEL:
- Development: AST validation + restricted globals (sufficient for trusted users)
- Production: External sandboxing via Piston/containers (required for multi-tenant)
- Enterprise: Full isolation with resource limits and network restrictions
"""

import ast
import sys
import traceback
import signal
import subprocess
import tempfile
import json
import os
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod


class ExecutionTimeout(Exception):
    """Raised when strategy execution exceeds timeout limit."""
    pass


class SecurityViolation(Exception):
    """Raised when code contains security violations."""
    pass


class SandboxBackend(ABC):
    """Abstract interface for different sandboxing backends."""
    
    @abstractmethod
    def execute(self, code: str, context: Dict[str, Any], timeout: int) -> Generator[Dict[str, Any], None, None]:
        """Execute code in sandboxed environment."""
        pass
    
    @abstractmethod
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for execution."""
        pass


class ASTSandbox(SandboxBackend):
    """
    AST-based sandbox for development environments.
    
    LIMITATIONS:
    - Cannot prevent all code injection attacks
    - Limited to Python-level restrictions
    - Not suitable for untrusted multi-tenant environments
    - Recommended for development and trusted users only
    """
    
    def __init__(self, timeout: int = 3600):
        self.timeout = timeout
        self.restricted_modules = {
            'subprocess', 'os', 'sys', 'shutil', 'socket', 'urllib',
            'http', 'ftplib', 'smtplib', 'telnetlib', 'multiprocessing',
            'threading', 'asyncio', 'concurrent'
        }
        self.dangerous_builtins = {
            'exec', 'eval', 'compile', 'open', 'input', 'raw_input', 
            'file', 'execfile', 'reload', '__import__'
        }
    
    def execute(self, code: str, context: Dict[str, Any], timeout: int) -> Generator[Dict[str, Any], None, None]:
        """Execute code with AST-based security validation."""
        validation = self.validate_code(code)
        if not validation['valid']:
            yield {
                "type": "security_violation",
                "description": f"Code validation failed: {validation['details']}",
                "timestamp": datetime.now().isoformat()
            }
            return
        
        exec_globals = self._build_execution_globals(context)
        exec_locals = {}
        
        try:
            with self._timeout_context(timeout or self.timeout):
                # Execute code to define strategy function
                exec(code, exec_globals, exec_locals)
                
                if 'execute_strategy' not in exec_locals:
                    yield {
                        "type": "execution_error",
                        "description": "Code must define 'execute_strategy' function",
                        "timestamp": datetime.now().isoformat()
                    }
                    return
                
                strategy_func = exec_locals['execute_strategy']
                
                # Execute strategy and validate notifications
                for notification in strategy_func(**context):
                    if self._validate_notification(notification):
                        yield notification
                    
        except ExecutionTimeout:
            yield {
                "type": "execution_timeout",
                "description": f"Strategy execution exceeded {timeout or self.timeout} seconds",
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
        """Validate code using AST analysis."""
        try:
            tree = ast.parse(code)
            
            # Security check
            security_check = self._check_security_violations(tree)
            if not security_check['safe']:
                return {
                    'valid': False,
                    'details': f"Security violation: {security_check['reason']}"
                }
            
            # Structure check
            structure_check = self._check_code_structure(tree)
            if not structure_check['valid']:
                return {
                    'valid': False,
                    'details': structure_check['reason']
                }
            
            return {'valid': True, 'details': 'AST validation passed'}
            
        except SyntaxError as e:
            return {'valid': False, 'details': f"Syntax error: {str(e)}"}
        except Exception as e:
            return {'valid': False, 'details': f"Validation error: {str(e)}"}
    
    def _build_execution_globals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build restricted global namespace."""
        safe_builtins = {
            name: getattr(__builtins__, name) 
            for name in dir(__builtins__)
            if not name.startswith('_') and name not in self.dangerous_builtins
        }
        
        return {
            '__builtins__': safe_builtins,
            'datetime': __import__('datetime'),
            'json': __import__('json'),
            're': __import__('re'),
            'math': __import__('math'),
            **context
        }
    
    def _check_security_violations(self, tree: ast.AST) -> Dict[str, Any]:
        """Check AST for security violations."""
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, restricted_modules, dangerous_builtins):
                self.violations = []
                self.restricted_modules = restricted_modules
                self.dangerous_builtins = dangerous_builtins
            
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name in self.restricted_modules:
                        self.violations.append(f"Restricted import: {alias.name}")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module and node.module in self.restricted_modules:
                    self.violations.append(f"Restricted import from: {node.module}")
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in self.dangerous_builtins:
                    self.violations.append(f"Dangerous function call: {node.func.id}")
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self.restricted_modules, self.dangerous_builtins)
        visitor.visit(tree)
        
        return {
            'safe': len(visitor.violations) == 0,
            'reason': '; '.join(visitor.violations) if visitor.violations else None
        }
    
    def _check_code_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Validate required code structure."""
        class StructureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_execute_strategy = False
                self.has_yield = False
                self.function_params = []
            
            def visit_FunctionDef(self, node):
                if node.name == 'execute_strategy':
                    self.has_execute_strategy = True
                    self.function_params = [arg.arg for arg in node.args.args]
                    
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
                'reason': f'execute_strategy must accept: {required_params}'
            }
        
        return {'valid': True}
    
    def _validate_notification(self, notification: Any) -> bool:
        """Validate notification format."""
        if not isinstance(notification, dict):
            return False
        
        required_fields = {'type', 'description', 'timestamp'}
        if not required_fields.issubset(notification.keys()):
            return False
        
        if notification.get('type') == 'final_report' and 'report' not in notification:
            return False
        
        return True
    
    @contextmanager
    def _timeout_context(self, seconds: int):
        """Context manager for execution timeout."""
        def timeout_handler(signum, frame):
            raise ExecutionTimeout(f"Execution exceeded {seconds} seconds")
        
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows fallback - limited timeout capability
            yield


class PistonSandbox(SandboxBackend):
    """
    Piston-based sandbox for production environments.
    Provides robust isolation using external execution engine.
    """
    
    def __init__(self, piston_url: str = "http://localhost:2000", timeout: int = 300):
        self.piston_url = piston_url
        self.timeout = timeout
        self.language = "python"
        self.version = "3.9.4"
    
    def execute(self, code: str, context: Dict[str, Any], timeout: int) -> Generator[Dict[str, Any], None, None]:
        """Execute code using Piston API."""
        try:
            import requests
        except ImportError:
            yield {
                "type": "execution_error",
                "description": "Piston sandbox requires 'requests' library",
                "timestamp": datetime.now().isoformat()
            }
            return
        
        # Prepare execution payload
        execution_code = self._prepare_execution_code(code, context)
        
        payload = {
            "language": self.language,
            "version": self.version,
            "files": [{"content": execution_code}],
            "args": [],
            "compile_timeout": 10000,
            "run_timeout": (timeout or self.timeout) * 1000
        }
        
        try:
            response = requests.post(
                f"{self.piston_url}/api/v2/execute",
                json=payload,
                timeout=timeout or self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                yield from self._parse_piston_output(result)
            else:
                yield {
                    "type": "execution_error",
                    "description": f"Piston execution failed: {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            yield {
                "type": "execution_error",
                "description": f"Piston communication failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for Piston execution."""
        # Basic validation - Piston provides the actual sandboxing
        try:
            ast.parse(code)
            return {'valid': True, 'details': 'Code syntax valid for Piston execution'}
        except SyntaxError as e:
            return {'valid': False, 'details': f"Syntax error: {str(e)}"}
    
    def _prepare_execution_code(self, strategy_code: str, context: Dict[str, Any]) -> str:
        """Prepare code for Piston execution with context injection."""
        context_json = json.dumps({
            k: v for k, v in context.items() 
            if k in ['prompt'] or isinstance(v, (str, int, float, bool, list, dict))
        })
        
        execution_wrapper = f"""
import json
from datetime import datetime

# Mock implementations for Piston environment
class MockSearchTool:
    def search(self, query, max_results=5):
        return [{{"title": f"Mock result for {{query}}", "url": "http://example.com", "snippet": "Mock content"}}]

class MockLLMInterface:
    def generate(self, prompt):
        return "Mock LLM response for demonstration purposes"

# Inject context
context = json.loads('''{context_json}''')
context['search_tool'] = MockSearchTool()
context['llm_interface'] = MockLLMInterface()

# Strategy code
{strategy_code}

# Execute strategy
try:
    for notification in execute_strategy(**context):
        print(json.dumps(notification))
except Exception as e:
    error_notification = {{
        "type": "execution_error",
        "description": str(e),
        "timestamp": datetime.now().isoformat()
    }}
    print(json.dumps(error_notification))
"""
        return execution_wrapper
    
    def _parse_piston_output(self, result: Dict) -> Generator[Dict[str, Any], None, None]:
        """Parse Piston execution output into notifications."""
        if result.get('run', {}).get('stdout'):
            output_lines = result['run']['stdout'].strip().split('\n')
            
            for line in output_lines:
                if line.strip():
                    try:
                        notification = json.loads(line)
                        yield notification
                    except json.JSONDecodeError:
                        # Handle non-JSON output
                        yield {
                            "type": "execution_output",
                            "description": line,
                            "timestamp": datetime.now().isoformat()
                        }
        
        if result.get('run', {}).get('stderr'):
            yield {
                "type": "execution_error",
                "description": result['run']['stderr'],
                "timestamp": datetime.now().isoformat()
            }


class ContainerSandbox(SandboxBackend):
    """
    Container-based sandbox for enterprise deployment.
    Provides maximum isolation using Docker containers.
    """
    
    def __init__(self, image: str = "python:3.9-slim", timeout: int = 300):
        self.image = image
        self.timeout = timeout
        self.container_limits = {
            'memory': '512m',
            'cpus': '0.5',
            'network': 'none'
        }
    
    def execute(self, code: str, context: Dict[str, Any], timeout: int) -> Generator[Dict[str, Any], None, None]:
        """Execute code in isolated Docker container."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            execution_code = self._prepare_container_code(code, context)
            f.write(execution_code)
            temp_file = f.name
        
        try:
            # Docker execution command
            docker_cmd = [
                'docker', 'run', '--rm',
                '--memory', self.container_limits['memory'],
                '--cpus', self.container_limits['cpus'],
                '--network', self.container_limits['network'],
                '-v', f'{temp_file}:/app/strategy.py:ro',
                self.image,
                'python', '/app/strategy.py'
            ]
            
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout
            )
            
            yield from self._parse_container_output(result)
            
        except subprocess.TimeoutExpired:
            yield {
                "type": "execution_timeout",
                "description": f"Container execution exceeded {timeout or self.timeout} seconds",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            yield {
                "type": "execution_error",
                "description": f"Container execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            os.unlink(temp_file)
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for container execution."""
        try:
            ast.parse(code)
            return {'valid': True, 'details': 'Code valid for container execution'}
        except SyntaxError as e:
            return {'valid': False, 'details': f"Syntax error: {str(e)}"}
    
    def _prepare_container_code(self, strategy_code: str, context: Dict[str, Any]) -> str:
        """Prepare code for container execution."""
        # Similar to Piston but optimized for container environment
        return self._prepare_execution_code(strategy_code, context)
    
    def _parse_container_output(self, result: subprocess.CompletedProcess) -> Generator[Dict[str, Any], None, None]:
        """Parse container execution output."""
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        notification = json.loads(line)
                        yield notification
                    except json.JSONDecodeError:
                        yield {
                            "type": "execution_output",
                            "description": line,
                            "timestamp": datetime.now().isoformat()
                        }
        
        if result.stderr:
            yield {
                "type": "execution_error",
                "description": result.stderr,
                "timestamp": datetime.now().isoformat()
            }


class StrategyExecutor:
    """
    Main strategy executor with tiered security model.
    Automatically selects appropriate sandbox based on environment and requirements.
    """
    
    def __init__(self, 
                 timeout: int = 3600,
                 security_level: str = "development",
                 sandbox_config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy executor with security configuration.
        
        Args:
            timeout: Maximum execution time in seconds
            security_level: Security level (development, production, enterprise)
            sandbox_config: Configuration for specific sandbox backend
        """
        self.timeout = timeout
        self.security_level = security_level
        self.sandbox_config = sandbox_config or {}
        self.sandbox = self._create_sandbox()
    
    def execute(self, code: str, context: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Execute strategy code using appropriate sandbox.
        
        Args:
            code: Generated strategy code
            context: Execution context with tools and parameters
            
        Yields:
            Dict: Progress notifications from strategy execution
        """
        yield {
            "type": "execution_started",
            "description": f"Starting execution with {self.security_level} security level",
            "timestamp": datetime.now().isoformat()
        }
        
        yield from self.sandbox.execute(code, context, self.timeout)
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code using current sandbox."""
        return self.sandbox.validate_code(code)
    
    def _create_sandbox(self) -> SandboxBackend:
        """Create appropriate sandbox based on security level."""
        if self.security_level == "development":
            return ASTSandbox(self.timeout)
        elif self.security_level == "production":
            return PistonSandbox(**self.sandbox_config)
        elif self.security_level == "enterprise":
            return ContainerSandbox(**self.sandbox_config)
        else:
            raise ValueError(f"Unknown security level: {self.security_level}")
    
    def upgrade_security(self, new_level: str, new_config: Optional[Dict[str, Any]] = None):
        """
        Upgrade security level without changing interface.
        
        Args:
            new_level: New security level
            new_config: New sandbox configuration
        """
        self.security_level = new_level
        if new_config:
            self.sandbox_config.update(new_config)
        self.sandbox = self._create_sandbox()


class CodeAnalyzer:
    """Enhanced code analysis with security assessment."""
    
    @staticmethod
    def analyze_security_risk(code: str) -> Dict[str, Any]:
        """Analyze potential security risks in code."""
        try:
            tree = ast.parse(code)
            
            risk_indicators = {
                'file_operations': 0,
                'network_operations': 0,
                'subprocess_calls': 0,
                'dynamic_execution': 0,
                'imports': []
            }
            
            class RiskVisitor(ast.NodeVisitor):
                def visit_Import(self, node):
                    for alias in node.names:
                        risk_indicators['imports'].append(alias.name)
                        if alias.name in ['os', 'subprocess', 'socket']:
                            risk_indicators[f'{alias.name}_operations'] = risk_indicators.get(f'{alias.name}_operations', 0) + 1
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['exec', 'eval']:
                            risk_indicators['dynamic_execution'] += 1
                        elif node.func.id in ['open']:
                            risk_indicators['file_operations'] += 1
                    
                    self.generic_visit(node)
            
            visitor = RiskVisitor()
            visitor.visit(tree)
            
            # Calculate risk score
            risk_score = (
                risk_indicators['file_operations'] * 3 +
                risk_indicators['network_operations'] * 4 +
                risk_indicators['subprocess_calls'] * 5 +
                risk_indicators['dynamic_execution'] * 5
            )
            
            risk_level = (
                "high" if risk_score > 10 else
                "medium" if risk_score > 5 else
                "low"
            )
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_indicators': risk_indicators,
                'recommended_security': (
                    "enterprise" if risk_level == "high" else
                    "production" if risk_level == "medium" else
                    "development"
                )
            }
            
        except Exception as e:
            return {
                'risk_level': 'unknown',
                'error': str(e),
                'recommended_security': 'production'
            }
    
    @staticmethod
    def extract_function_info(code: str) -> Dict[str, Any]:
        """Extract detailed function information."""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'execute_strategy':
                    return {
                        'parameters': [arg.arg for arg in node.args.args],
                        'has_docstring': ast.get_docstring(node) is not None,
                        'line_count': getattr(node, 'end_lineno', 0) - node.lineno,
                        'yield_count': sum(1 for child in ast.walk(node) if isinstance(child, ast.Yield)),
                        'complexity_estimate': CodeAnalyzer._estimate_function_complexity(node)
                    }
            
            return {'error': 'execute_strategy function not found'}
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    @staticmethod
    def _estimate_function_complexity(node: ast.FunctionDef) -> str:
        """Estimate function complexity based on AST analysis."""
        complexity_score = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                complexity_score += 3
            elif isinstance(child, (ast.If, ast.IfExp)):
                complexity_score += 2
            elif isinstance(child, ast.Call):
                complexity_score += 1
        
        return (
            "low" if complexity_score < 10 else
            "medium" if complexity_score < 25 else
            "high"
        )


# Convenience functions for backward compatibility
def create_executor(security_level: str = "development", **config) -> StrategyExecutor:
    """
    Create strategy executor with specified security level.
    
    Args:
        security_level: development, production, or enterprise
        **config: Security-specific configuration
        
    Returns:
        Configured StrategyExecutor instance
    """
    return StrategyExecutor(security_level=security_level, sandbox_config=config)


def validate_execution_environment() -> Dict[str, Any]:
    """
    Validate current execution environment capabilities.
    
    Returns:
        Dict: Environment assessment and recommendations
    """
    environment = {
        'ast_validation': True,
        'signal_timeout': hasattr(signal, 'SIGALRM'),
        'docker_available': False,
        'piston_available': False,
        'recommended_level': 'development'
    }
    
    # Check Docker availability
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, timeout=5)
        environment['docker_available'] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Check Piston availability (would require actual network check)
    environment['piston_available'] = False  # Conservative default
    
    # Recommend security level
    if environment['docker_available']:
        environment['recommended_level'] = 'enterprise'
    elif environment['piston_available']:
        environment['recommended_level'] = 'production'
    
    return environment
