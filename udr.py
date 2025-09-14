"""
Universal Deep Research (UDR) - Main API
Provides single entry point for creating custom deep research workflows.
"""

import json
from typing import Dict, List, Generator, Any, Optional
from processor import StrategyProcessor
from executor import StrategyExecutor
from tools import SearchTool, LLMInterface


class UDR:
    """
    Universal Deep Research system that converts natural language research 
    strategies into executable code and runs them with real-time progress tracking.
    """
    
    def __init__(self, 
                 llm_interface: Optional[LLMInterface] = None,
                 search_tool: Optional[SearchTool] = None,
                 execution_timeout: int = 3600):
        """
        Initialize UDR system.
        
        Args:
            llm_interface: Language model interface for strategy processing and research
            search_tool: Search tool for information retrieval
            execution_timeout: Maximum execution time in seconds
        """
        self.processor = StrategyProcessor(llm_interface)
        self.executor = StrategyExecutor(execution_timeout)
        self.search_tool = search_tool or SearchTool()
        self.llm_interface = llm_interface or LLMInterface()
        
    def research(self, 
                 strategy: str, 
                 prompt: str,
                 **kwargs) -> Generator[Dict[str, Any], None, str]:
        """
        Execute a research workflow using a custom strategy.
        
        Args:
            strategy: Natural language description of research strategy
            prompt: Research question/topic to investigate
            **kwargs: Additional parameters for strategy execution
            
        Yields:
            Dict: Progress notifications with type, description, timestamp
            
        Returns:
            str: Final research report
        """
        
        # Phase 1: Strategy Processing
        yield {
            "type": "strategy_processing_started",
            "description": "Converting research strategy to executable code",
            "timestamp": self._get_timestamp()
        }
        
        try:
            executable_code = self.processor.process_strategy(strategy)
            
            yield {
                "type": "strategy_processing_completed",
                "description": "Strategy successfully converted to code",
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            yield {
                "type": "strategy_processing_failed", 
                "description": f"Failed to process strategy: {str(e)}",
                "timestamp": self._get_timestamp()
            }
            return ""
        
        # Phase 2: Strategy Execution
        yield {
            "type": "strategy_execution_started",
            "description": "Executing research strategy",
            "timestamp": self._get_timestamp()
        }
        
        # Prepare execution context
        execution_context = {
            "search_tool": self.search_tool,
            "llm_interface": self.llm_interface,
            "prompt": prompt,
            **kwargs
        }
        
        try:
            # Execute strategy and yield all notifications
            for notification in self.executor.execute(executable_code, execution_context):
                yield notification
                
                # Return final report if completed
                if notification.get("type") == "final_report":
                    return notification.get("report", "")
                    
        except Exception as e:
            yield {
                "type": "execution_failed",
                "description": f"Strategy execution failed: {str(e)}", 
                "timestamp": self._get_timestamp()
            }
            return ""
        
        return ""
    
    def validate_strategy(self, strategy: str) -> Dict[str, Any]:
        """
        Validate a research strategy without executing it.
        
        Args:
            strategy: Natural language research strategy
            
        Returns:
            Dict: Validation results with success status and details
        """
        try:
            code = self.processor.process_strategy(strategy)
            syntax_check = self.executor.validate_code(code)
            
            return {
                "valid": syntax_check["valid"],
                "details": syntax_check.get("details", "Strategy is valid"),
                "generated_code_preview": code[:500] + "..." if len(code) > 500 else code
            }
            
        except Exception as e:
            return {
                "valid": False,
                "details": f"Strategy validation failed: {str(e)}"
            }
    
    def list_builtin_strategies(self) -> List[Dict[str, str]]:
        """
        Get list of built-in research strategies.
        
        Returns:
            List of strategy dictionaries with name and description
        """
        from strategies import get_builtin_strategies
        return get_builtin_strategies()
    
    def get_builtin_strategy(self, name: str) -> Optional[str]:
        """
        Retrieve a built-in research strategy by name.
        
        Args:
            name: Strategy name (minimal, expansive, intensive)
            
        Returns:
            Strategy text or None if not found
        """
        from strategies import get_strategy_by_name
        return get_strategy_by_name(name)
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience functions for common usage patterns
def quick_research(prompt: str, strategy_name: str = "minimal") -> Generator[Dict[str, Any], None, str]:
    """
    Quick research using built-in strategy.
    
    Args:
        prompt: Research question
        strategy_name: Built-in strategy to use (minimal, expansive, intensive)
        
    Yields:
        Progress notifications
        
    Returns:
        Research report
    """
    udr = UDR()
    strategy = udr.get_builtin_strategy(strategy_name)
    
    if not strategy:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return udr.research(strategy, prompt)


def custom_research(strategy: str, prompt: str) -> Generator[Dict[str, Any], None, str]:
    """
    Research using custom strategy.
    
    Args:
        strategy: Natural language research strategy
        prompt: Research question
        
    Yields:
        Progress notifications
        
    Returns:
        Research report
    """
    udr = UDR()
    return udr.research(strategy, prompt)
