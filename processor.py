"""
Strategy Processor - Converts natural language research strategies to executable code.
Core component of the UDR two-phase operation system.
"""

import re
from typing import Optional
from tools import LLMInterface


class StrategyProcessor:
    """
    Converts user-defined natural language research strategies into executable 
    Python code with proper structure and constraints.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize strategy processor.
        
        Args:
            llm_interface: Language model interface for code generation
        """
        self.llm_interface = llm_interface or LLMInterface()
        
    def process_strategy(self, strategy: str) -> str:
        """
        Convert natural language strategy to executable Python code.
        
        Args:
            strategy: Natural language description of research strategy
            
        Returns:
            str: Python code implementing the strategy as a generator function
            
        Raises:
            ValueError: If strategy is invalid or code generation fails
        """
        if not strategy.strip():
            raise ValueError("Strategy cannot be empty")
            
        # Generate code using LLM with structured prompt
        code_prompt = self._build_code_generation_prompt(strategy)
        generated_code = self.llm_interface.generate(code_prompt)
        
        # Clean and validate generated code
        clean_code = self._clean_generated_code(generated_code)
        self._validate_generated_code(clean_code)
        
        return clean_code
    
    def _build_code_generation_prompt(self, strategy: str) -> str:
        """
        Build structured prompt for code generation with constraints and examples.
        
        Args:
            strategy: Natural language research strategy
            
        Returns:
            str: Complete prompt for LLM code generation
        """
        return f"""
Convert the following research strategy into executable Python code.

STRATEGY TO IMPLEMENT:
{strategy}

REQUIREMENTS:
1. Create a generator function named 'execute_strategy' that yields progress notifications
2. Function must accept: prompt (str), search_tool, llm_interface, and **kwargs
3. Each notification must be a dictionary with 'type', 'description', 'timestamp' fields
4. Use yield statements for all progress updates
5. Final yield must have type 'final_report' with 'report' field containing the result
6. Add explicit comments mapping each code section to strategy steps
7. Handle errors gracefully with appropriate notifications

AVAILABLE TOOLS:
- search_tool.search(query) -> List[Dict] # Returns search results
- llm_interface.generate(prompt) -> str # Generate text using language model
- Standard Python libraries (json, datetime, etc.)

CONSTRAINTS:
- No external network calls except through provided tools
- No file system access
- No subprocess calls
- No infinite loops
- Include timestamp in each notification using datetime.now().isoformat()

CODE STRUCTURE TEMPLATE:
```python
from datetime import datetime
import json

def execute_strategy(prompt, search_tool, llm_interface, **kwargs):
    # Step 1 - [First strategy step description]
    yield {{
        "type": "step_started",
        "description": "Starting first step...",
        "timestamp": datetime.now().isoformat()
    }}
    
    # Implementation for step 1
    # ...
    
    # Continue for each strategy step
    # ...
    
    # Final step - Generate and return report
    yield {{
        "type": "final_report",
        "description": "Research completed",
        "timestamp": datetime.now().isoformat(),
        "report": final_report_content
    }}
```

Generate ONLY the Python code implementing the strategy. Do not include explanations or markdown formatting.
"""
    
    def _clean_generated_code(self, raw_code: str) -> str:
        """
        Clean and format generated code, removing markdown artifacts.
        
        Args:
            raw_code: Raw code from LLM generation
            
        Returns:
            str: Cleaned Python code
        """
        # Remove markdown code blocks
        code = re.sub(r'```python\s*\n?', '', raw_code)
        code = re.sub(r'```\s*$', '', code)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Ensure proper indentation
        lines = code.split('\n')
        if lines and not lines[0].startswith('from ') and not lines[0].startswith('import '):
            # Add necessary imports if missing
            imports = [
                "from datetime import datetime",
                "import json"
            ]
            code = '\n'.join(imports) + '\n\n' + code
        
        return code
    
    def _validate_generated_code(self, code: str) -> None:
        """
        Validate generated code structure and requirements.
        
        Args:
            code: Generated Python code
            
        Raises:
            ValueError: If code doesn't meet requirements
        """
        if not code.strip():
            raise ValueError("Generated code is empty")
            
        # Check for required function
        if 'def execute_strategy(' not in code:
            raise ValueError("Generated code must contain 'execute_strategy' function")
            
        # Check for yield statements
        if 'yield' not in code:
            raise ValueError("Generated code must use yield statements for notifications")
            
        # Check for final report
        if 'final_report' not in code:
            raise ValueError("Generated code must include final_report notification")
            
        # Basic syntax check
        try:
            compile(code, '<strategy>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax errors: {str(e)}")
    
    def validate_strategy_text(self, strategy: str) -> dict:
        """
        Validate strategy text before processing.
        
        Args:
            strategy: Natural language strategy text
            
        Returns:
            dict: Validation results with 'valid' bool and 'issues' list
        """
        issues = []
        
        if not strategy.strip():
            issues.append("Strategy text is empty")
            
        if len(strategy.split()) < 10:
            issues.append("Strategy appears too brief for meaningful implementation")
            
        # Check for basic strategy structure indicators
        strategy_lower = strategy.lower()
        has_steps = any(indicator in strategy_lower for indicator in [
            'step', 'first', 'then', 'next', 'finally', 'search', 'analyze'
        ])
        
        if not has_steps:
            issues.append("Strategy should include clear steps or actions")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


# Utility functions for common strategy patterns
def extract_strategy_steps(strategy: str) -> list:
    """
    Extract numbered or bulleted steps from strategy text.
    
    Args:
        strategy: Natural language strategy
        
    Returns:
        list: Individual strategy steps
    """
    # Look for numbered steps (1., 2., etc.)
    numbered_pattern = r'^\s*\d+\.?\s*(.+)$'
    numbered_steps = re.findall(numbered_pattern, strategy, re.MULTILINE)
    
    if numbered_steps:
        return numbered_steps
        
    # Look for bulleted steps (-, *, •, etc.)
    bullet_pattern = r'^\s*[-*•]\s*(.+)$'
    bullet_steps = re.findall(bullet_pattern, strategy, re.MULTILINE)
    
    if bullet_steps:
        return bullet_steps
        
    # Fall back to sentence splitting
    sentences = [s.strip() for s in strategy.split('.') if s.strip()]
    return sentences[:10]  # Limit to prevent excessive steps


def estimate_strategy_complexity(strategy: str) -> str:
    """
    Estimate complexity level of a research strategy.
    
    Args:
        strategy: Natural language strategy
        
    Returns:
        str: Complexity level (minimal, moderate, intensive)
    """
    word_count = len(strategy.split())
    step_count = len(extract_strategy_steps(strategy))
    
    complexity_indicators = [
        'iterate', 'loop', 'multiple', 'comprehensive', 'detailed',
        'cross-reference', 'validate', 'extensive', 'thorough'
    ]
    
    indicator_count = sum(1 for indicator in complexity_indicators 
                         if indicator in strategy.lower())
    
    if word_count < 100 and step_count <= 5 and indicator_count <= 1:
        return 'minimal'
    elif word_count > 300 or step_count > 10 or indicator_count > 3:
        return 'intensive'
    else:
        return 'moderate'
