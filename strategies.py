"""
Built-in Research Strategies - Implementation of minimal, expansive, and intensive strategies
from the Universal Deep Research paper appendix.
"""

from typing import Dict, List, Optional


# Strategy definitions from paper appendix
MINIMAL_STRATEGY = """
1. Send a notification of type "prompt_received" with description saying what PROMPT has been received, e.g. "Received research request: {PROMPT}"

2. Send a notification of type "prompt_analysis_started", with description indicating that we are now analyzing the research request.

3. Take the PROMPT and ask a language model to produce 3 search phrases that could help with retrieving results from search engine for the purpose of compiling a report the user asks for in the PROMPT. The search phrases should be simple and objective, e.g. "important events 1972" or "energy consumption composition in India today". Use a long prompt for the model that describes in detail what is supposed to be performed and the expected output format. Instruct the model to return the search phrases on one line each. Tell the model not to output any other text -- just the newline-separated phrases. Then, parse the output of the language model line by line and save the resulting search phrases as "phrases" for further research, skipping over empty lines.

4. Send a notification of type "prompt_analysis_completed", with a description saying as much.

4.1 Send a notification of type "task_analysis_completed", informing the user that the search plan has been completed and informing them how many search phrases will be invoked, e.g. "Search planning completed. Will be searching through {len(topics)}+ terms."

5. For each phrase in phrases output by step 3., perform the following:
- Send a notification of type "search_started", with the description indicating what search phrase we are using for the search, e.g. "Searching for phrase '{phrase}'"
- Perform search with the phrase.
- Once the search returns some results, append their contents to CONTEXT one by one, separating them by double newlines from what is already present in the CONTEXT.
- Send a notification of type "search_result_processing_completed", indicating in its description that the search results for term {term} have been processed.

6. Send a notification to the user with type "research_completed", indicating that the "Research phase is now completed.".

7. Send a notification with type "report_building", with the description indicating that the report is being built.

8. Take CONTEXT. Call the language model, instructing it to take CONTEXT (to be appended into the LM call) and produce a deep research report on the topic requested in PROMPT. The resulting report should go into detail wherever possible, rely only on the information available in CONTEXT, address the instruction given in the PROMPT, and be formatted in Markdown. This is to be communicated in the prompt. Do not shy away from using long, detailed and descriptive prompts! Tell the model not to output any other text, just the report. The result produced by the language model is to be called REPORT.

9. Send a notification with type "report_done", indicating that the report has been completed. Add "report" as a field containing the REPORT to be an additional payload to the notification.

10. Output the REPORT.
"""

EXPANSIVE_STRATEGY = """
1. Send a notification of type "prompt_received" with description saying what PROMPT has been received, e.g. "Received research request: {PROMPT}"

2. Send a notification of type "prompt_analysis_started", with description indicating that we are now analyzing the research request.

3. Take the PROMPT and ask a language model to produce 2 topics that could be useful to investigate in order to produce the report requested in the PROMPT. The topics should be simple and sufficiently different from each other, e.g. "important events of 1972" or "energy consumption composition in India today". Instruct the model to return the topics on one line each. Tell the model not to output any other text. Then, parse the output of the language model line by line and save the resulting topics as "topics" for further research.

4. Send a notification of type "prompt_analysis_completed", with description saying as much.

5. Throughout the search and report generation process, we shall rely on a single storage of context. Let's refer to it just as to "context" from now on. Initially, there is no context.

6. For each topic in topics, perform the following
6.1. Take the PROMPT and the topic, and ask a language model to produce up to 2 search phrases that could be useful to collect information on the particular topic. Each search phrase should be simple and directly relate to the topic e.g., for topic "important events of 1972", the search phrases could be "what happened in 1972", "1972 events worldwide", "important events 1971-1973". For topic "energy consumption composition in India today", the search phrases could be "renewable energy production in India today", "fossil fuel energy reliance India", "energy security India". Call the returned phrases simply "phrases" from now on.

6.2. For each phrase in phrases output by step 6.1., perform the following:
- Send a notification of type "search_started", with the description indicating what search phrase we are using for the search, e.g. "Searching for phrase '{phrase}'"
- Perform search with the phrase. Once the search returns some results, append their contents to context one by one, separating them by double newlines from what is already present in the context.
- Send a notification of type "search_result_processing_completed", indicating in its description that the search results for term {term} have been processed.

7. Send a notification with type "report_building", with the description indicating that the report is being built.

8. Take CONTEXT. Call the language model, instructing it to take context (to be appended into the LM call) and produce a deep research report on the topic requested in PROMPT. The resulting report should go into detail wherever possible, rely only on the information available in context, address the instruction given in the PROMPT, and be formatted in Markdown. This is to be communicated in the prompt. Do not shy away from using long, detailed and descriptive prompts! Tell the model not to output any other text, just the report. The result produced by the language model is to be called REPORT.

9. Send a notification with type "report_done", indicating that the report has been completed. Add "report" as a field containing the REPORT to be an additional payload to the notification.

10. Output the REPORT.
"""

INTENSIVE_STRATEGY = """
1. Send a notification of type "prompt_received" with description saying what PROMPT has been received, e.g. "Received research request: {PROMPT}"

2. Send a notification of type "prompt_analysis_started", with description indicating that we are now analyzing the research request.

3. Throughout the search and report generation process, we shall rely on two storages of context. One shall be called "supercontext" and contain all contexts of all resources read throughout the search phase. The other one shall be called "subcontext" and pertain to only one interaction of the search process. At the beginning, both the supercontext and subcontext are empty.

4. Take the PROMPT and ask a language model to produce 2 search phrases that could help with retrieving results from search engine for the purpose of compiling a report the user asks for in the PROMPT. The search phrases should be simple and objective, e.g. "important events 1972" or "energy consumption composition in India today". Use a long prompt for the model that describes in detail what is supposed to be performed and the expected output format. Instruct the model to return the search phrases on one line each. Tell the model not to output any other text -- just the newline-separated phrases. Then, parse the output of the language model line by line and save the resulting search phrases as "phrases" for further research, skipping over empty lines.

4.1. Send a notification of type "prompt_analysis_completed", with a description saying as much.

5. Perform the following 2 times:
- Clear the subcontext.
- For each phrase in phrases, perform the following:
  * Send a notification of type "search_started", with the description indicating what search phrase we are using for the search, e.g. "Searching for phrase '{phrase}'"
  * Perform search with the phrase. Once the search returns some results, append their contents to subcontext one by one, separating them by double newlines from what is already present in the subcontext.
  * Send a notification of type "search_result_processing_completed", indicating in its description that the search results for term {term} have been processed.
- Once the subcontext has been put together by aggregating the contributions due to all search phrases, ask a language model, given the subcontext and the PROMPT given by the user, to come up with 2 more phrases (distinct to phrases that are already in phrases) on the basis of the new subcontext being available. Again, the search phrases should be simple and objective, e.g. "important events 1972" or "energy consumption composition in India today". Use a long prompt for the model that describes in detail what is supposed to be performed and the expected output format. Instruct the model to return the search phrases on one line each. Tell the model not to output any other text -- just the newline-separated phrases. Then, parse the output of the language model line by line and save the resulting search phrases as "phrases" for further research, skipping over empty lines. Clear all the old phrases and let the newly returned phrases by the phrases for the next iteration of this loop.

6. Send a notification with type "report_building", with the description indicating that the report is being built.

7. Take CONTEXT. Call the language model, instructing it to take CONTEXT (to be appended into the LM call) and produce a deep research report on the topic requested in PROMPT. The resulting report should go into detail wherever possible, rely only on the information available in CONTEXT, address the instruction given in the PROMPT, and be formatted in Markdown. This is to be communicated in the prompt. Do not shy away from using long, detailed and descriptive prompts! Tell the model not to output any other text, just the report. The result produced by the language model is to be called REPORT.

8. Send a notification with type "report_done", indicating that the report has been completed. Add "report" as a field containing the REPORT to be an additional payload to the notification.

9. Output the REPORT.
"""


class StrategyLibrary:
    """
    Library of built-in research strategies with metadata and validation.
    """
    
    def __init__(self):
        """Initialize strategy library with built-in strategies."""
        self.strategies = {
            "minimal": {
                "name": "Minimal Research Strategy",
                "description": "Simple 3-phrase search with single report generation",
                "complexity": "low",
                "estimated_time": "2-5 minutes",
                "search_phases": 1,
                "text": MINIMAL_STRATEGY
            },
            "expansive": {
                "name": "Expansive Research Strategy", 
                "description": "Topic-based search with 2 topics and multiple phrases",
                "complexity": "medium",
                "estimated_time": "5-15 minutes",
                "search_phases": 1,
                "text": EXPANSIVE_STRATEGY
            },
            "intensive": {
                "name": "Intensive Research Strategy",
                "description": "Iterative 2-phase search with phrase refinement",
                "complexity": "high",
                "estimated_time": "15-30 minutes", 
                "search_phases": 2,
                "text": INTENSIVE_STRATEGY
            }
        }
    
    def get_strategy(self, name: str) -> Optional[str]:
        """
        Get strategy text by name.
        
        Args:
            name: Strategy name (minimal, expansive, intensive)
            
        Returns:
            Strategy text or None if not found
        """
        strategy = self.strategies.get(name.lower())
        return strategy["text"] if strategy else None
    
    def get_strategy_info(self, name: str) -> Optional[Dict]:
        """
        Get strategy metadata by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy metadata dict or None if not found
        """
        return self.strategies.get(name.lower())
    
    def list_strategies(self) -> List[Dict[str, str]]:
        """
        List all available strategies with basic info.
        
        Returns:
            List of strategy info dictionaries
        """
        return [
            {
                "name": name,
                "title": info["name"],
                "description": info["description"],
                "complexity": info["complexity"]
            }
            for name, info in self.strategies.items()
        ]
    
    def validate_strategy_name(self, name: str) -> bool:
        """
        Check if strategy name exists.
        
        Args:
            name: Strategy name to check
            
        Returns:
            bool: True if strategy exists
        """
        return name.lower() in self.strategies
    
    def get_complexity_info(self, complexity: str) -> Dict[str, str]:
        """
        Get information about complexity levels.
        
        Args:
            complexity: Complexity level (low, medium, high)
            
        Returns:
            Dict with complexity information
        """
        complexity_info = {
            "low": {
                "description": "Single search phase with minimal processing",
                "use_case": "Quick overview or simple fact-finding",
                "trade_offs": "Faster but less comprehensive coverage"
            },
            "medium": {
                "description": "Multi-topic search with organized information gathering", 
                "use_case": "Balanced depth and breadth for most research needs",
                "trade_offs": "Good balance of speed and comprehensiveness"
            },
            "high": {
                "description": "Iterative search with refinement and deep exploration",
                "use_case": "Comprehensive analysis requiring thorough investigation",
                "trade_offs": "More comprehensive but significantly slower"
            }
        }
        
        return complexity_info.get(complexity, {})


# Global strategy library instance
_strategy_library = StrategyLibrary()


# Convenience functions for easy access
def get_builtin_strategies() -> List[Dict[str, str]]:
    """
    Get list of all built-in strategies.
    
    Returns:
        List of strategy info dictionaries
    """
    return _strategy_library.list_strategies()


def get_strategy_by_name(name: str) -> Optional[str]:
    """
    Get strategy text by name.
    
    Args:
        name: Strategy name (minimal, expansive, intensive)
        
    Returns:
        Strategy text or None if not found
    """
    return _strategy_library.get_strategy(name)


def get_strategy_info(name: str) -> Optional[Dict]:
    """
    Get strategy metadata by name.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy metadata dict or None if not found
    """
    return _strategy_library.get_strategy_info(name)


def validate_strategy_name(name: str) -> bool:
    """
    Check if strategy name is valid.
    
    Args:
        name: Strategy name to validate
        
    Returns:
        bool: True if strategy exists
    """
    return _strategy_library.validate_strategy_name(name)


def recommend_strategy(query_complexity: str = "medium", 
                      time_constraint: str = "medium") -> str:
    """
    Recommend strategy based on query and time constraints.
    
    Args:
        query_complexity: Complexity of research query (low, medium, high)
        time_constraint: Available time (low, medium, high)
        
    Returns:
        str: Recommended strategy name
    """
    # Simple recommendation logic
    if time_constraint == "low":
        return "minimal"
    elif query_complexity == "high" and time_constraint == "high":
        return "intensive"
    else:
        return "expansive"


def create_custom_strategy_template(search_phases: int = 1, 
                                  phrases_per_phase: int = 3) -> str:
    """
    Generate template for custom strategy creation.
    
    Args:
        search_phases: Number of search phases
        phrases_per_phase: Search phrases per phase
        
    Returns:
        str: Custom strategy template
    """
    template = f"""
# Custom Research Strategy Template

1. Send a notification of type "prompt_received" with description of received prompt

2. Send a notification of type "prompt_analysis_started" to indicate analysis beginning

3. Generate {phrases_per_phase} search phrases based on the research prompt

4. Send a notification of type "prompt_analysis_completed"

5. For each search phase (total: {search_phases}):
   - Execute searches for all phrases
   - Collect and aggregate results
   - Send progress notifications
   {"- Generate new phrases based on results (for multi-phase)" if search_phases > 1 else ""}

6. Send a notification of type "report_building"

7. Generate comprehensive research report from collected information

8. Send final notification with type "final_report" containing the complete report

# Customize the above steps based on your specific research requirements
"""
    
    return template.strip()


# Strategy analysis utilities
def analyze_strategy_text(strategy_text: str) -> Dict[str, Any]:
    """
    Analyze strategy text for characteristics and complexity.
    
    Args:
        strategy_text: Strategy text to analyze
        
    Returns:
        Dict: Analysis results with complexity indicators
    """
    lines = strategy_text.strip().split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Count steps (lines starting with numbers)
    step_count = sum(1 for line in non_empty_lines 
                    if line.strip() and line.strip()[0].isdigit())
    
    # Look for complexity indicators
    complexity_keywords = {
        'loop': ['times', 'for each', 'iterate', 'repeat'],
        'conditional': ['if', 'when', 'depending', 'based on'],
        'multi_phase': ['phase', 'iteration', 'round'],
        'refinement': ['refine', 'improve', 'update', 'modify']
    }
    
    text_lower = strategy_text.lower()
    complexity_score = 0
    found_keywords = {}
    
    for category, keywords in complexity_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            found_keywords[category] = found
            complexity_score += len(found)
    
    # Estimate complexity
    if step_count <= 5 and complexity_score <= 2:
        estimated_complexity = 'low'
    elif step_count > 10 or complexity_score > 5:
        estimated_complexity = 'high'  
    else:
        estimated_complexity = 'medium'
    
    return {
        'step_count': step_count,
        'line_count': len(non_empty_lines),
        'complexity_score': complexity_score,
        'estimated_complexity': estimated_complexity,
        'complexity_indicators': found_keywords,
        'word_count': len(strategy_text.split())
    }
