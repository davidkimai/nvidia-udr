"""
Research Strategies - Advanced strategy library with search, modification, and combination capabilities.
Implements the paper's recommendation for robust strategy management and customization.
"""

import re
from typing import Dict, List, Optional, Any, Union


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
    Advanced library of research strategies with search, modification, and combination capabilities.
    Implements the paper's recommendation for robust strategy management.
    """
    
    def __init__(self):
        """Initialize strategy library with built-in strategies and advanced metadata."""
        self.strategies = {
            "minimal": {
                "name": "Minimal Research Strategy",
                "description": "Simple 3-phrase search with single report generation",
                "complexity": "low",
                "domain": "general",
                "estimated_time": "2-5 minutes",
                "search_phases": 1,
                "search_phrases": 3,
                "validation_level": "basic",
                "report_format": "markdown",
                "text": MINIMAL_STRATEGY,
                "tags": ["quick", "simple", "single-phase", "basic"]
            },
            "expansive": {
                "name": "Expansive Research Strategy", 
                "description": "Topic-based search with 2 topics and multiple phrases",
                "complexity": "medium",
                "domain": "general",
                "estimated_time": "5-15 minutes",
                "search_phases": 1,
                "search_phrases": 4,
                "validation_level": "moderate",
                "report_format": "markdown",
                "text": EXPANSIVE_STRATEGY,
                "tags": ["balanced", "topic-based", "comprehensive", "moderate"]
            },
            "intensive": {
                "name": "Intensive Research Strategy",
                "description": "Iterative 2-phase search with phrase refinement",
                "complexity": "high",
                "domain": "general",
                "estimated_time": "15-30 minutes", 
                "search_phases": 2,
                "search_phrases": 4,
                "validation_level": "comprehensive",
                "report_format": "markdown",
                "text": INTENSIVE_STRATEGY,
                "tags": ["iterative", "comprehensive", "multi-phase", "advanced"]
            }
        }
        
        # Add domain-specific strategies for enhanced functionality
        self._add_domain_strategies()
    
    def _add_domain_strategies(self):
        """Add domain-specific strategy variants."""
        # Academic research strategy
        self.strategies["academic"] = {
            "name": "Academic Research Strategy",
            "description": "Structured approach with source validation and citations",
            "complexity": "medium",
            "domain": "academic",
            "estimated_time": "10-20 minutes",
            "search_phases": 1,
            "search_phrases": 5,
            "validation_level": "academic",
            "report_format": "academic",
            "text": self._generate_academic_strategy(),
            "tags": ["academic", "citations", "structured", "validation"]
        }
        
        # Technical analysis strategy
        self.strategies["technical"] = {
            "name": "Technical Analysis Strategy",
            "description": "Deep technical investigation with implementation focus",
            "complexity": "high",
            "domain": "technology",
            "estimated_time": "20-40 minutes",
            "search_phases": 2,
            "search_phrases": 6,
            "validation_level": "technical",
            "report_format": "technical",
            "text": self._generate_technical_strategy(),
            "tags": ["technical", "implementation", "detailed", "analysis"]
        }
        
        # Quick fact-finding strategy
        self.strategies["rapid"] = {
            "name": "Rapid Fact-Finding Strategy",
            "description": "Ultra-fast single-query research for quick answers",
            "complexity": "low",
            "domain": "general",
            "estimated_time": "1-2 minutes",
            "search_phases": 1,
            "search_phrases": 1,
            "validation_level": "minimal",
            "report_format": "brief",
            "text": self._generate_rapid_strategy(),
            "tags": ["rapid", "quick", "minimal", "factual"]
        }
    
    def search_strategies(self, **filters) -> List[Dict[str, Any]]:
        """
        Search and filter strategies based on criteria.
        
        Args:
            complexity: Strategy complexity (low, medium, high)
            domain: Target domain (general, academic, technology, etc.)
            max_time: Maximum execution time in minutes
            search_phrases: Number of search phrases
            tags: Required tags (list or single string)
            
        Returns:
            List of matching strategy metadata
        """
        results = []
        
        for strategy_name, strategy_data in self.strategies.items():
            if self._matches_filters(strategy_data, filters):
                result = strategy_data.copy()
                result["strategy_name"] = strategy_name
                results.append(result)
        
        # Sort by complexity and estimated time
        complexity_order = {"low": 1, "medium": 2, "high": 3}
        results.sort(key=lambda x: (complexity_order.get(x["complexity"], 2), x["estimated_time"]))
        
        return results
    
    def modify_strategy(self, base_strategy: str, **modifications) -> str:
        """
        Modify an existing strategy with custom parameters.
        
        Args:
            base_strategy: Name of base strategy to modify
            search_phrases: Number of search phrases
            add_validation: Add cross-validation steps
            report_format: Output format (markdown, academic, technical, brief)
            search_phases: Number of search phases
            
        Returns:
            Modified strategy text
        """
        if base_strategy not in self.strategies:
            raise ValueError(f"Unknown base strategy: {base_strategy}")
        
        base_text = self.strategies[base_strategy]["text"]
        modified_text = base_text
        
        # Apply modifications
        if "search_phrases" in modifications:
            modified_text = self._modify_search_phrases(modified_text, modifications["search_phrases"])
        
        if "add_validation" in modifications and modifications["add_validation"]:
            modified_text = self._add_validation_steps(modified_text)
        
        if "report_format" in modifications:
            modified_text = self._modify_report_format(modified_text, modifications["report_format"])
        
        if "search_phases" in modifications:
            modified_text = self._modify_search_phases(modified_text, modifications["search_phases"])
        
        return modified_text
    
    def combine_strategies(self, strategies: List[str], merge_mode: str = "sequential", **options) -> str:
        """
        Combine multiple strategies into a unified approach.
        
        Args:
            strategies: List of strategy names to combine
            merge_mode: How to combine (sequential, parallel, weighted)
            aggregation: How to aggregate results (simple, weighted)
            
        Returns:
            Combined strategy text
        """
        if not strategies or len(strategies) < 2:
            raise ValueError("Must provide at least 2 strategies to combine")
        
        for strategy in strategies:
            if strategy not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        if merge_mode == "sequential":
            return self._combine_sequential(strategies, options)
        elif merge_mode == "parallel":
            return self._combine_parallel(strategies, options)
        elif merge_mode == "weighted":
            return self._combine_weighted(strategies, options)
        else:
            raise ValueError(f"Unknown merge mode: {merge_mode}")
    
    def create_template(self, base_strategy: str, customizable_fields: List[str] = None) -> str:
        """
        Create a customizable template from existing strategy.
        
        Args:
            base_strategy: Base strategy to use as template
            customizable_fields: Fields that should be customizable
            
        Returns:
            Strategy template with placeholder fields
        """
        if base_strategy not in self.strategies:
            raise ValueError(f"Unknown base strategy: {base_strategy}")
        
        customizable_fields = customizable_fields or ["search_phrases", "report_format"]
        base_text = self.strategies[base_strategy]["text"]
        
        template = f"# Customizable Strategy Template (based on {base_strategy})\n\n"
        template += "# Customization Options:\n"
        
        for field in customizable_fields:
            template += f"# - {field}: [CUSTOMIZE THIS]\n"
        
        template += "\n# Strategy Implementation:\n"
        template += base_text
        
        # Add customization placeholders
        if "search_phrases" in customizable_fields:
            template = re.sub(r'\b\d+\s+search\s+phrases?', "[SEARCH_PHRASES_COUNT] search phrases", template)
        
        if "report_format" in customizable_fields:
            template = re.sub(r'formatted in Markdown', "formatted in [REPORT_FORMAT]", template)
        
        return template
    
    def get_strategy(self, name: str) -> Optional[str]:
        """Get strategy text by name."""
        strategy = self.strategies.get(name.lower())
        return strategy["text"] if strategy else None
    
    def get_strategy_info(self, name: str) -> Optional[Dict]:
        """Get strategy metadata by name."""
        return self.strategies.get(name.lower())
    
    def list_strategies(self) -> List[Dict[str, str]]:
        """List all available strategies with basic info."""
        return [
            {
                "name": name,
                "title": info["name"],
                "description": info["description"],
                "complexity": info["complexity"],
                "domain": info["domain"],
                "estimated_time": info["estimated_time"]
            }
            for name, info in self.strategies.items()
        ]
    
    def _matches_filters(self, strategy_data: Dict, filters: Dict) -> bool:
        """Check if strategy matches filter criteria."""
        for key, value in filters.items():
            if key == "max_time":
                # Parse time string and compare
                time_str = strategy_data.get("estimated_time", "")
                max_minutes = self._parse_time_to_minutes(time_str)
                if max_minutes > value:
                    return False
            elif key == "tags":
                strategy_tags = strategy_data.get("tags", [])
                required_tags = [value] if isinstance(value, str) else value
                if not any(tag in strategy_tags for tag in required_tags):
                    return False
            elif key in strategy_data:
                if strategy_data[key] != value:
                    return False
        
        return True
    
    def _parse_time_to_minutes(self, time_str: str) -> int:
        """Parse time string to minutes for comparison."""
        if not time_str:
            return 0
        
        # Extract numbers from time string
        numbers = re.findall(r'\d+', time_str)
        if numbers:
            return int(numbers[-1])  # Use the larger number if range
        return 0
    
    def _modify_search_phrases(self, strategy_text: str, count: int) -> str:
        """Modify number of search phrases in strategy."""
        # Replace specific numbers with new count
        modified = re.sub(r'\b\d+\s+search\s+phrases?', f"{count} search phrases", strategy_text)
        modified = re.sub(r'produce \d+ search phrases', f"produce {count} search phrases", modified)
        return modified
    
    def _add_validation_steps(self, strategy_text: str) -> str:
        """Add cross-validation steps to strategy."""
        validation_step = """
6.5. Cross-validate information by comparing results across search phrases:
- Identify consistent themes and facts across multiple sources
- Flag contradictory information for additional verification
- Send notification of type "validation_completed" when cross-validation is done
"""
        
        # Insert before report building step
        return strategy_text.replace(
            'Send a notification with type "report_building"',
            validation_step + '\n7. Send a notification with type "report_building"'
        )
    
    def _modify_report_format(self, strategy_text: str, format_type: str) -> str:
        """Modify report format in strategy."""
        format_instructions = {
            "academic": "formatted with proper citations, abstract, and bibliography",
            "technical": "formatted with code examples, diagrams, and implementation details",
            "brief": "formatted as a concise summary with key points only"
        }
        
        instruction = format_instructions.get(format_type, "formatted in Markdown")
        return strategy_text.replace("formatted in Markdown", instruction)
    
    def _modify_search_phases(self, strategy_text: str, phases: int) -> str:
        """Modify number of search phases in strategy."""
        if phases == 1:
            # Convert to single phase
            return re.sub(r'Perform the following \d+ times:', 'Perform the following once:', strategy_text)
        else:
            # Modify phase count
            return re.sub(r'Perform the following \d+ times:', f'Perform the following {phases} times:', strategy_text)
    
    def _combine_sequential(self, strategies: List[str], options: Dict) -> str:
        """Combine strategies sequentially."""
        combined = "# Sequential Combined Strategy\n\n"
        
        for i, strategy_name in enumerate(strategies, 1):
            strategy_text = self.strategies[strategy_name]["text"]
            combined += f"## Phase {i}: {strategy_name.title()} Strategy\n\n"
            combined += strategy_text + "\n\n"
        
        combined += "## Final Integration\n"
        combined += "11. Combine all reports from different phases into unified final report\n"
        combined += "12. Send final notification with integrated results\n"
        
        return combined
    
    def _combine_parallel(self, strategies: List[str], options: Dict) -> str:
        """Combine strategies in parallel execution."""
        combined = "# Parallel Combined Strategy\n\n"
        combined += "1. Execute multiple research approaches simultaneously:\n\n"
        
        for strategy_name in strategies:
            combined += f"   - {strategy_name.title()} approach\n"
        
        combined += "\n2. Aggregate results from all parallel executions\n"
        combined += "3. Cross-validate findings across different approaches\n"
        combined += "4. Generate unified report with best insights from all strategies\n"
        
        return combined
    
    def _combine_weighted(self, strategies: List[str], options: Dict) -> str:
        """Combine strategies with weighted importance."""
        weights = options.get("weights", [1.0] * len(strategies))
        
        combined = "# Weighted Combined Strategy\n\n"
        combined += "1. Execute strategies with different priorities:\n\n"
        
        for strategy_name, weight in zip(strategies, weights):
            combined += f"   - {strategy_name.title()}: weight {weight}\n"
        
        combined += "\n2. Weight results based on strategy reliability and complexity\n"
        combined += "3. Prioritize findings from higher-weighted strategies\n"
        combined += "4. Generate balanced report reflecting weighted contributions\n"
        
        return combined
    
    def _generate_academic_strategy(self) -> str:
        """Generate academic-focused strategy."""
        return """
1. Send notification of academic research initiation
2. Analyze prompt for research scope and academic requirements
3. Generate 5 scholarly search phrases with academic terminology
4. Execute searches prioritizing academic and peer-reviewed sources
5. Validate source credibility and academic authority
6. Cross-reference findings across multiple academic sources
7. Generate report with proper citations and academic formatting
8. Include bibliography and source confidence ratings
"""
    
    def _generate_technical_strategy(self) -> str:
        """Generate technical analysis strategy."""
        return """
1. Send notification of technical analysis initiation
2. Identify technical components and implementation requirements
3. Generate 6 technical search phrases covering theory and practice
4. Search for technical specifications, documentation, and implementations
5. Analyze technical feasibility and implementation approaches
6. Identify potential challenges and solution alternatives
7. Generate technical report with implementation details
8. Include code examples and technical diagrams where applicable
"""
    
    def _generate_rapid_strategy(self) -> str:
        """Generate rapid fact-finding strategy."""
        return """
1. Send notification of rapid research initiation
2. Extract key factual question from prompt
3. Generate single focused search phrase
4. Execute quick search for immediate answers
5. Extract key facts and verify basic accuracy
6. Generate brief factual summary
7. Send completion notification with concise results
"""


# Global strategy library instance
_strategy_library = StrategyLibrary()


# Enhanced convenience functions
def get_builtin_strategies() -> List[Dict[str, str]]:
    """Get list of all available strategies including domain-specific ones."""
    return _strategy_library.list_strategies()


def get_strategy_by_name(name: str) -> Optional[str]:
    """Get strategy text by name."""
    return _strategy_library.get_strategy(name)


def get_strategy_info(name: str) -> Optional[Dict]:
    """Get strategy metadata by name."""
    return _strategy_library.get_strategy_info(name)


def search_strategies(**filters) -> List[Dict[str, Any]]:
    """Search strategies with filters."""
    return _strategy_library.search_strategies(**filters)


def modify_strategy(base_strategy: str, **modifications) -> str:
    """Modify existing strategy with custom parameters."""
    return _strategy_library.modify_strategy(base_strategy, **modifications)


def combine_strategies(strategies: List[str], merge_mode: str = "sequential", **options) -> str:
    """Combine multiple strategies."""
    return _strategy_library.combine_strategies(strategies, merge_mode, **options)


def create_strategy_template(base_strategy: str, customizable_fields: List[str] = None) -> str:
    """Create customizable strategy template."""
    return _strategy_library.create_template(base_strategy, customizable_fields)


def validate_strategy_name(name: str) -> bool:
    """Check if strategy name is valid."""
    return name.lower() in _strategy_library.strategies


def recommend_strategy(query_complexity: str = "medium", 
                      time_constraint: str = "medium",
                      domain: str = "general") -> str:
    """
    Recommend strategy based on requirements.
    
    Args:
        query_complexity: Complexity of research query (low, medium, high)
        time_constraint: Available time (low, medium, high)
        domain: Research domain (general, academic, technology)
        
    Returns:
        Recommended strategy name
    """
    # Enhanced recommendation logic
    filters = {"domain": domain}
    
    if time_constraint == "low":
        filters["complexity"] = "low"
    elif query_complexity == "high" and time_constraint == "high":
        filters["complexity"] = "high"
    else:
        filters["complexity"] = "medium"
    
    strategies = search_strategies(**filters)
    
    if strategies:
        return strategies[0]["strategy_name"]
    
    # Fallback to basic recommendation
    if time_constraint == "low":
        return "rapid"
    elif query_complexity == "high" and time_constraint == "high":
        return "intensive"
    elif domain == "academic":
        return "academic"
    elif domain == "technology":
        return "technical"
    else:
        return "expansive"


def analyze_strategy_text(strategy_text: str) -> Dict[str, Any]:
    """
    Analyze strategy text for characteristics and complexity.
    Enhanced with domain detection and advanced metrics.
    """
    lines = strategy_text.strip().split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Count steps
    step_count = sum(1 for line in non_empty_lines 
                    if line.strip() and line.strip()[0].isdigit())
    
    # Enhanced complexity indicators
    complexity_keywords = {
        'loop': ['times', 'for each', 'iterate', 'repeat'],
        'conditional': ['if', 'when', 'depending', 'based on'],
        'multi_phase': ['phase', 'iteration', 'round'],
        'refinement': ['refine', 'improve', 'update', 'modify'],
        'validation': ['validate', 'verify', 'cross-reference', 'confirm'],
        'academic': ['citation', 'peer-reviewed', 'scholarly', 'bibliography'],
        'technical': ['implementation', 'code', 'technical', 'specification']
    }
    
    text_lower = strategy_text.lower()
    complexity_score = 0
    found_keywords = {}
    domain_indicators = {}
    
    for category, keywords in complexity_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            found_keywords[category] = found
            complexity_score += len(found)
            
            # Track domain indicators
            if category in ['academic', 'technical']:
                domain_indicators[category] = len(found)
    
    # Determine likely domain
    likely_domain = "general"
    if domain_indicators.get('academic', 0) > 2:
        likely_domain = "academic"
    elif domain_indicators.get('technical', 0) > 2:
        likely_domain = "technology"
    
    # Enhanced complexity estimation
    if step_count <= 3 and complexity_score <= 1:
        estimated_complexity = 'low'
    elif step_count > 8 or complexity_score > 4:
        estimated_complexity = 'high'  
    else:
        estimated_complexity = 'medium'
    
    return {
        'step_count': step_count,
        'line_count': len(non_empty_lines),
        'complexity_score': complexity_score,
        'estimated_complexity': estimated_complexity,
        'likely_domain': likely_domain,
        'complexity_indicators': found_keywords,
        'domain_indicators': domain_indicators,
        'word_count': len(strategy_text.split()),
        'has_validation': 'validation' in found_keywords,
        'has_multi_phase': 'multi_phase' in found_keywords
    }
