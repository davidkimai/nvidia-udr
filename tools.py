"""
Tools Module - Search and LLM interface utilities for UDR system.
Provides clean abstractions for external service integration.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus


class SearchTool:
    """
    Web search interface with multiple backend support.
    Default implementation uses placeholder responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, backend: str = "mock"):
        """
        Initialize search tool.
        
        Args:
            api_key: API key for search service
            backend: Search backend to use (mock, brave, etc.)
        """
        self.api_key = api_key
        self.backend = backend
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search and return results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search result dictionaries with title, url, snippet
        """
        # Rate limiting
        self._enforce_rate_limit()
        
        if self.backend == "mock":
            return self._mock_search(query, max_results)
        elif self.backend == "brave":
            return self._brave_search(query, max_results)
        else:
            raise ValueError(f"Unsupported search backend: {self.backend}")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _mock_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Mock search implementation for testing and development.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of mock search results
        """
        mock_results = [
            {
                "title": f"Search Result {i+1} for: {query}",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a mock search result snippet for query '{query}'. "
                          f"It contains relevant information about the search topic. "
                          f"Result number {i+1} provides detailed coverage of the subject.",
                "source": "Mock Search Engine"
            }
            for i in range(min(max_results, 3))
        ]
        
        return mock_results
    
    def _brave_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Brave Search API implementation.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of search results from Brave API
        """
        # Placeholder for actual Brave Search integration
        # Requires proper API implementation
        try:
            import requests
            
            headers = {
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json"
            }
            
            params = {
                "q": query,
                "count": max_results,
                "search_lang": "en",
                "country": "US",
                "safesearch": "moderate"
            }
            
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for result in data.get("web", {}).get("results", []):
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("description", ""),
                        "source": "Brave Search"
                    })
                
                return results[:max_results]
            else:
                # Fall back to mock results on API failure
                return self._mock_search(query, max_results)
                
        except Exception:
            # Fall back to mock results on error
            return self._mock_search(query, max_results)


class LLMInterface(ABC):
    """
    Abstract base class for language model interfaces.
    Enables swapping different LLM providers.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using language model.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the language model.
        
        Returns:
            Dict with model name, version, capabilities
        """
        pass


class MockLLMInterface(LLMInterface):
    """
    Mock LLM interface for testing and development.
    Provides realistic but fake responses.
    """
    
    def __init__(self, response_delay: float = 0.5):
        """
        Initialize mock LLM interface.
        
        Args:
            response_delay: Simulated response delay in seconds
        """
        self.response_delay = response_delay
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate mock response based on prompt content.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Mock generated response
        """
        # Simulate processing delay
        time.sleep(self.response_delay)
        
        # Generate contextual mock responses
        prompt_lower = prompt.lower()
        
        if "search phrases" in prompt_lower:
            return self._generate_search_phrases(prompt)
        elif "research report" in prompt_lower:
            return self._generate_research_report(prompt)
        elif "code" in prompt_lower:
            return self._generate_code_response(prompt)
        else:
            return self._generate_generic_response(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return mock model information."""
        return {
            "name": "MockLLM",
            "version": "1.0.0",
            "capabilities": ["text_generation", "code_generation"],
            "max_tokens": 4096
        }
    
    def _generate_search_phrases(self, prompt: str) -> str:
        """Generate mock search phrases."""
        return """artificial intelligence research
machine learning applications
deep learning neural networks"""
    
    def _generate_research_report(self, prompt: str) -> str:
        """Generate mock research report."""
        return """# Research Report

## Overview
This is a comprehensive research report generated based on the available information.

## Key Findings
1. The topic demonstrates significant complexity and multiple perspectives
2. Current research shows promising developments in the field
3. Several challenges remain to be addressed

## Conclusion
The research indicates substantial progress while highlighting areas for future investigation.
"""
    
    def _generate_code_response(self, prompt: str) -> str:
        """Generate mock code response."""
        return """from datetime import datetime
import json

def execute_strategy(prompt, search_tool, llm_interface, **kwargs):
    # Step 1 - Begin research process
    yield {
        "type": "research_started",
        "description": "Starting research process",
        "timestamp": datetime.now().isoformat()
    }
    
    # Step 2 - Perform search
    search_results = search_tool.search(prompt)
    
    yield {
        "type": "search_completed",
        "description": f"Found {len(search_results)} search results",
        "timestamp": datetime.now().isoformat()
    }
    
    # Step 3 - Generate final report
    report = "# Research Report\\n\\nBased on search results, this report summarizes findings."
    
    yield {
        "type": "final_report",
        "description": "Research completed successfully",
        "timestamp": datetime.now().isoformat(),
        "report": report
    }
"""
    
    def _generate_generic_response(self, prompt: str) -> str:
        """Generate generic mock response."""
        return f"""Based on the provided information, here is a comprehensive response addressing the key points in your request.

The analysis reveals several important considerations:
- Primary factors include multiple interconnected elements
- Secondary considerations involve contextual relationships
- Implications span across various domains and applications

This response demonstrates understanding of the request while providing actionable insights for practical implementation."""


class OpenAIInterface(LLMInterface):
    """
    OpenAI API interface for production use.
    Requires openai library and API key.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7):
        """
        Initialize OpenAI interface.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            temperature: Generation temperature
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai library required for OpenAI interface")
    
    def generate(self, prompt: str, max_tokens: int = 2048, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return OpenAI model information."""
        return {
            "name": self.model,
            "version": "OpenAI API",
            "capabilities": ["text_generation", "code_generation", "analysis"],
            "provider": "OpenAI"
        }


# Utility functions for tool configuration
def create_search_tool(backend: str = "mock", api_key: Optional[str] = None) -> SearchTool:
    """
    Create search tool with specified backend.
    
    Args:
        backend: Search backend (mock, brave)
        api_key: API key for the backend
        
    Returns:
        Configured SearchTool instance
    """
    return SearchTool(api_key=api_key, backend=backend)


def create_llm_interface(provider: str = "mock", **kwargs) -> LLMInterface:
    """
    Create LLM interface with specified provider.
    
    Args:
        provider: LLM provider (mock, openai)
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured LLMInterface instance
    """
    if provider == "mock":
        return MockLLMInterface(**kwargs)
    elif provider == "openai":
        return OpenAIInterface(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def validate_tool_configuration(search_tool: SearchTool, llm_interface: LLMInterface) -> bool:
    """
    Validate tool configuration for UDR system.
    
    Args:
        search_tool: Search tool instance
        llm_interface: LLM interface instance
        
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Test search tool
        test_results = search_tool.search("test query", max_results=1)
        if not isinstance(test_results, list):
            return False
        
        # Test LLM interface
        test_response = llm_interface.generate("test prompt")
        if not isinstance(test_response, str):
            return False
        
        return True
        
    except Exception:
        return False
