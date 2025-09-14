# Universal Deep Research (UDR)

A generalist agentic system that enables users to create, edit, and refine custom deep research strategies without requiring additional training or fine-tuning.

## Overview

Universal Deep Research addresses three critical limitations in existing deep research tools:

1. **Limited Customization**: Existing tools use rigid research strategies with minimal user control
2. **Non-Interchangeable Models**: Cannot combine different language models with different research tools
3. **Lack of Specialized Strategies**: No support for domain-specific research workflows

UDR solves these problems by converting natural language research strategies into executable code, enabling complete customization of the research process while working with any language model.

## Quick Start

```python
from udr import quick_research

# Simple research using built-in minimal strategy
for notification in quick_research("benefits of renewable energy", "minimal"):
    print(f"[{notification['type']}] {notification['description']}")
    if notification['type'] == 'final_report':
        print(notification['report'])
        break
```

## Installation

```bash
git clone https://github.com/davidkimai/universal-deep-research
cd universal-deep-research
pip install -r requirements.txt
```

## Core Features

### Two-Phase Operation

1. **Strategy Processing**: Converts natural language research strategies to executable Python code
2. **Strategy Execution**: Runs generated code in sandboxed environment with real-time progress tracking

### Built-in Strategies

- **Minimal**: Simple 3-phrase search with single report generation (2-5 minutes)
- **Expansive**: Topic-based search with multiple phrases per topic (5-15 minutes)  
- **Intensive**: Iterative 2-phase search with phrase refinement (15-30 minutes)

### Security Features

- Sandboxed code execution prevents host system access
- Validation of generated code for security violations
- Timeout protection for long-running strategies

## Usage Examples

### Basic Usage with Built-in Strategy

```python
from udr import UDR

# Initialize UDR system
udr = UDR()

# Use built-in expansive strategy
strategy = udr.get_builtin_strategy("expansive")
prompt = "Impact of artificial intelligence on healthcare"

# Execute research with progress tracking
for notification in udr.research(strategy, prompt):
    print(f"Progress: {notification['description']}")
    if notification['type'] == 'final_report':
        print("Research completed!")
        print(notification['report'])
        break
```

### Custom Research Strategy

```python
from udr import custom_research

# Define custom strategy in natural language
custom_strategy = """
1. Generate 3 search phrases related to the research topic
2. For each phrase, perform web search and collect results
3. Analyze results to identify key themes and patterns
4. Cross-reference information across sources
5. Generate comprehensive report with findings and conclusions
"""

prompt = "Sustainable urban development practices"

# Execute custom strategy
for notification in custom_research(custom_strategy, prompt):
    if notification['type'] == 'final_report':
        print(notification['report'])
        break
```

### Advanced Configuration

```python
from udr import UDR
from tools import create_search_tool, create_llm_interface

# Configure custom tools
search_tool = create_search_tool(backend="brave", api_key="your-api-key")
llm_interface = create_llm_interface(provider="openai", api_key="your-api-key")

# Initialize with custom configuration
udr = UDR(
    llm_interface=llm_interface,
    search_tool=search_tool,
    execution_timeout=1800  # 30 minutes
)

# Use intensive strategy for comprehensive research
strategy = udr.get_builtin_strategy("intensive")
prompt = "Climate change impact on global food security"

for notification in udr.research(strategy, prompt):
    print(f"[{notification['timestamp']}] {notification['description']}")
    if notification['type'] == 'final_report':
        break
```

### Strategy Validation

```python
from udr import UDR

udr = UDR()

# Validate strategy before execution
custom_strategy = "1. Search for information\n2. Generate report"
validation = udr.validate_strategy(custom_strategy)

if validation['valid']:
    print("Strategy is valid and ready for execution")
else:
    print(f"Strategy validation failed: {validation['details']}")
```

### Batch Processing

```python
from udr import quick_research

queries = [
    "Remote work productivity trends",
    "Sustainable packaging solutions", 
    "Cybersecurity in IoT devices"
]

results = []
for query in queries:
    for notification in quick_research(query, "minimal"):
        if notification['type'] == 'final_report':
            results.append({
                'query': query,
                'report': notification['report']
            })
            break

print(f"Processed {len(results)} research queries")
```

## API Reference

### Main Classes

#### UDR

Main interface for the Universal Deep Research system.

```python
class UDR:
    def __init__(self, llm_interface=None, search_tool=None, execution_timeout=3600)
    def research(self, strategy: str, prompt: str, **kwargs) -> Generator
    def validate_strategy(self, strategy: str) -> Dict
    def list_builtin_strategies(self) -> List[Dict]
    def get_builtin_strategy(self, name: str) -> Optional[str]
```

#### StrategyProcessor

Converts natural language strategies to executable code.

```python
class StrategyProcessor:
    def process_strategy(self, strategy: str) -> str
    def validate_strategy_text(self, strategy: str) -> dict
```

#### StrategyExecutor

Executes generated strategy code in sandboxed environment.

```python
class StrategyExecutor:
    def execute(self, code: str, context: Dict) -> Generator
    def validate_code(self, code: str) -> Dict
```

### Tool Interfaces

#### SearchTool

Web search interface with multiple backend support.

```python
class SearchTool:
    def __init__(self, api_key=None, backend="mock")
    def search(self, query: str, max_results=5) -> List[Dict]
```

#### LLMInterface

Abstract interface for language models.

```python
class LLMInterface:
    def generate(self, prompt: str, **kwargs) -> str
    def get_model_info(self) -> Dict
```

### Convenience Functions

```python
# Quick research with built-in strategies
quick_research(prompt: str, strategy_name: str = "minimal") -> Generator

# Custom strategy research
custom_research(strategy: str, prompt: str) -> Generator

# Tool creation utilities
create_search_tool(backend: str = "mock", api_key=None) -> SearchTool
create_llm_interface(provider: str = "mock", **kwargs) -> LLMInterface
```

## Built-in Strategy Examples

### Minimal Strategy
Simple approach for quick research needs:
- Generate 3 search phrases
- Execute searches and collect results
- Generate single comprehensive report

### Expansive Strategy
Balanced approach for most research tasks:
- Identify 2 main research topics
- Generate 2 search phrases per topic
- Systematic information gathering
- Structured report generation

### Intensive Strategy
Comprehensive approach for deep research:
- Initial search with 2 phrases
- Iterative refinement over 2 phases
- Dynamic phrase generation based on results
- Extensive cross-validation

## Creating Custom Strategies

Custom strategies are written in natural language and automatically converted to executable code. Follow these guidelines:

### Strategy Structure
```
1. [Notification step] - Send progress notification
2. [Analysis step] - Analyze research prompt
3. [Search planning] - Generate search phrases/topics
4. [Search execution] - Perform searches and collect data
5. [Report generation] - Create final research report
```

### Best Practices
- Use numbered steps for clear structure
- Include progress notifications for user feedback
- Specify search parameters and data collection methods
- Define report format and content requirements
- Handle error cases and edge conditions

### Example Custom Strategy
```
1. Send notification that custom research has started
2. Analyze the research prompt to identify 3 key subtopics
3. For each subtopic, generate 2 specific search phrases
4. Execute searches systematically, collecting and categorizing results
5. Cross-reference findings across subtopics to identify patterns
6. Generate structured report with executive summary and detailed findings
7. Include source citations and confidence levels for each claim
```

## Architecture

### Two-Phase Operation
1. **Phase 1 - Strategy Processing**: Natural language strategy → executable Python code
2. **Phase 2 - Strategy Execution**: Code execution in sandboxed environment

### Key Components
- **Strategy Processor**: LLM-powered code generation with validation
- **Strategy Executor**: Sandboxed execution with timeout protection
- **Tool Interfaces**: Pluggable search and LLM backends
- **Security Layer**: Code validation and execution isolation

### Design Principles
- **Model Agnostic**: Works with any sufficiently capable language model
- **Tool Agnostic**: Supports multiple search and LLM backends
- **Security First**: Sandboxed execution prevents system access
- **User Controlled**: Complete customization of research methodology

## Configuration

### Search Backends
- **Mock**: Development and testing (default)
- **Brave**: Brave Search API integration
- **Custom**: Implement SearchTool interface

### LLM Providers
- **Mock**: Development and testing (default)
- **OpenAI**: GPT-3.5/GPT-4 integration
- **Custom**: Implement LLMInterface interface

### Environment Variables
```bash
# Optional API keys for production use
export BRAVE_API_KEY="your-brave-search-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Testing

Run the comprehensive test suite:

```bash
python tests.py
```

Test categories:
- Strategy processing and validation
- Code execution and security
- Tool interfaces and backends
- Integration and error handling
- Performance and reliability

## Limitations

1. **Code Generation Dependence**: Reliability depends on underlying LLM quality
2. **Strategy Design Complexity**: Users must create logically sound strategies
3. **Limited Real-time Interaction**: No mid-execution strategy modification
4. **Mock Implementations**: Default tools are for development only

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in research, please cite:

```bibtex
@article{belcak2025universal,
  title={Universal Deep Research: Bring Your Own Model and Strategy},
  author={Belcak, Peter and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2509.00244},
  year={2025}
}
```

## Acknowledgments

Based on the research paper "Universal Deep Research: Bring Your Own Model and Strategy" by Peter Belcak and Pavlo Molchanov from NVIDIA Research.

This implementation provides a practical, accessible version of the UDR system described in the paper, with focus on ease of use and extensibility.
