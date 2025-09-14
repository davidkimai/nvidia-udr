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
git clone https://github.com/your-username/universal-deep-research
cd universal-deep-research
pip install -r requirements.txt
```

## Security Model

**CRITICAL**: UDR generates and executes user-defined code. Understanding the security model is essential for safe deployment.

### Development Environment (Default)
- Uses AST-based validation to check for dangerous imports and function calls
- Restricted global namespace removes dangerous builtins (`exec`, `eval`, `open`, etc.)
- Timeout protection prevents infinite loops
- **Suitable for**: Development, testing, trusted users, research reproduction

### Production Environment (Recommended)
For production deployment with untrusted strategies, implement additional sandboxing:

```python
# Example production setup with enhanced isolation
from udr import UDR
import subprocess
import tempfile

def production_execute_strategy(strategy, prompt):
    # Write strategy to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(generated_code)
        temp_file = f.name
    
    # Execute in isolated container/VM
    result = subprocess.run([
        'docker', 'run', '--rm', '--network=none',
        '--memory=512m', '--cpus=0.5',
        'python:3.9-slim', 'python', temp_file
    ], capture_output=True, timeout=300)
    
    return result.stdout.decode()
```

### Security Validation Layers
1. **Strategy Text Analysis**: Validates strategy structure and complexity
2. **Code Generation Constraints**: LLM prompted with security requirements
3. **AST Security Check**: Scans generated code for dangerous patterns
4. **Execution Environment**: Restricted globals and timeout protection
5. **Runtime Monitoring**: Notification validation and error handling

### Security Considerations
- **Trusted Strategies Only**: Default setup assumes trusted strategy authors
- **LLM Prompt Injection**: Generated code quality depends on underlying LLM robustness
- **Resource Limits**: Implement memory/CPU limits for production use
- **Network Isolation**: Consider disabling network access for untrusted code

## Research Reproducibility

### Paper Result Reproduction
Reproduce the exact results from the original research paper:

```bash
python reproduce_paper_results.py
```

This script generates reports for all examples in Appendix B, enabling direct comparison with paper outputs.

### Validation Methodology
1. Execute built-in strategies with paper's exact prompts
2. Compare output structure and content quality
3. Measure performance characteristics (timing, complexity)
4. Validate notification sequences and error handling

## Core Features

### Two-Phase Operation

1. **Strategy Processing**: Converts natural language research strategies to executable Python code
2. **Strategy Execution**: Runs generated code in sandboxed environment with real-time progress tracking

### Built-in Strategies

- **Minimal**: Simple 3-phrase search with single report generation (2-5 minutes)
- **Expansive**: Topic-based search with multiple phrases per topic (5-15 minutes)  
- **Intensive**: Iterative 2-phase search with phrase refinement (15-30 minutes)

### Strategy Library Features

Advanced strategy management and customization:

```python
from strategies import StrategyLibrary

library = StrategyLibrary()

# Search strategies by complexity or domain
strategies = library.search_strategies(complexity="medium", domain="technology")

# Modify existing strategies
modified = library.modify_strategy("minimal", 
                                 search_phrases=5, 
                                 add_validation=True)

# Combine strategies
hybrid = library.combine_strategies(["minimal", "expansive"], 
                                  merge_mode="sequential")
```

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

### Production Configuration with Security

```python
from udr import UDR
from tools import create_search_tool, create_llm_interface

# Configure with production security settings
search_tool = create_search_tool(backend="brave", api_key="your-api-key")
llm_interface = create_llm_interface(provider="openai", api_key="your-api-key")

udr = UDR(
    llm_interface=llm_interface,
    search_tool=search_tool,
    execution_timeout=900  # 15 minutes max
)

# Additional security validation
strategy = "Your custom strategy here"
validation = udr.validate_strategy(strategy)

if not validation['valid']:
    raise SecurityError(f"Strategy failed validation: {validation['details']}")

# Safe execution with monitoring
for notification in udr.research(strategy, prompt):
    # Log all notifications for audit trail
    log_security_event(notification)
    
    if notification['type'] == 'final_report':
        break
```

### Strategy Library Advanced Usage

```python
from strategies import StrategyLibrary

library = StrategyLibrary()

# Search and filter strategies
tech_strategies = library.search_strategies(
    domain="technology",
    complexity="medium",
    max_time="15 minutes"
)

# Create template from existing strategy
template = library.create_template("expansive", 
                                 customizable_fields=["search_phrases", "topics"])

# Modify strategy parameters
custom_strategy = library.modify_strategy(
    base_strategy="minimal",
    search_phrases=5,
    add_cross_validation=True,
    report_format="academic"
)

# Combine multiple strategies
hybrid_strategy = library.combine_strategies(
    strategies=["minimal", "expansive"],
    merge_mode="parallel",  # Execute searches in parallel
    aggregation="weighted"  # Weight results by strategy complexity
)
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

Converts natural language strategies to executable code with security validation.

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

#### StrategyLibrary

Advanced strategy management and customization.

```python
class StrategyLibrary:
    def search_strategies(self, **filters) -> List[Dict]
    def modify_strategy(self, base_strategy: str, **modifications) -> str
    def combine_strategies(self, strategies: List[str], **options) -> str
    def create_template(self, base_strategy: str, **options) -> str
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

### Security Guidelines
- Avoid references to file system operations
- Do not request network access beyond provided tools
- Keep computational complexity reasonable
- Include error handling for failed searches
- Validate all external data before processing

### Best Practices
- Use numbered steps for clear structure
- Include progress notifications for user feedback
- Specify search parameters and data collection methods
- Define report format and content requirements
- Handle error cases and edge conditions

## Architecture

### Two-Phase Operation
1. **Phase 1 - Strategy Processing**: Natural language strategy → executable Python code
2. **Phase 2 - Strategy Execution**: Code execution in sandboxed environment

### Security Architecture
```
Natural Language Strategy
         ↓
Strategy Text Validation
         ↓
LLM Code Generation (with constraints)
         ↓
AST Security Analysis
         ↓
Code Structure Validation
         ↓
Sandboxed Execution Environment
         ↓
Runtime Monitoring & Timeout
         ↓
Validated Notifications
```

### Key Components
- **Strategy Processor**: LLM-powered code generation with validation
- **Strategy Executor**: Sandboxed execution with timeout protection
- **Tool Interfaces**: Pluggable search and LLM backends
- **Security Layer**: Multi-layer validation and execution isolation
- **Strategy Library**: Advanced strategy management and customization

## Configuration

### Development Setup (Default)
```python
# Mock implementations for development
from udr import UDR

udr = UDR()  # Uses mock tools by default
```

### Production Setup
```python
# Real services with enhanced security
from udr import UDR
from tools import create_search_tool, create_llm_interface

search_tool = create_search_tool(backend="brave", api_key="your-key")
llm_interface = create_llm_interface(provider="openai", api_key="your-key")

udr = UDR(
    llm_interface=llm_interface,
    search_tool=search_tool,
    execution_timeout=900  # Shorter timeout for production
)
```

### Environment Variables
```bash
# API keys for production backends
export BRAVE_API_KEY="your-brave-search-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# Security settings
export UDR_EXECUTION_TIMEOUT="900"
export UDR_MAX_SEARCH_RESULTS="10"
export UDR_ENABLE_AUDIT_LOG="true"
```

## Testing and Validation

### Run Test Suite
```bash
python tests.py
```

### Reproduce Paper Results
```bash
python reproduce_paper_results.py
```

### Security Testing
```bash
python test_security.py
```

Test categories:
- Strategy processing and validation
- Code execution and security
- Tool interfaces and backends
- Integration and error handling
- Performance and reliability
- Security validation and isolation

## Limitations

1. **Code Generation Dependence**: Reliability depends on underlying LLM quality
2. **Security Model**: Default setup assumes trusted strategy authors
3. **Strategy Design Complexity**: Users must create logically sound strategies
4. **Limited Real-time Interaction**: No mid-execution strategy modification
5. **Production Sandboxing**: Requires additional infrastructure for untrusted code

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Security Contributions
When contributing security-related features:
- Follow the multi-layer validation approach
- Include comprehensive test cases
- Document security implications
- Consider both development and production use cases

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

This implementation provides a practical, accessible version of the UDR system described in the paper, with focus on ease of use, security, and research reproducibility.
