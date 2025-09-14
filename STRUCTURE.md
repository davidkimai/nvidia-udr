# Universal Deep Research - Repository Structure

```
universal-deep-research/
├── README.md              # Complete project documentation with examples
├── requirements.txt       # Python dependencies only
├── udr.py                 # Main API - single entry point
├── processor.py           # Strategy processing (natural language to code)
├── executor.py            # Strategy execution (sandboxed code runner)
├── tools.py               # Search and LLM interface utilities
├── strategies.py          # Built-in strategy examples (minimal, expansive, intensive)
├── examples.py            # Usage examples and demonstrations
└── tests.py               # All tests in single file
```

**Design Principles Applied:**
- **Flat structure**: Maximum 1 level of nesting
- **Single responsibility**: Each file has one clear purpose
- **Frictionless imports**: `from udr import UDR` or `import processor`
- **Just works**: All core functionality in 7 files
- **High signal**: No redundant directories or documentation layers
- **Under 500 lines**: Each file focused and concise
