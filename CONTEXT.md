# Universal Deep Research: Bring Your Own Model and Strategy

## Title
Universal Deep Research: Bring Your Own Model and Strategy

## Hypotheses
*Note: Paper does not explicitly state formal hypotheses. The following are inferred from the problem statement and contributions:*

1. **Rigidity Hypothesis**: Existing deep research tools employ rigid research strategies with limited user customization beyond the research prompt
2. **Convertibility Hypothesis**: User-defined natural language research strategies can be effectively converted to executable code without additional model training
3. **Generalizability Hypothesis**: A single system can wrap around any language model and enable custom research strategies without model-specific modifications
4. **Quality Equivalence Hypothesis**: User-defined research strategies can produce research reports of comparable quality to hard-coded systems

## Executive Summary

Universal Deep Research (UDR) addresses three critical limitations in existing deep research tools: (P1) restricted user control over resource hierarchy and cross-validation, (P2) inability to create specialized document research strategies for high-value industries, and (P3) non-interchangeable models between different deep research systems.

UDR introduces a generalist agentic system that converts user-defined natural language research strategies into executable code, enabling complete customization of the research process without requiring additional training or fine-tuning. The system operates in two phases: strategy processing (natural language to code conversion) and strategy execution (isolated code execution with structured notifications).

## Methods

### System Architecture
1. **Two-Phase Operation**:
   - **Phase 1 - Strategy Processing**: Converts natural language research strategies to executable code
   - **Phase 2 - Strategy Execution**: Runs generated code in isolated environment with structured progress notifications

2. **Strategy Conversion Process**:
   - Language model receives strategy with constraints on available functions and code structures
   - Generated function accepts research prompt and returns generator with yield statements for notifications
   - Comments explicitly map code segments to strategy steps to prevent shortcuts and hallucinations

3. **State Management**:
   - Uses persistent code variables instead of growing context windows
   - Enables operation within small context windows (8k tokens sufficient for full workflows)
   - Allows accurate reference to information from earlier steps

4. **Tool Integration**:
   - Synchronous function calls for transparent, deterministic behavior
   - Language model treated as callable utility for localized tasks (summarization, ranking, extraction)
   - Structured notifications via yield statements for real-time progress updates

### Security and Isolation
- Sandboxed execution environment prevents host system access
- Code generation constraints limit potential for prompt injection exploits
- Recommended use of solutions like Piston for production deployment isolation

## Results

### Demonstrated Capabilities
1. **Strategy Variety**: Successfully implemented minimal, expansive, and intensive research strategies
2. **Output Quality**: Generated comprehensive research reports with proper formatting, citations, and structured content
3. **User Interface**: Functional demonstration UI with strategy selection, editing, progress monitoring, and report viewing
4. **Model Compatibility**: System works with different language models (demonstrated with Llama 3.3 70B)

### Example Outputs
- Cultural analysis (African or European swallow airspeed velocity)
- Event summaries (significant events on specific dates)
- Financial analysis (US stock movements with opening/closing data)
- Historical research (Ulysses Grant military leadership and political legacy)

## Statistical Summaries
*Not provided in original paper - primarily a system architecture and demonstration study*

## Code Implementations or Validations
*Original paper does not provide source code. Implementation details focus on architectural descriptions and natural language strategy examples in appendices.*

## Discussion

### Key Contributions
1. **Flexibility**: First system to allow complete user control over research strategy definition
2. **Model Agnostic**: Works with any sufficiently capable language model without modification
3. **Efficiency**: Separates control logic from language model reasoning, reducing computational costs
4. **Transparency**: Generated code is fully interpretable and auditable

### Design Principles
- Strategy step-by-step code generation with explicit comment mapping
- CPU-executable orchestration logic with focused LLM invocations
- Small context window operation through persistent variable state
- Structured notification system for user transparency

## Limitations

1. **Code Generation Dependence**: System reliability depends on underlying language model's code generation quality
2. **Strategy Design Burden**: Users must create logically sound, coherent research strategies
3. **Limited Real-time Interactivity**: No mid-execution user intervention beyond stopping workflow
4. **Strategy Complexity**: Creating sophisticated strategies for complex queries proved tedious for end users

## Implications

### Research Community
- Enables rapid prototyping of custom research methodologies
- Provides framework for comparing different research approaches systematically
- Opens path for natural language programming of agentic behaviors

### Industry Applications
- Addresses enterprise need for specialized document research strategies
- Enables automation of high-value research workloads in finance, legal, healthcare, real estate
- Allows pairing of most competitive models with most competitive research tools

### Methodological Impact
- Demonstrates feasibility of user-controlled agentic behavior programming
- Shows potential for deterministic control over otherwise free-form language model reasoning

## Future Directions

### Recommended Research Areas (from paper)
1. **R1**: Deploy with library of research strategies for modification rather than requiring user creation from scratch
2. **R2**: Explore user control over language model "thinking" or reasoning processes
3. **R3**: Investigate automatic conversion of user prompts to deterministically controlled agents

### Technical Extensions
- Enhanced real-time interactivity and dynamic strategy modification
- Improved strategy validation and coherence checking
- Advanced security and isolation mechanisms for broader deployment
- Integration with specialized domain knowledge bases and tools

## Data Availability
*Not applicable - system architecture and demonstration study*

## References
*Refer to original paper references [1-10]*

## Reproducibility Checklist

### Available Materials
- [x] System architecture clearly described
- [x] Strategy examples provided in appendices
- [x] User interface screenshots and descriptions
- [x] Example inputs and outputs documented

### Missing for Full Reproduction
- [ ] Source code implementation
- [ ] Specific model configurations and parameters
- [ ] Detailed API specifications for tool integration
- [ ] Performance benchmarks and evaluation metrics
- [ ] Dataset of research strategies for validation

### Implementation Requirements
- Language model with code generation capabilities
- Web search functionality
- Sandboxed code execution environment
- Basic user interface for strategy input and result display
