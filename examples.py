"""
Usage Examples - Practical demonstrations of UDR system functionality.
Shows common usage patterns and integration approaches.
"""

from udr import UDR, quick_research, custom_research
from tools import create_search_tool, create_llm_interface
from strategies import get_strategy_by_name, get_builtin_strategies


def basic_usage_example():
    """
    Basic usage with built-in minimal strategy.
    Demonstrates simplest approach to research.
    """
    print("Basic Usage Example")
    print("=" * 50)
    
    # Quick research using built-in strategy
    prompt = "What are the main benefits of renewable energy?"
    
    print(f"Research prompt: {prompt}")
    print("\nExecuting minimal strategy...")
    
    # Execute research and print progress
    for notification in quick_research(prompt, "minimal"):
        print(f"[{notification['type']}] {notification['description']}")
        
        # Print final report when completed
        if notification['type'] == 'final_report':
            print("\nFinal Report:")
            print("-" * 30)
            print(notification['report'])
            break


def custom_strategy_example():
    """
    Example using custom research strategy.
    Shows how to define and use custom research approaches.
    """
    print("\nCustom Strategy Example")
    print("=" * 50)
    
    # Define custom strategy
    custom_strategy_text = """
    1. Send notification of type "strategy_started" indicating beginning of custom research
    
    2. Generate 2 specific search phrases related to the research topic
    
    3. For each search phrase:
       - Execute search and collect results
       - Send progress notification
       - Store results for analysis
    
    4. Analyze collected information and identify key themes
    
    5. Generate focused report addressing specific aspects of the research question
    
    6. Send final notification with complete analysis
    """
    
    prompt = "Impact of artificial intelligence on healthcare diagnostics"
    
    print(f"Research prompt: {prompt}")
    print("Using custom strategy...")
    
    # Execute with custom strategy
    for notification in custom_research(custom_strategy_text, prompt):
        print(f"[{notification['type']}] {notification['description']}")
        
        if notification['type'] == 'final_report':
            print("\nCustom Strategy Report:")
            print("-" * 30)
            print(notification['report'])
            break


def advanced_configuration_example():
    """
    Advanced configuration with custom tools and parameters.
    Shows full system configuration options.
    """
    print("\nAdvanced Configuration Example")
    print("=" * 50)
    
    # Create custom tool configuration
    search_tool = create_search_tool(backend="mock")
    llm_interface = create_llm_interface(provider="mock", response_delay=0.1)
    
    # Initialize UDR with custom configuration
    udr = UDR(
        llm_interface=llm_interface,
        search_tool=search_tool,
        execution_timeout=1800  # 30 minutes
    )
    
    # Use expansive strategy
    strategy = get_strategy_by_name("expansive")
    prompt = "Analysis of sustainable urban development practices"
    
    print(f"Research prompt: {prompt}")
    print("Using expansive strategy with custom configuration...")
    
    # Execute research with progress tracking
    notifications = []
    for notification in udr.research(strategy, prompt):
        notifications.append(notification)
        print(f"[{notification['type']}] {notification['description']}")
        
        if notification['type'] == 'final_report':
            print(f"\nResearch completed with {len(notifications)} total notifications")
            print("Report preview:")
            print("-" * 30)
            print(notification['report'][:300] + "...")
            break


def strategy_validation_example():
    """
    Example showing strategy validation and analysis.
    Demonstrates validation before execution.
    """
    print("\nStrategy Validation Example")
    print("=" * 50)
    
    udr = UDR()
    
    # Test valid strategy
    valid_strategy = get_strategy_by_name("minimal")
    validation_result = udr.validate_strategy(valid_strategy)
    
    print("Validating minimal strategy:")
    print(f"Valid: {validation_result['valid']}")
    print(f"Details: {validation_result['details']}")
    
    # Test invalid strategy
    invalid_strategy = "This is not a proper research strategy"
    validation_result = udr.validate_strategy(invalid_strategy)
    
    print("\nValidating invalid strategy:")
    print(f"Valid: {validation_result['valid']}")
    print(f"Details: {validation_result['details']}")


def strategy_comparison_example():
    """
    Compare different built-in strategies for same research question.
    Shows trade-offs between complexity levels.
    """
    print("\nStrategy Comparison Example")
    print("=" * 50)
    
    prompt = "Current trends in machine learning research"
    strategies = ["minimal", "expansive"]  # Skip intensive for demo brevity
    
    for strategy_name in strategies:
        print(f"\nTesting {strategy_name} strategy:")
        print("-" * 30)
        
        notification_count = 0
        start_time = None
        
        for notification in quick_research(prompt, strategy_name):
            if notification['type'] == 'strategy_execution_started':
                start_time = notification['timestamp']
            
            notification_count += 1
            print(f"[{notification['type']}] {notification['description']}")
            
            if notification['type'] == 'final_report':
                print(f"\nStrategy: {strategy_name}")
                print(f"Total notifications: {notification_count}")
                print(f"Report length: {len(notification['report'])} characters")
                break


def error_handling_example():
    """
    Demonstrate error handling and recovery patterns.
    Shows system behavior under various failure conditions.
    """
    print("\nError Handling Example")
    print("=" * 50)
    
    udr = UDR()
    
    # Test with empty strategy
    try:
        empty_strategy = ""
        for notification in udr.research(empty_strategy, "test prompt"):
            print(f"[{notification['type']}] {notification['description']}")
            if 'failed' in notification['type']:
                break
    except Exception as e:
        print(f"Caught exception: {e}")
    
    # Test with malformed strategy
    try:
        malformed_strategy = "1. Do something\n2. Do something else\n3. ???"
        validation = udr.validate_strategy(malformed_strategy)
        print(f"\nMalformed strategy validation: {validation}")
    except Exception as e:
        print(f"Validation error: {e}")


def integration_patterns_example():
    """
    Show common integration patterns for embedding UDR in applications.
    Demonstrates different usage paradigms.
    """
    print("\nIntegration Patterns Example")
    print("=" * 50)
    
    # Pattern 1: Async-style with notification handling
    def handle_notification(notification):
        """Custom notification handler for integration."""
        if notification['type'] == 'search_started':
            print(f"Starting search: {notification['description']}")
        elif notification['type'] == 'final_report':
            print("Research completed successfully")
            return notification['report']
        else:
            print(f"Progress: {notification['description']}")
    
    # Pattern 2: Configuration-based approach
    research_config = {
        'strategy': 'minimal',
        'prompt': 'Benefits of cloud computing for small businesses',
        'timeout': 600,
        'backend': 'mock'
    }
    
    print("Configuration-based research:")
    print(f"Config: {research_config}")
    
    # Execute with configuration
    udr = UDR(execution_timeout=research_config['timeout'])
    strategy = get_strategy_by_name(research_config['strategy'])
    
    final_report = None
    for notification in udr.research(strategy, research_config['prompt']):
        result = handle_notification(notification)
        if result:
            final_report = result
            break
    
    if final_report:
        print(f"Integration successful: Report generated ({len(final_report)} chars)")


def performance_analysis_example():
    """
    Analyze performance characteristics of different strategies.
    Demonstrates monitoring and optimization approaches.
    """
    print("\nPerformance Analysis Example")
    print("=" * 50)
    
    import time
    
    strategies = get_builtin_strategies()
    
    for strategy_info in strategies:
        strategy_name = strategy_info['name']
        print(f"\nAnalyzing {strategy_name} strategy:")
        
        # Get strategy metadata
        from strategies import get_strategy_info
        metadata = get_strategy_info(strategy_name)
        
        print(f"Complexity: {metadata['complexity']}")
        print(f"Estimated time: {metadata['estimated_time']}")
        print(f"Search phases: {metadata['search_phases']}")
        
        # Quick validation test
        strategy_text = get_strategy_by_name(strategy_name)
        udr = UDR()
        validation = udr.validate_strategy(strategy_text)
        print(f"Validation: {'PASS' if validation['valid'] else 'FAIL'}")


def batch_processing_example():
    """
    Example of processing multiple research queries in batch.
    Shows scalable usage patterns.
    """
    print("\nBatch Processing Example")
    print("=" * 50)
    
    # Sample research queries
    research_queries = [
        "Impact of remote work on productivity",
        "Sustainable packaging solutions for e-commerce",
        "Cybersecurity challenges in IoT devices"
    ]
    
    results = []
    
    for i, query in enumerate(research_queries, 1):
        print(f"\nProcessing query {i}/{len(research_queries)}: {query}")
        
        # Use minimal strategy for batch processing
        final_report = None
        for notification in quick_research(query, "minimal"):
            if notification['type'] == 'final_report':
                final_report = notification['report']
                break
            elif 'failed' in notification['type']:
                print(f"Query failed: {notification['description']}")
                break
        
        if final_report:
            results.append({
                'query': query,
                'report': final_report,
                'status': 'completed'
            })
            print(f"Query {i} completed successfully")
        else:
            results.append({
                'query': query,
                'report': None,
                'status': 'failed'
            })
    
    print(f"\nBatch processing completed: {len(results)} queries processed")
    successful = sum(1 for r in results if r['status'] == 'completed')
    print(f"Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")


def main():
    """
    Run all examples in sequence.
    Demonstrates complete UDR system capabilities.
    """
    print("Universal Deep Research (UDR) - Usage Examples")
    print("=" * 60)
    
    examples = [
        basic_usage_example,
        custom_strategy_example,
        advanced_configuration_example,
        strategy_validation_example,
        strategy_comparison_example,
        error_handling_example,
        integration_patterns_example,
        performance_analysis_example,
        batch_processing_example
    ]
    
    for example_func in examples:
        try:
            example_func()
            print("\n" + "="*60)
        except Exception as e:
            print(f"\nExample failed: {e}")
            print("="*60)


if __name__ == "__main__":
    main()
