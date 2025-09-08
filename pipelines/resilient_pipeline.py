"""
Resilient Pipeline - Production Error Handling Example
Demonstrates comprehensive error handling patterns in Kubeflow Pipelines:

1. Exponential backoff with jitter
2. Circuit breaker pattern  
3. Graceful degradation
4. Conditional execution for fallbacks
5. Exit handlers for cleanup
6. Resource-aware processing
7. Data quality validation

This pipeline shows how to combine multiple error handling strategies
for production-ready ML workflows.
"""

from kfp import dsl
from kfp.dsl import component
from error_handling_components import (
    fetch_data_with_retry,
    circuit_breaker_check,
    adaptive_data_processing,
    validate_data_quality,
    simulate_unreliable_service,
    cleanup_resources,
    generate_fallback_data
)


@dsl.pipeline(
    name="resilient-error-handling-pipeline",
    description="Demonstrates production error handling patterns with circuit breakers, retries, and graceful degradation"
)
def resilient_pipeline(
    # Data source parameters
    primary_data_url: str = "https://httpbin.org/json",
    fallback_data_rows: int = 50,
    
    # Error handling parameters
    max_fetch_retries: int = 3,
    circuit_failure_threshold: int = 2,
    
    # Processing parameters
    min_memory_gb: float = 1.0,
    target_memory_gb: float = 2.0,
    
    # Data quality thresholds
    min_rows: int = 10,
    max_null_percentage: float = 15.0,
    
    # Service reliability simulation
    service_failure_rate: float = 0.4
):
    """
    A resilient pipeline that demonstrates error handling at multiple levels.
    
    Pipeline Flow:
    1. Check circuit breaker status
    2. Conditionally fetch data or use fallback
    3. Validate data quality
    4. Process data with adaptive resource usage
    5. Simulate unreliable service calls with retries
    6. Clean up resources in exit handler
    """
    
    # === CIRCUIT BREAKER PATTERN ===
    # Check if we should attempt primary data fetch
    circuit_check = circuit_breaker_check(
        service_name="primary_data_api",
        failure_count=1,  # Simulate some previous failures
        failure_threshold=circuit_failure_threshold,
        recovery_window_minutes=5
    )
    
    # === CONDITIONAL EXECUTION FOR FALLBACKS ===
    # Primary path: Fetch data if circuit is closed
    with dsl.If(circuit_check.output == True):
        print("🟢 Circuit closed - attempting primary data fetch")
        
        primary_fetch = fetch_data_with_retry(
            url=primary_data_url,
            max_retries=max_fetch_retries,
            timeout_seconds=30
        )
        # Add task-level retry as additional safety net
        primary_fetch.set_retry(num_retries=2)
        primary_fetch.set_memory_limit('1Gi')
        
        # Validate primary data quality
        primary_quality = validate_data_quality(
            data_path=primary_fetch.outputs["output_path"],
            min_rows=min_rows,
            max_null_percentage=max_null_percentage
        )
        
        # Process primary data with adaptive resource management
        with dsl.If(primary_quality.output == True):
            primary_processing = adaptive_data_processing(
                data_path=primary_fetch.outputs["output_path"],
                min_memory_gb=min_memory_gb,
                target_memory_gb=target_memory_gb
            )
            primary_processing.set_memory_limit('4Gi')
            primary_processing.set_cpu_limit('2')
    
    # Fallback path: Generate synthetic data if circuit is open or primary fails
    with dsl.If(circuit_check.output == False):
        print("🔴 Circuit open - using fallback data")
        
        fallback_data = generate_fallback_data(
            rows=fallback_data_rows
        )
        fallback_data.set_memory_limit('512Mi')  # Minimal resources for fallback
        
        # Process fallback data (always valid by design)
        fallback_processing = adaptive_data_processing(
            data_path=fallback_data.outputs["output_path"],
            min_memory_gb=0.5,  # Lower requirements for fallback
            target_memory_gb=1.0
        )
        fallback_processing.set_memory_limit('2Gi')
    
    # === UNRELIABLE SERVICE SIMULATION ===
    # Simulate calling external services with different reliability
    reliable_service = simulate_unreliable_service(
        failure_rate=0.1,  # 10% failure rate - more reliable
        service_name="reliable_service"
    )
    reliable_service.set_retry(num_retries=2)
    
    unreliable_service = simulate_unreliable_service(
        failure_rate=service_failure_rate,  # Higher failure rate
        service_name="unreliable_service"
    )
    unreliable_service.set_retry(num_retries=4)  # More retries for unreliable service
    
    # === RESOURCE CLEANUP WITH EXIT HANDLER ===
    # Cleanup task that runs regardless of pipeline success/failure
    cleanup_task = cleanup_resources(
        temp_prefix="resilient_pipeline_"
    )
    cleanup_task.set_memory_limit('256Mi')  # Minimal resources for cleanup
    
    # Wrap main pipeline logic in exit handler
    with dsl.ExitHandler(cleanup_task):
        # Establish task dependencies
        reliable_service.after(circuit_check)
        unreliable_service.after(reliable_service)


@dsl.pipeline(
    name="error-scenario-testing-pipeline", 
    description="Pipeline specifically designed to test different error scenarios"
)
def error_testing_pipeline(
    test_scenario: str = "network_timeout",
    failure_rate: float = 0.8
):
    """
    Test pipeline for validating error handling behavior.
    
    Scenarios:
    - network_timeout: Simulates network timeouts
    - high_failure_rate: Tests circuit breaker activation
    - resource_constraint: Tests graceful degradation
    - data_quality_issues: Tests fallback data handling
    """
    
    if test_scenario == "network_timeout":
        # Test exponential backoff with network issues
        fetch_task = fetch_data_with_retry(
            url="https://httpbin.org/delay/10",  # Will timeout
            max_retries=3,
            timeout_seconds=5  # Short timeout to force failures
        )
        fetch_task.set_retry(num_retries=1)
        
    elif test_scenario == "high_failure_rate":
        # Test circuit breaker with high failure rate
        circuit_test = circuit_breaker_check(
            service_name="test_service",
            failure_count=5,  # Above threshold
            failure_threshold=3
        )
        
        # This should not execute due to open circuit
        with dsl.If(circuit_test.output == True):
            failing_service = simulate_unreliable_service(
                failure_rate=1.0,  # Always fails
                service_name="always_failing_service"
            )
        
        # Fallback should execute
        with dsl.If(circuit_test.output == False):
            fallback = generate_fallback_data(rows=20)
            
    elif test_scenario == "resource_constraint":
        # Test graceful degradation with limited resources
        processing_task = adaptive_data_processing(
            data_path="/dev/null",  # Will fail to load, triggers fallback
            min_memory_gb=0.1,
            target_memory_gb=0.2  # Very low memory to force degradation
        )
        processing_task.set_memory_limit('256Mi')  # Constrained resources
        
    else:
        # Default: Test multiple error patterns
        multi_test = simulate_unreliable_service(
            failure_rate=failure_rate,
            service_name=f"test_{test_scenario}"
        )
        multi_test.set_retry(num_retries=3)
    
    # Always include cleanup
    cleanup = cleanup_resources(temp_prefix=f"test_{test_scenario}_")
    with dsl.ExitHandler(cleanup):
        pass  # Main logic is above


if __name__ == "__main__":
    """
    Compile the resilient pipeline for deployment.
    """
    import sys
    import os
    
    # Add the project root to Python path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    from kfp import compiler
    
    # Compile main resilient pipeline
    compiler.Compiler().compile(
        pipeline_func=resilient_pipeline,
        package_path="resilient_pipeline.yaml"
    )
    print("✅ Compiled resilient_pipeline.yaml")
    
    # Compile error testing pipeline
    compiler.Compiler().compile(
        pipeline_func=error_testing_pipeline,
        package_path="error_testing_pipeline.yaml"
    )
    print("✅ Compiled error_testing_pipeline.yaml")
    
    print("\n🚀 To run these pipelines:")
    print("1. Main resilient pipeline:")
    print("   python pipeline_runner.py --pipeline resilient_pipeline.yaml --experiment error-handling")
    print("\n2. Error testing scenarios:")
    print("   python pipeline_runner.py --pipeline error_testing_pipeline.yaml --experiment error-testing")
    print("   # Test different scenarios by changing the test_scenario parameter")
