# Production Error Handling in Kubeflow Pipelines

This guide demonstrates comprehensive error handling patterns for production-ready Kubeflow Pipelines. The implementation includes all the resilience patterns mentioned in your error handling section.

## 🎯 What's Implemented

### Error Handling Patterns

1. **Exponential Backoff with Jitter** - Smart retry logic for transient failures
2. **Circuit Breaker Pattern** - Prevents cascade failures by stopping requests to failing services
3. **Graceful Degradation** - Adapts processing based on available resources
4. **Conditional Execution** - Uses fallback paths when primary operations fail
5. **Exit Handlers** - Guarantees cleanup regardless of pipeline success/failure
6. **Resource-Aware Processing** - Adapts behavior based on memory/CPU availability
7. **Data Quality Validation** - Validates input data and triggers fallbacks

### Error Categories

The implementation classifies errors into categories for appropriate handling:

- **TRANSIENT** - Network timeouts, temporary issues (retry with backoff)
- **DATA_QUALITY** - Bad input data, validation failures (use fallback data)
- **RESOURCE** - OOM, disk full, quota exceeded (graceful degradation)
- **LOGIC** - Bugs in code, assertion failures (fail fast)
- **DEPENDENCY** - External service issues (circuit breaker)

## 📁 Files Structure

```
pipelines/
├── error_handling_components.py    # Reusable error handling components
├── resilient_pipeline.py          # Main production pipeline with all patterns
├── resilient_pipeline.yaml        # Compiled main pipeline
├── error_testing_pipeline.yaml    # Compiled testing pipeline
└── components/
    └── ml_ops_components.py        # Your existing ML components

test_error_handling.py              # Comprehensive test suite
```

## 🔧 Components Overview

### Core Error Handling Components

#### `fetch_data_with_retry`
- Implements exponential backoff with jitter
- Handles network timeouts and connection errors
- Categorizes errors for appropriate retry logic
- Falls back to error data on complete failure

#### `circuit_breaker_check`
- Implements circuit breaker pattern
- Returns boolean indicating if operations should proceed
- Configurable failure thresholds and recovery windows
- Prevents cascade failures

#### `adaptive_data_processing`
- Graceful degradation based on available memory
- Three processing modes: full, basic, minimal
- Monitors system resources using `psutil`
- Adapts complexity based on constraints

#### `validate_data_quality`
- Validates row counts and null percentages
- Returns boolean for conditional pipeline execution
- Handles various data formats (CSV, JSON)
- Provides detailed quality metrics

#### `simulate_unreliable_service`
- Simulates external service failures for testing
- Configurable failure rates
- Used to test retry and circuit breaker logic

#### `cleanup_resources`
- Guaranteed cleanup in exit handlers
- Removes temporary files and resources
- Idempotent and fault-tolerant
- Runs regardless of pipeline success/failure

#### `generate_fallback_data`
- Creates synthetic data when primary sources fail
- Configurable data size and structure
- Lower quality score to indicate synthetic nature

## 🚀 Pipeline Implementations

### Main Resilient Pipeline (`resilient_pipeline.py`)

Demonstrates the complete error handling workflow:

1. **Circuit Breaker Check** - Determines if primary data fetch should be attempted
2. **Conditional Primary Path** - Fetches data with retries if circuit is closed
3. **Data Quality Validation** - Validates fetched data quality
4. **Adaptive Processing** - Processes data with resource-aware degradation
5. **Conditional Fallback Path** - Generates synthetic data if circuit is open
6. **Service Simulation** - Tests unreliable external services
7. **Exit Handler Cleanup** - Guarantees resource cleanup

### Error Testing Pipeline (`error_testing_pipeline.py`)

Specialized pipeline for testing specific error scenarios:

- **Network Timeout Testing** - Forces timeouts to test retry logic
- **High Failure Rate Testing** - Triggers circuit breaker activation
- **Resource Constraint Testing** - Tests graceful degradation
- **Custom Scenario Testing** - Configurable test scenarios

## 🧪 Testing Framework

### Comprehensive Test Suite (`test_error_handling.py`)

The test suite validates all error handling patterns:

```python
# Run the complete test suite
python test_error_handling.py
```

#### Test Categories

1. **Exponential Backoff Test**
   - Uses delayed endpoints to force timeouts
   - Validates retry behavior with increasing delays
   - Confirms fallback activation after max retries

2. **Circuit Breaker Test**
   - Simulates high failure rates
   - Validates circuit opening behavior
   - Confirms fallback path activation

3. **Graceful Degradation Test**
   - Constrains available memory
   - Validates processing mode adaptation
   - Confirms quality score adjustments

4. **Error Scenario Tests**
   - Tests multiple failure patterns
   - Validates recovery mechanisms
   - Analyzes error patterns across runs

#### Test Results Analysis

The test suite provides:
- Individual test results with run IDs
- Error pattern analysis across multiple runs
- Common failure categorization
- Recovery pattern identification
- Detailed run timelines and diagnostics

## 🎮 Usage Examples

### Running the Main Pipeline

```bash
# Compile the pipeline
cd pipelines
python resilient_pipeline.py

# Submit with default parameters
python ../pipeline_runner.py --pipeline resilient_pipeline.yaml --experiment error-handling

# Submit with custom parameters
python ../pipeline_runner.py --pipeline resilient_pipeline.yaml --experiment error-handling \
  --parameters '{"service_failure_rate": 0.8, "circuit_failure_threshold": 1}'
```

### Testing Error Scenarios

```bash
# Test network timeout scenario
python ../pipeline_runner.py --pipeline error_testing_pipeline.yaml --experiment error-testing \
  --parameters '{"test_scenario": "network_timeout", "failure_rate": 0.9}'

# Test circuit breaker scenario
python ../pipeline_runner.py --pipeline error_testing_pipeline.yaml --experiment error-testing \
  --parameters '{"test_scenario": "high_failure_rate", "failure_rate": 1.0}'

# Test resource constraints
python ../pipeline_runner.py --pipeline error_testing_pipeline.yaml --experiment error-testing \
  --parameters '{"test_scenario": "resource_constraint", "failure_rate": 0.5}'
```

### Running the Test Suite

```bash
# Run comprehensive error handling tests
python test_error_handling.py

# Results are saved to: error_handling_test_results_<timestamp>.json
```

## 📊 Monitoring and Analysis

### Using the Run Analyzer

```bash
# Analyze a specific run
python run_analyzer.py --run-id <RUN_ID>

# Get run timeline
python -c "
from run_analyzer import RunAnalyzer
analyzer = RunAnalyzer()
timeline = analyzer.get_run_timeline('<RUN_ID>')
print(timeline)
"

# Diagnose failures
python -c "
from run_analyzer import RunAnalyzer
analyzer = RunAnalyzer()
diagnosis = analyzer.diagnose_failure('<RUN_ID>')
print(diagnosis)
"
```

### Key Metrics to Monitor

1. **Retry Success Rates** - How often retries succeed vs. fail
2. **Circuit Breaker Activations** - Frequency of circuit opening
3. **Degradation Frequency** - How often graceful degradation occurs
4. **Fallback Usage** - Percentage of runs using fallback data
5. **Cleanup Success** - Exit handler execution success rate

## 🔧 Customization

### Adding New Error Handling Components

1. **Follow the Component Pattern**:
   ```python
   @component(base_image="python:3.11", packages_to_install=["required-packages"])
   def your_error_handling_component(
       input_param: str,
       output_path: OutputPath()
   ) -> bool:  # Simple return types work best with conditionals
       # Your error handling logic
       return success_status
   ```

2. **Integrate with Pipeline**:
   ```python
   @dsl.pipeline(name="your-resilient-pipeline")
   def your_pipeline():
       error_check = your_error_handling_component()
       
       with dsl.If(error_check.output == True):
           # Primary path
           primary_task = your_primary_component()
       
       with dsl.If(error_check.output == False):
           # Fallback path
           fallback_task = your_fallback_component()
   ```

### Configuring Error Thresholds

Adjust these parameters based on your requirements:

- `max_retries`: Number of retry attempts (default: 3)
- `failure_threshold`: Circuit breaker activation threshold (default: 3)
- `recovery_window_minutes`: Circuit breaker recovery time (default: 5)
- `min_memory_gb`: Minimum memory for processing (default: 1.0)
- `target_memory_gb`: Target memory for full processing (default: 4.0)

## 🎯 Best Practices

### Error Handling Strategy

1. **Layer Your Defenses**
   - Component-level retries with exponential backoff
   - Task-level retries for additional safety
   - Circuit breakers for external dependencies
   - Graceful degradation for resource constraints

2. **Fail Fast vs. Retry Smart**
   - Retry transient errors (network, temporary issues)
   - Fail fast on logic errors (bugs, invalid input)
   - Use circuit breakers for dependency failures

3. **Monitor and Alert**
   - Track error patterns across pipeline runs
   - Set up alerts for circuit breaker activations
   - Monitor fallback usage rates
   - Analyze retry success patterns

4. **Test Regularly**
   - Run error scenario tests in CI/CD
   - Validate circuit breaker behavior
   - Test graceful degradation under load
   - Verify cleanup handlers work correctly

### Production Deployment

1. **Resource Limits**
   - Set appropriate memory and CPU limits
   - Use resource requests for guaranteed allocation
   - Monitor resource utilization patterns

2. **Timeout Configuration**
   - Set realistic timeouts for external calls
   - Use progressive timeouts (shorter for retries)
   - Consider downstream service SLAs

3. **Logging and Observability**
   - Log all error handling decisions
   - Include error categories in logs
   - Track retry attempts and outcomes
   - Monitor circuit breaker state changes

## 🔍 Troubleshooting

### Common Issues

1. **Pipeline Compilation Errors**
   - Check component output types match input expectations
   - Ensure `InputPath()` and `OutputPath()` are used correctly
   - Verify conditional logic uses simple boolean returns

2. **Runtime Failures**
   - Check resource limits are sufficient
   - Verify external service endpoints are accessible
   - Ensure cleanup handlers are idempotent

3. **Performance Issues**
   - Adjust retry delays and jitter
   - Optimize circuit breaker thresholds
   - Monitor resource utilization patterns

### Debug Commands

```bash
# Check pipeline compilation
cd pipelines && python resilient_pipeline.py

# Validate component syntax
python -c "from error_handling_components import *; print('Components loaded successfully')"

# Test individual components
python -c "
from error_handling_components import circuit_breaker_check
result = circuit_breaker_check('test', 1, 3, 5)
print(f'Circuit breaker result: {result}')
"
```

## 🎉 Summary

This implementation provides production-ready error handling for Kubeflow Pipelines with:

✅ **Comprehensive Error Patterns** - All major resilience patterns implemented  
✅ **Extensive Testing** - Complete test suite with scenario validation  
✅ **Production Ready** - Resource limits, cleanup, monitoring  
✅ **Highly Configurable** - Adjustable thresholds and parameters  
✅ **Well Documented** - Clear examples and best practices  
✅ **Monitoring Integration** - Works with existing run analyzer tools  

The pipelines demonstrate how to handle failures gracefully, adapt to resource constraints, and maintain system stability under various error conditions. Use this as a foundation for building robust ML workflows that can handle the unpredictability of production environments.
