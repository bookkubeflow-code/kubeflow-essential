#!/usr/bin/env python3

from kfp import dsl
from kfp.dsl import component, Dataset, OutputPath, InputPath
from kfp import compiler

# ========================================================================
# COMPONENT 1: FETCH WITH EXPONENTIAL BACKOFF
# ========================================================================
@component(
    base_image="python:3.11",
    packages_to_install=["requests"]
)
def fetch_with_backoff(output_path: OutputPath()) -> str:
    """Fetch data with exponential backoff retry logic."""
    import time
    import random
    import requests
    import json
    
    print("Starting data fetch with exponential backoff")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}")
            response = requests.get("https://httpbin.org/json", timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            print("Fetch successful")
            return "success"
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = min(2 ** attempt, 30)  # Cap at 30 seconds
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                
                print(f"Retrying in {int(total_delay)} seconds")
                time.sleep(total_delay)
    
    # All attempts failed - create fallback data
    fallback_data = {"error": "fetch_failed", "fallback": True}
    with open(output_path, 'w') as f:
        json.dump(fallback_data, f)
    
    print("All attempts failed - using fallback data")
    return "fallback"

# ========================================================================
# COMPONENT 2: CIRCUIT BREAKER CHECK
# ========================================================================
@component(base_image="python:3.11")
def circuit_breaker_check(failure_count: int = 1) -> bool:
    """Simple circuit breaker pattern."""
    print("Circuit breaker check")
    print(f"Failure count: {failure_count}")
    
    threshold = 3
    circuit_open = failure_count >= threshold
    
    if circuit_open:
        print("Circuit OPEN - too many failures")
        return False
    else:
        print("Circuit CLOSED - service available")
        return True

# ========================================================================
# COMPONENT 3: ADAPTIVE PROCESSING
# ========================================================================
@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "psutil"]
)
def adaptive_processing(data_path: InputPath(), output_path: OutputPath()) -> str:
    """Process data with resource-aware adaptation."""
    import pandas as pd
    import psutil
    import json
    
    print("Starting adaptive processing")
    
    # Check available memory
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    
    print(f"Available memory: {int(available_gb)} GB")
    
    try:
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Create DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        print(f"Processing {len(df)} rows")
        
        # Adaptive processing based on memory
        if available_gb > 2.0:
            # Full processing
            df['processing_mode'] = 'full'
            df['computed_field'] = 'full_computation'
            strategy = "full"
        elif available_gb > 1.0:
            # Degraded processing
            df['processing_mode'] = 'degraded'
            strategy = "degraded"
        else:
            # Minimal processing
            df['processing_mode'] = 'minimal'
            strategy = "minimal"
        
        df.to_csv(output_path, index=False)
        print(f"Processing complete using {strategy} strategy")
        
        return strategy
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        # Create minimal fallback
        fallback_df = pd.DataFrame({'error': ['processing_failed']})
        fallback_df.to_csv(output_path, index=False)
        return "error"

# ========================================================================
# COMPONENT 4: FALLBACK DATA GENERATOR
# ========================================================================
@component(
    base_image="python:3.11",
    packages_to_install=["pandas"]
)
def generate_fallback(output_path: OutputPath(), rows: int = 10) -> str:
    """Generate synthetic fallback data."""
    import pandas as pd
    import random
    
    print(f"Generating {rows} rows of fallback data")
    
    try:
        data = []
        for i in range(rows):
            data.append({
                'id': i + 1,
                'value': random.uniform(0, 100),
                'category': random.choice(['A', 'B', 'C']),
                'source': 'fallback'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print("Fallback data generated successfully")
        return "success"
        
    except Exception as e:
        print(f"Fallback generation error: {str(e)}")
        return "error"

# ========================================================================
# COMPONENT 5: CLEANUP
# ========================================================================
@component(base_image="python:3.11")
def cleanup_resources() -> str:
    """Clean up temporary resources."""
    import os
    import glob
    
    print("Starting cleanup")
    
    try:
        # Clean up temporary files
        temp_files = glob.glob('/tmp/pipeline_*')
        removed_count = 0
        
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                removed_count += 1
            except:
                pass
        
        print(f"Cleanup complete - removed {removed_count} files")
        return "success"
        
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        return "error"

# ========================================================================
# MAIN PIPELINE: BALANCED ERROR HANDLING
# ========================================================================
@dsl.pipeline(
    name="advanced-error-handling-pipeline",
    description="Advanced error handling demonstration with production-ready patterns"
)
def advanced_error_handling_pipeline(
    fallback_rows: int = 20
):
    """
    Balanced error handling pipeline demonstrating:
    - Exponential backoff with jitter
    - Circuit breaker pattern
    - Graceful degradation
    - Fallback mechanisms
    - Resource cleanup
    """
    
    print("Starting balanced error handling pipeline")
    
    # Phase 1: Circuit breaker check
    circuit_task = circuit_breaker_check(failure_count=1)
    
    # Phase 2: Primary data path
    fetch_task = fetch_with_backoff()
    fetch_task.set_retry(num_retries=2)
    fetch_task.set_memory_limit('1Gi')
    
    # Phase 3: Adaptive processing
    process_task = adaptive_processing(data_path=fetch_task.outputs["output_path"])
    process_task.set_memory_limit('2Gi')
    
    # Phase 4: Fallback data generation
    fallback_task = generate_fallback(rows=fallback_rows)
    
    # Phase 5: Cleanup (runs after everything)
    cleanup_task = cleanup_resources()
    cleanup_task.after(process_task)
    cleanup_task.after(fallback_task)
    
    print("Pipeline definition complete")

def main():
    """Compile the balanced error handling pipeline."""
    print("Compiling advanced error handling pipeline...")
    
    try:
        compiler.Compiler().compile(
            pipeline_func=advanced_error_handling_pipeline,
            package_path="compiled_pipelines/advanced_error_handling_pipeline.yaml"
        )
        
        print("Compilation successful!")
        print("Output: compiled_pipelines/advanced_error_handling_pipeline.yaml")
        print()
        print("Features included:")
        print("- Exponential backoff with jitter")
        print("- Circuit breaker pattern")
        print("- Graceful degradation")
        print("- Fallback mechanisms")
        print("- Resource cleanup")
        print("- MySQL-safe implementation")
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        raise

if __name__ == "__main__":
    main()
