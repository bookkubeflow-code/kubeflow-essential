"""
Error Handling Components for Kubeflow Pipelines
Demonstrates production-ready error handling patterns including:
- Exponential backoff with jitter
- Circuit breaker pattern
- Graceful degradation
- Resource-aware processing
- Data quality validation
"""

from kfp import dsl
from kfp.dsl import component, InputPath, OutputPath
from typing import Dict, Any
from enum import Enum


class ErrorCategory(Enum):
    TRANSIENT = "transient"       # Network timeouts, temporary issues
    DATA_QUALITY = "data_quality" # Bad input data, validation failures  
    RESOURCE = "resource"         # OOM, disk full, quota exceeded
    LOGIC = "logic"              # Bugs in code, assertion failures
    DEPENDENCY = "dependency"     # External service issues


@component(
    base_image="python:3.11",
    packages_to_install=["requests", "pandas"]
)
def fetch_data_with_retry(
    url: str,
    output_path: OutputPath(),
    max_retries: int = 3,
    timeout_seconds: int = 30
) -> Dict[str, Any]:
    """
    Fetch data from API with exponential backoff and jitter.
    Demonstrates handling of transient network errors.
    """
    import time
    import random
    import requests
    import json
    
    print(f"Fetching data from {url} with max {max_retries} retries")
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            
            # Write successful response to output
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            return {
                "status": "success",
                "attempts": attempt + 1,
                "data_size": len(response.text),
                "error_category": None
            }
            
        except requests.exceptions.Timeout as e:
            error_category = "transient"
            print(f"Timeout error on attempt {attempt + 1}: {e}")
            
        except requests.exceptions.ConnectionError as e:
            error_category = "transient"
            print(f"Connection error on attempt {attempt + 1}: {e}")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                error_category = "transient"  # Server errors are often transient
            else:
                error_category = "logic"  # Client errors usually aren't
            print(f"HTTP error on attempt {attempt + 1}: {e}")
            
        except Exception as e:
            error_category = "dependency"
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        # Only retry for transient errors
        if attempt < max_retries - 1 and error_category == "transient":
            # Exponential backoff with jitter
            base_delay = 2 ** attempt
            max_delay = min(base_delay, 60)  # Cap at 60 seconds
            jitter = random.uniform(0, max_delay * 0.1)  # 10% jitter
            delay = max_delay + jitter
            
            print(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        else:
            break
    
    # All attempts failed
    with open(output_path, 'w') as f:
        f.write('{"error": "failed_to_fetch"}')
    
    return {
        "status": "failed",
        "attempts": max_retries,
        "data_size": 0,
        "error_category": error_category
    }


@component(base_image="python:3.11")
def circuit_breaker_check(
    service_name: str,
    failure_count: int,
    failure_threshold: int = 3,
    recovery_window_minutes: int = 5
) -> bool:
    """
    Implement circuit breaker pattern to prevent cascade failures.
    """
    import time
    
    circuit_open = failure_count >= failure_threshold
    current_time = int(time.time())
    
    if circuit_open:
        print(f"🔴 Circuit OPEN for {service_name} after {failure_count} failures")
        recovery_time = current_time + (recovery_window_minutes * 60)
        should_attempt = False
    else:
        print(f"🟢 Circuit CLOSED for {service_name} ({failure_count}/{failure_threshold} failures)")
        recovery_time = 0
        should_attempt = True
    
    return should_attempt


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "psutil", "numpy"]
)
def adaptive_data_processing(
    data_path: InputPath(),
    output_path: OutputPath(),
    min_memory_gb: float = 1.0,
    target_memory_gb: float = 4.0
) -> Dict[str, Any]:
    """
    Process data with graceful degradation based on available resources.
    Adapts processing complexity based on memory availability.
    """
    import psutil
    import pandas as pd
    import numpy as np
    import json
    
    # Check available memory
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    total_gb = memory_info.total / (1024**3)
    
    print(f"Memory: {available_gb:.2f}GB available / {total_gb:.2f}GB total")
    
    try:
        # Try to load data
        with open(data_path, 'r') as f:
            content = f.read()
        
        # Simulate different data formats
        if content.startswith('{'):
            # JSON data
            data = json.loads(content)
            df = pd.DataFrame([data] if isinstance(data, dict) else data)
        else:
            # Assume CSV-like data
            lines = content.strip().split('\n')
            df = pd.DataFrame([line.split(',') for line in lines])
        
        print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        # Create minimal fallback data
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    # Adaptive processing based on available memory
    if available_gb >= target_memory_gb:
        # Full processing - complex operations
        print("🚀 Full processing mode")
        try:
            df['complex_feature_1'] = df.iloc[:, 0].astype(float) * df.iloc[:, 1].astype(float)
            df['complex_feature_2'] = np.sqrt(df.iloc[:, 0].astype(float) ** 2 + df.iloc[:, 1].astype(float) ** 2)
            df['statistical_feature'] = df.iloc[:, 0].astype(float).rolling(window=3).mean()
            quality_score = 1.0
            strategy = "full"
        except Exception as e:
            print(f"Full processing failed: {e}, falling back to basic")
            df['basic_feature'] = df.iloc[:, 0].astype(str) + "_processed"
            quality_score = 0.7
            strategy = "degraded_from_full"
            
    elif available_gb >= min_memory_gb:
        # Basic processing - simpler operations
        print("⚡ Basic processing mode")
        try:
            df['basic_feature'] = df.iloc[:, 0].astype(str) + "_basic"
            if len(df.columns) > 1:
                df['sum_feature'] = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0) + \
                                   pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0)
            quality_score = 0.7
            strategy = "basic"
        except Exception as e:
            print(f"Basic processing failed: {e}, falling back to minimal")
            df['minimal_feature'] = "processed"
            quality_score = 0.3
            strategy = "degraded_from_basic"
            
    else:
        # Minimal processing - just pass through with basic validation
        print("🔧 Minimal processing mode")
        df['minimal_feature'] = "minimal_processed"
        quality_score = 0.3
        strategy = "minimal"
    
    # Save processed data
    df.to_csv(output_path, index=False)
    
    return {
        "quality_score": quality_score,
        "strategy": strategy,
        "rows_processed": len(df),
        "columns_created": len([col for col in df.columns if 'feature' in col]),
        "available_memory_gb": round(available_gb, 2),
        "memory_utilization": round((total_gb - available_gb) / total_gb * 100, 2)
    }


@component(
    base_image="python:3.11",
    packages_to_install=["pandas"]
)
def validate_data_quality(
    data_path: InputPath(),
    min_rows: int = 10,
    max_null_percentage: float = 20.0
) -> bool:
    """
    Validate data quality and determine if fallback processing is needed.
    """
    import pandas as pd
    import json
    
    try:
        # Try to read as CSV first
        try:
            df = pd.read_csv(data_path)
        except:
            # Fallback to reading as text and parsing
            with open(data_path, 'r') as f:
                content = f.read()
            
            if content.startswith('{'):
                data = json.loads(content)
                df = pd.DataFrame([data] if isinstance(data, dict) else data)
            else:
                lines = content.strip().split('\n')
                df = pd.DataFrame([line.split(',') for line in lines])
        
        row_count = len(df)
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 100
        
        # Quality checks
        has_sufficient_rows = row_count >= min_rows
        has_acceptable_nulls = null_percentage <= max_null_percentage
        is_valid = has_sufficient_rows and has_acceptable_nulls
        
        quality_issues = []
        if not has_sufficient_rows:
            quality_issues.append(f"Insufficient rows: {row_count} < {min_rows}")
        if not has_acceptable_nulls:
            quality_issues.append(f"Too many nulls: {null_percentage:.1f}% > {max_null_percentage}%")
        
        print(f"Data quality check: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        print(f"Rows: {row_count}, Null%: {null_percentage:.1f}%")
        if quality_issues:
            print("Issues:", "; ".join(quality_issues))
        
        return is_valid
        
    except Exception as e:
        print(f"Data validation failed: {e}")
        return False


@component(base_image="python:3.11")
def simulate_unreliable_service(
    failure_rate: float = 0.3,
    service_name: str = "external_api"
) -> Dict[str, Any]:
    """
    Simulate an unreliable external service for testing error handling.
    """
    import random
    import time
    
    # Simulate processing time
    time.sleep(random.uniform(0.5, 2.0))
    
    if random.random() < failure_rate:
        error_types = ["timeout", "connection_error", "server_error", "rate_limit"]
        error_type = random.choice(error_types)
        
        print(f"❌ {service_name} failed with {error_type}")
        raise Exception(f"Simulated {error_type} from {service_name}")
    
    print(f"✅ {service_name} succeeded")
    return {
        "status": "success",
        "service_name": service_name,
        "response_time": round(random.uniform(0.1, 1.0), 3)
    }


@component(base_image="python:3.11")
def cleanup_resources(
    temp_prefix: str = "pipeline_temp_"
) -> Dict[str, Any]:
    """
    Clean up temporary files and resources.
    This runs in exit handlers regardless of pipeline success/failure.
    """
    import os
    import glob
    import tempfile
    
    cleanup_count = 0
    errors = []
    
    # Clean up temporary files
    temp_patterns = [
        f"/tmp/{temp_prefix}*",
        f"{tempfile.gettempdir()}/{temp_prefix}*"
    ]
    
    for pattern in temp_patterns:
        try:
            temp_files = glob.glob(pattern)
            for file_path in temp_files:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleanup_count += 1
                        print(f"🗑️  Removed {file_path}")
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                        cleanup_count += 1
                        print(f"🗑️  Removed directory {file_path}")
                except Exception as e:
                    error_msg = f"Failed to remove {file_path}: {e}"
                    errors.append(error_msg)
                    print(f"⚠️  {error_msg}")
        except Exception as e:
            error_msg = f"Failed to glob pattern {pattern}: {e}"
            errors.append(error_msg)
            print(f"⚠️  {error_msg}")
    
    # Simulate other cleanup operations
    print("🧹 Performing additional cleanup...")
    print("   - Closing database connections")
    print("   - Releasing memory caches")
    print("   - Finalizing logs")
    
    return {
        "files_cleaned": cleanup_count,
        "errors": errors,
        "cleanup_successful": len(errors) == 0
    }


@component(
    base_image="python:3.11",
    packages_to_install=["pandas"]
)
def generate_fallback_data(
    output_path: OutputPath(),
    rows: int = 100
) -> Dict[str, Any]:
    """
    Generate synthetic fallback data when primary data sources fail.
    """
    import pandas as pd
    import random
    
    print(f"🔄 Generating {rows} rows of fallback data")
    
    # Generate synthetic data
    data = {
        'id': range(1, rows + 1),
        'value1': [random.uniform(0, 100) for _ in range(rows)],
        'value2': [random.uniform(0, 50) for _ in range(rows)],
        'category': [random.choice(['A', 'B', 'C']) for _ in range(rows)],
        'is_synthetic': [True] * rows
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    return {
        "rows_generated": rows,
        "data_source": "synthetic",
        "quality_score": 0.5,  # Lower quality since it's synthetic
        "columns": list(df.columns)
    }
