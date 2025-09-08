# Kubeflow Pipelines Development Environment

A complete, production-ready setup for developing and deploying Kubeflow Pipelines with advanced authentication, validation, and versioning capabilities.

## 🚀 Quick Start

### 1. Set Up Environment
```bash
# Create Python 3.11 virtual environment
python3.11 -m venv kfp_env_311
source kfp_env_311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
```bash
# Set up secure credentials (first time only)
python setup_credentials.py

# Source the generated environment file
source kubeflow.env
```

### 3. Compile and Deploy Pipeline
```bash
# Advanced compilation with validation and versioning
python compile_and_run.py
```

## 📁 Project Structure

```
kubeflow-pipelines/
├── 🔐 setup_credentials.py           # Secure credential setup
├── 🚀 kfp_client.py                  # Working pipeline client  
├── 🔧 compile_and_run.py             # Advanced compiler
├── 🔑 dex_auth.py                    # Authentication library
├── 🏃 pipeline_runner.py             # Production pipeline runner
├── 📊 run_analyzer.py                # Advanced run analysis
├── 🧪 test_error_handling.py         # Error handling test suite
├── 📁 pipelines/                     # Your ML pipeline code
│   ├── ml_training_pipeline.py       # Main ML pipeline
│   ├── resilient_pipeline.py         # Production error handling pipeline
│   ├── error_handling_components.py  # Error handling components
│   ├── smart_caching_pipeline.py     # Smart caching example
│   └── components/ml_ops_components.py # ML pipeline components
├── 📁 compiled_pipelines/            # Generated YAML files
├── 📋 COMPILE_PIPELINE_SETUP_GUIDE.md # Detailed compilation docs
├── 📋 ERROR_HANDLING_GUIDE.md        # Production error handling guide
└── 🔒 kubeflow.env.example          # Credential template
```

## 🛠️ Tools Overview

### `setup_credentials.py` - Secure Configuration
- Interactive credential setup
- Environment variable management
- Connection testing
- Prevents hardcoded credentials in source code

### `compile_and_run.py` - Advanced Pipeline Compiler
- Smart compilation with validation
- Versioned outputs with timestamps
- Automatic submission with fallbacks
- Pipeline quality checks and warnings

### `kfp_client.py` - Direct Pipeline Client
- Working Dex authentication
- Direct HTTP API calls
- Complete pipeline lifecycle support
- Bypasses KFP client authentication issues

### `pipeline_runner.py` - Production Pipeline Runner 🆕
- Real-time monitoring and status tracking
- Automatic retry logic with exponential backoff
- Timeout handling for long-running pipelines
- Comprehensive error reporting and logging
- Run metadata tracking and storage
- Integration with custom Dex authentication

## 📚 Documentation

- **[Compile Pipeline Setup Guide](COMPILE_PIPELINE_SETUP_GUIDE.md)** - Complete guide for `compile_and_run.py`
- **[Error Handling Guide](ERROR_HANDLING_GUIDE.md)** - Production error handling patterns and testing
- **[kubeflow.env.example](kubeflow.env.example)** - Environment configuration template

## 🛡️ Production Error Handling (NEW! 🆕)

Complete implementation of production-ready error handling patterns:

### Error Handling Patterns
- **Exponential Backoff with Jitter** - Smart retry logic for transient failures
- **Circuit Breaker Pattern** - Prevents cascade failures
- **Graceful Degradation** - Adapts to resource constraints
- **Conditional Execution** - Fallback paths for failures
- **Exit Handlers** - Guaranteed cleanup
- **Resource-Aware Processing** - Memory/CPU adaptive behavior

### Available Pipelines
```bash
# Main resilient pipeline with all error handling patterns
pipelines/resilient_pipeline.yaml

# Error scenario testing pipeline
pipelines/error_testing_pipeline.yaml

# Your existing ML pipeline
pipelines/ml_pipeline.yaml
```

### Quick Test
```bash
# Run comprehensive error handling test suite
python test_error_handling.py

# Test specific error scenarios
python pipeline_runner.py --pipeline pipelines/error_testing_pipeline.yaml \
  --experiment error-testing \
  --parameters '{"test_scenario": "network_timeout", "failure_rate": 0.9}'
```

## 🔧 Usage Examples

### Basic Compilation
```bash
source kfp_env_311/bin/activate
source kubeflow.env
python compile_and_run.py
```

### Direct Pipeline Submission
```bash
source kfp_env_311/bin/activate
source kubeflow.env
python kfp_client.py
```

### Production Pipeline Runner (NEW! 🆕)
```bash
# Run with monitoring and error handling
source kfp_env_311/bin/activate
source kubeflow.env
python pipeline_runner.py

# Or use programmatically
from pipeline_runner import PipelineRunner
runner = PipelineRunner()
result = runner.run_pipeline(
    pipeline_path='compiled_pipelines/ml_training_latest.yaml',
    experiment_name='Production',
    wait_for_completion=True
)
```

### Test Your Setup
```bash
source kfp_env_311/bin/activate
source kubeflow.env
python setup_credentials.py test
```

## 🎯 Key Features

- ✅ **Secure Authentication** - Environment-based credentials, no hardcoding
- ✅ **Dex Integration** - Working authentication with Kubeflow's Dex
- ✅ **Pipeline Validation** - Automatic quality checks and best practices
- ✅ **Version Management** - Timestamped outputs with convenient symlinks
- ✅ **Intelligent Fallbacks** - Multiple submission methods with graceful degradation
- ✅ **Production Ready** - Enterprise-grade tooling and error handling
- ✅ **Pipeline Monitoring** - Real-time status tracking and timeout handling
- ✅ **Run Management** - Comprehensive execution tracking and metadata storage

## 🔍 Troubleshooting

### Authentication Issues
```bash
# Test your credentials
python setup_credentials.py test

# Check environment variables
env | grep KUBEFLOW
```

### Compilation Issues
```bash
# Test pipeline import
python -c "from pipelines.ml_training_pipeline import ml_training_pipeline; print('✅ Import successful')"

# Manual compilation
cd pipelines/
python ml_training_pipeline.py
```

### Connection Issues
```bash
# Check Kubeflow is running
kubectl get pods -n kubeflow

# Verify port-forward
ps aux | grep port-forward

# Test UI access
open http://localhost:8080
```

## 🎉 Success Indicators

Your environment is working when you see:

```bash
$ python compile_and_run.py

✅ Found pipeline: ml-training-pipeline
🔨 Compiling ml_training...
✅ Compiled to: compiled_pipelines/ml_training_*.yaml
✅ Pipeline validation passed
🎉 Compilation complete!
```

## 🔗 Related Resources

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [KFP SDK v2 Reference](https://kubeflow-pipelines.readthedocs.io/)
- [Dex Authentication](https://dexidp.io/)

## 🆘 Support

1. **Check the logs** for specific error messages
2. **Review the validation output** for pipeline issues  
3. **Test credentials** with the setup script
4. **Use manual UI submission** as a reliable fallback
5. **Check the detailed guides** in the documentation files

---

**Ready to build amazing ML pipelines!** 🚀 