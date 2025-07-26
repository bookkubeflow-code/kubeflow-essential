# Pipeline Compilation Guide

## Overview

`compile_and_run.py` is an **advanced pipeline compiler** that provides enterprise-grade features for managing Kubeflow Pipelines development. It compiles your pipeline code into deployable YAML files with validation, versioning, and automatic submission capabilities.

## 🚀 Quick Start

### 1. Set Up Credentials (First Time Only)

```bash
# Set up secure credentials
python setup_credentials.py

# Source the generated environment file
source kubeflow.env
```

### 2. Compile Your Pipeline

```bash
# Activate your virtual environment
source kfp_env_311/bin/activate

# Run the advanced compiler
python compile_and_run.py
```

## 📋 What It Does

The advanced compiler performs these steps automatically:

1. **🔍 Discovery:** Finds your pipeline (`pipelines/ml_training_pipeline.py`)
2. **🔨 Compilation:** Compiles to versioned YAML files
3. **✅ Validation:** Checks for common issues and best practices
4. **📝 Metadata:** Adds compilation timestamps and details
5. **🔗 Linking:** Creates convenient `latest` symlinks
6. **🚀 Submission:** Attempts automatic submission (with fallback instructions)

## 📁 Output Structure

After running, you'll get organized outputs:

```
compiled_pipelines/
├── ml_training_20250722_143052.yaml    # ✅ Versioned file
└── ml_training_latest.yaml             # 🔗 Always points to newest
```

## 🔧 Features

### Smart Compilation
- **Versioned outputs** with timestamps
- **Latest symlinks** for easy access
- **Rich metadata** embedded in YAML comments

### Pipeline Validation
- Checks container image specifications
- Validates resource limits and requests
- Warns about outdated Python versions
- Reports configuration best practices

### Intelligent Submission
- **Method 1:** Advanced Dex authentication (automatic)
- **Method 2:** Basic KFP client (fallback)
- **Method 3:** Manual UI instructions (always works)

### Error Handling
- Graceful authentication failures
- Clear troubleshooting instructions
- Detailed error reporting

## 📊 Example Output

```bash
$ python compile_and_run.py

✅ Found pipeline: ml-training-pipeline
🎯 Starting advanced pipeline compilation...
🔨 Compiling ml_training...
✅ Compiled to: compiled_pipelines/ml_training_20250722_143052.yaml
✅ Pipeline validation passed

📁 Pipeline files available:
   • Versioned: compiled_pipelines/ml_training_20250722_143052.yaml
   • Latest: compiled_pipelines/ml_training_latest.yaml

🚀 Attempting to submit pipeline...
🔐 Trying advanced Dex authentication...
✅ Pipeline submitted via Dex authentication!
🔗 View at: http://localhost:8080/#/runs/details/abc123

🎉 Compilation complete!
📊 Pipeline ready for execution in Kubeflow UI
```

## 🔧 Configuration Options

### Environment Variables

The compiler respects these environment variables:

```bash
# Kubeflow connection
export KUBEFLOW_ENDPOINT=http://localhost:8080
export KUBEFLOW_USERNAME=user@example.com
export KUBEFLOW_PASSWORD=12341234
export KUBEFLOW_USER_NAMESPACE=kubeflow-user-example-com
```

### Pipeline Parameters

Your pipeline can accept parameters that will be used during submission:

```python
@dsl.pipeline(name="ml-training-pipeline")
def ml_training_pipeline(
    dataset_url: str = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
):
    # Pipeline implementation
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```
❌ Could not import pipeline: No module named 'pipelines'
```

**Solution:** Run from project root directory:
```bash
cd /path/to/kubeflow-pipelines
python compile_and_run.py
```

#### 2. Authentication Failures
```
❌ All programmatic submission methods failed: 401 Unauthorized
```

**Expected behavior** with Dex authentication. The compiler will show manual instructions:

```
📋 Manual Submission Instructions:
==================================================
1. Open browser: http://localhost:8080
2. Login with any email/password (e.g., user@example.com / 12341234)
3. Navigate to: Pipelines → Upload Pipeline
4. Upload file: compiled_pipelines/ml_training_latest.yaml
5. Click 'Create Run' and name it
6. Set parameters if needed and submit
==================================================
```

#### 3. Pipeline Validation Warnings
```
⚠️  Template 'load-data' has no resource limits
⚠️  Template 'train-model' uses outdated Python 3.8
```

**Solution:** Update your pipeline components:
```python
@dsl.component(
    base_image="python:3.11",  # ✅ Use newer Python
    packages_to_install=["pandas", "scikit-learn"]
)
def load_data(...):
    # Add resource management
    dsl.get_pipeline_conf().set_cpu_request('100m')
    dsl.get_pipeline_conf().set_memory_request('512Mi')
```

#### 4. Missing Virtual Environment
```
zsh: command not found: python
```

**Solution:** Activate your environment:
```bash
source kfp_env_311/bin/activate
python compile_and_run.py
```

### Debugging Tips

#### Check Pipeline Syntax
```bash
# Test import manually
python -c "from pipelines.ml_training_pipeline import ml_training_pipeline; print('✅ Import successful')"
```

#### Verify Environment
```bash
# Check environment variables
env | grep KUBEFLOW

# Test credentials
python setup_credentials.py test
```

#### Manual Compilation
```bash
# Compile without submission
cd pipelines/
python ml_training_pipeline.py
```

## 🎯 Best Practices

### 1. Pipeline Development Workflow

```bash
# 1. Edit your pipeline
vim pipelines/ml_training_pipeline.py

# 2. Test compilation
python compile_and_run.py

# 3. Check validation results
cat compiled_pipelines/ml_training_latest.yaml | head -20

# 4. Submit via UI or use the generated file
```

### 2. Version Management

The compiler creates timestamped versions automatically:

```bash
# See all versions
ls -la compiled_pipelines/

# Use specific version
cp compiled_pipelines/ml_training_20250722_143052.yaml my_production_pipeline.yaml

# Always use latest for development
# compiled_pipelines/ml_training_latest.yaml
```

### 3. CI/CD Integration

The advanced compiler supports multiple CI/CD patterns for automated pipeline deployment:

#### GitHub Actions Example

```yaml
# .github/workflows/kubeflow-pipeline.yml
name: Deploy Kubeflow Pipeline

on:
  push:
    branches: [main]
    paths: ['pipelines/**']
  pull_request:
    branches: [main]
    paths: ['pipelines/**']

env:
  KUBEFLOW_ENDPOINT: ${{ secrets.KUBEFLOW_ENDPOINT }}
  KUBEFLOW_USERNAME: ${{ secrets.KUBEFLOW_USERNAME }}
  KUBEFLOW_PASSWORD: ${{ secrets.KUBEFLOW_PASSWORD }}
  KUBEFLOW_USER_NAMESPACE: ${{ secrets.KUBEFLOW_USER_NAMESPACE }}

jobs:
  validate-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Validate pipeline syntax
      run: |
        python -c "from pipelines.ml_training_pipeline import ml_training_pipeline; print('✅ Pipeline syntax valid')"
    
    - name: Compile pipeline
      run: |
        python compile_and_run.py
    
    - name: Upload pipeline artifacts
      uses: actions/upload-artifact@v3
      with:
        name: compiled-pipeline
        path: compiled_pipelines/
        retention-days: 30
    
    - name: Deploy to staging (PR)
      if: github.event_name == 'pull_request'
      run: |
        echo "🧪 Pipeline compiled for staging environment"
        # Add staging deployment logic here
    
    - name: Deploy to production (main)
      if: github.ref == 'refs/heads/main'
      run: |
        echo "🚀 Pipeline compiled for production environment"
        # Add production deployment logic here
```

#### Jenkins Pipeline Example

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        KUBEFLOW_ENDPOINT = credentials('kubeflow-endpoint')
        KUBEFLOW_USERNAME = credentials('kubeflow-username')
        KUBEFLOW_PASSWORD = credentials('kubeflow-password')
        KUBEFLOW_USER_NAMESPACE = credentials('kubeflow-namespace')
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python3.11 -m venv kfp_env
                    source kfp_env/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Validate') {
            steps {
                sh '''
                    source kfp_env/bin/activate
                    python -c "from pipelines.ml_training_pipeline import ml_training_pipeline; print('✅ Pipeline syntax valid')"
                '''
            }
        }
        
        stage('Compile') {
            steps {
                sh '''
                    source kfp_env/bin/activate
                    python compile_and_run.py
                '''
            }
        }
        
        stage('Archive') {
            steps {
                archiveArtifacts artifacts: 'compiled_pipelines/*.yaml', fingerprint: true
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    source kfp_env/bin/activate
                    python kfp_client.py
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
```

#### GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - validate
  - compile
  - deploy

variables:
  KUBEFLOW_ENDPOINT: ${KUBEFLOW_ENDPOINT}
  KUBEFLOW_USERNAME: ${KUBEFLOW_USERNAME}
  KUBEFLOW_PASSWORD: ${KUBEFLOW_PASSWORD}
  KUBEFLOW_USER_NAMESPACE: ${KUBEFLOW_USER_NAMESPACE}

before_script:
  - python3.11 -m venv kfp_env
  - source kfp_env/bin/activate
  - pip install -r requirements.txt

validate_pipeline:
  stage: validate
  script:
    - python -c "from pipelines.ml_training_pipeline import ml_training_pipeline; print('✅ Pipeline syntax valid')"
  only:
    changes:
      - pipelines/**/*

compile_pipeline:
  stage: compile
  script:
    - python compile_and_run.py
  artifacts:
    paths:
      - compiled_pipelines/
    expire_in: 1 week
  only:
    changes:
      - pipelines/**/*

deploy_staging:
  stage: deploy
     script:
     - echo "🧪 Deploying to staging"
     - python kfp_client.py
  environment:
    name: staging
  only:
    - develop

deploy_production:
  stage: deploy
     script:
     - echo "🚀 Deploying to production"
     - python kfp_client.py
  environment:
    name: production
  only:
    - main
  when: manual
```

#### Docker-based CI/CD

```dockerfile
# Dockerfile.pipeline-ci
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY pipelines/ ./pipelines/
COPY *.py ./

# Set environment variables
ENV PYTHONPATH=/app

# Default command for CI
CMD ["python", "compile_and_run.py"]
```

```bash
# CI script using Docker
#!/bin/bash
set -e

echo "🐳 Building pipeline CI container..."
docker build -f Dockerfile.pipeline-ci -t pipeline-ci:latest .

echo "🔨 Compiling pipeline in container..."
docker run --rm \
  -e KUBEFLOW_ENDPOINT="$KUBEFLOW_ENDPOINT" \
  -e KUBEFLOW_USERNAME="$KUBEFLOW_USERNAME" \
  -e KUBEFLOW_PASSWORD="$KUBEFLOW_PASSWORD" \
  -e KUBEFLOW_USER_NAMESPACE="$KUBEFLOW_USER_NAMESPACE" \
  -v $(pwd)/compiled_pipelines:/app/compiled_pipelines \
  pipeline-ci:latest

echo "✅ Pipeline compiled successfully!"
```

#### Kubernetes Job for CI/CD

```yaml
# k8s-pipeline-ci-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pipeline-compiler-job
spec:
  template:
    spec:
      containers:
      - name: pipeline-compiler
        image: pipeline-ci:latest
        env:
        - name: KUBEFLOW_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: kubeflow-credentials
              key: endpoint
        - name: KUBEFLOW_USERNAME
          valueFrom:
            secretKeyRef:
              name: kubeflow-credentials
              key: username
        - name: KUBEFLOW_PASSWORD
          valueFrom:
            secretKeyRef:
              name: kubeflow-credentials
              key: password
        - name: KUBEFLOW_USER_NAMESPACE
          valueFrom:
            secretKeyRef:
              name: kubeflow-credentials
              key: namespace
        volumeMounts:
        - name: pipeline-output
          mountPath: /app/compiled_pipelines
      volumes:
      - name: pipeline-output
        persistentVolumeClaim:
          claimName: pipeline-artifacts-pvc
      restartPolicy: Never
```

#### Environment-Specific Deployments

```bash
# deploy-pipeline.sh - Environment-aware deployment script
#!/bin/bash

ENVIRONMENT=${1:-staging}
PIPELINE_VERSION=${2:-latest}

echo "🚀 Deploying pipeline to $ENVIRONMENT environment"

# Set environment-specific variables
case $ENVIRONMENT in
  staging)
    export KUBEFLOW_ENDPOINT="https://staging-kubeflow.company.com"
    export KUBEFLOW_USER_NAMESPACE="kubeflow-user-staging"
    ;;
  production)
    export KUBEFLOW_ENDPOINT="https://kubeflow.company.com"
    export KUBEFLOW_USER_NAMESPACE="kubeflow-user-prod"
    ;;
  *)
    echo "❌ Unknown environment: $ENVIRONMENT"
    exit 1
    ;;
esac

# Compile pipeline
python compile_and_run.py

# Deploy using specific version
if [ "$PIPELINE_VERSION" != "latest" ]; then
  PIPELINE_FILE="compiled_pipelines/ml_training_$PIPELINE_VERSION.yaml"
else
  PIPELINE_FILE="compiled_pipelines/ml_training_latest.yaml"
fi

echo "📤 Submitting pipeline: $PIPELINE_FILE"
python -c "
from kfp_client import RawKFPClient
client = RawKFPClient()
result = client.submit_pipeline_run(
    pipeline_file='$PIPELINE_FILE',
    experiment_name='$ENVIRONMENT-experiments',
    run_name='automated-deploy-$(date +%Y%m%d-%H%M%S)'
)
print(f'✅ Pipeline deployed: {result[\"url\"]}')
"
```

## 🔗 Related Tools

- **`kfp_client.py`**: Direct pipeline submission with working authentication
- **`setup_credentials.py`**: Secure credential management
- **`pipelines/ml_training_pipeline.py`**: Your actual pipeline code

## 🆘 Getting Help

1. **Check validation output** for specific issues
2. **Review generated YAML** for compilation problems
3. **Test credentials** with `setup_credentials.py test`
4. **Use manual UI submission** as reliable fallback
5. **Check Kubeflow logs** for runtime issues

## ✅ Success Indicators

- ✅ **Clean compilation** with no errors
- ✅ **Validation passed** (or acceptable warnings)
- ✅ **Files generated** in `compiled_pipelines/`
- ✅ **Metadata embedded** in YAML comments
- ✅ **Symlinks updated** to latest version

Your pipeline is ready when you see:
```
🎉 Compilation complete!
📊 Pipeline ready for execution in Kubeflow UI
``` 