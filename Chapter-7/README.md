# Katib Hyperparameter Tuning Example

A comprehensive example project demonstrating hyperparameter tuning with Katib on Kubeflow. This project accompanies Chapter 7 of the Kubeflow book and covers all aspects of defining and running experiments, from basic configurations to advanced optimization strategies.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [The Training Script](#the-training-script)
6. [Building the Training Container](#building-the-training-container)
7. [Experiment Configuration Deep Dive](#experiment-configuration-deep-dive)
8. [Running Experiments](#running-experiments)
9. [Monitoring and Logging Trials](#monitoring-and-logging-trials)
10. [Algorithm Examples](#algorithm-examples)
11. [Metrics Collector Configuration](#metrics-collector-configuration)
12. [Troubleshooting](#troubleshooting)
13. [Next Steps](#next-steps)

---

## Project Overview

This project demonstrates hyperparameter tuning for a **Random Forest / Gradient Boosting Classifier** on the **Breast Cancer Wisconsin dataset** using Katib. We chose this example because:

- **No GPU required**: Runs on any Kubernetes cluster
- **Fast iteration**: Each trial completes in seconds
- **Multiple hyperparameters**: Demonstrates integer, float, and categorical parameter types
- **Clear metrics**: Accuracy and F1-score are easy to understand and optimize
- **Framework agnostic**: Patterns apply to any ML framework (TensorFlow, PyTorch, XGBoost, etc.)

### What You'll Learn

- How to structure a training script for Katib integration
- How to define search spaces with different parameter types
- How to configure various optimization algorithms
- How to set up metrics collectors (StdOut, File)
- How to implement early stopping for resource efficiency
- How to monitor experiments via CLI and UI
- How to extract and use optimal parameters

---

## Prerequisites

### Kubernetes Cluster with Kubeflow

Ensure you have a running Kubeflow installation with Katib components:

```bash
# Check Kubeflow is installed
kubectl get pods -n kubeflow

# Check Katib components are running
kubectl get pods -n kubeflow | grep katib
```

Expected Katib components:
- `katib-controller` - Orchestrates experiments and trials
- `katib-db-manager` - Manages experiment data persistence
- `katib-mysql` (or postgres) - Database backend
- `katib-ui` - Web interface for experiment management

### Local Development Tools

```bash
# Docker for building images
docker --version

# kubectl for cluster interaction
kubectl version --client

# jq for JSON parsing (used by monitoring scripts)
jq --version

# Optional: Python for local testing
python --version
```

---

## Project Structure

```
katib-example/
├── README.md                              # This documentation
├── requirements.txt                       # Python dependencies
├── Dockerfile                             # Training container definition
│
├── src/
│   └── train.py                           # Main training script
│
├── experiments/
│   ├── 01-basic-random-search.yaml        # Basic experiment with random search
│   ├── 02-grid-search.yaml                # Exhaustive grid search
│   ├── 03-bayesian-optimization.yaml      # Model-based Bayesian optimization
│   ├── 04-hyperband.yaml                  # Hyperband with early stopping
│   ├── 05-tpe.yaml                        # Tree Parzen Estimator
│   ├── 06-custom-metrics-collector.yaml   # Custom metrics collection patterns
│   └── 07-early-stopping-only.yaml        # Early stopping with any algorithm
│
├── scripts/
│   ├── build-and-push.sh                  # Build and push Docker image
│   ├── submit-experiment.sh               # Submit experiment to cluster
│   ├── monitor-experiment.sh              # Real-time experiment monitoring
│   ├── get-optimal-params.sh              # Extract best parameters
│   └── cleanup-experiments.sh             # Clean up completed experiments
│
└── manifests/
    └── namespace.yaml                     # Namespace and RBAC configuration
```

---

## Quick Start

### 1. Build and Push the Training Container

```bash
# Clone the project
cd katib-example

# Build and push to your registry
./scripts/build-and-push.sh docker.io/your-username v1.0
```

### 2. Update Experiment YAMLs

Replace the placeholder image in experiment files:

```bash
# Update all experiment files with your image
sed -i 's|your-username/katib-sklearn-example:v1.0|YOUR_REGISTRY/katib-sklearn-example:v1.0|g' experiments/*.yaml
```

### 3. Submit an Experiment

```bash
# Submit the basic random search experiment
./scripts/submit-experiment.sh experiments/01-basic-random-search.yaml
```

### 4. Monitor Progress

```bash
# Watch experiment in real-time
./scripts/monitor-experiment.sh sklearn-random-search
```

### 5. Get Optimal Parameters

```bash
# Extract best parameters as JSON
./scripts/get-optimal-params.sh sklearn-random-search
```

---

## The Training Script

The training script (`src/train.py`) is the heart of the hyperparameter tuning process. It demonstrates all the patterns required for Katib integration.

### Key Design Principles

#### 1. Command-Line Arguments for Hyperparameters

Katib injects hyperparameter values through command-line arguments. Your script must accept them:

```python
parser.add_argument('--n-estimators', type=int, default=100,
                    help='Number of trees in the forest (50-300)')
parser.add_argument('--max-depth', type=int, default=10,
                    help='Maximum depth of each tree (3-20)')
parser.add_argument('--learning-rate', type=float, default=0.1,
                    help='Learning rate for Gradient Boosting (0.01-0.3)')
parser.add_argument('--criterion', type=str, default='gini',
                    choices=['gini', 'entropy', 'log_loss'],
                    help='Function to measure split quality')
```

#### 2. Structured Metric Logging

The default `StdOut` metrics collector looks for `metric_name=value` patterns:

```python
def log_metric(name: str, value: float):
    """Log a metric in Katib-compatible format."""
    print(f"{name}={value:.6f}")
    sys.stdout.flush()  # Ensure immediate visibility

# Usage
log_metric("accuracy", 0.9561)
log_metric("f1_score", 0.9489)
```

**CRITICAL**: The metric name must exactly match `objectiveMetricName` in your Experiment YAML.

#### 3. Intermediate Metrics for Early Stopping

Logging metrics at intervals enables early stopping algorithms:

```python
def log_intermediate_metric(epoch: int, name: str, value: float):
    """Log intermediate metrics for early stopping algorithms."""
    print(f"epoch={epoch} {name}={value:.6f}")
    sys.stdout.flush()

# During training loop
for epoch in range(num_epochs):
    # Training logic...
    log_intermediate_metric(epoch, "validation-accuracy", val_acc)
```

This format enables Hyperband and median stopping to identify underperforming trials early.

#### 4. Error Handling

Ensure Katib knows when trials fail:

```python
try:
    # Training logic
    log_metric("accuracy", metrics['accuracy'])
except Exception as e:
    print(f"ERROR: Training failed: {e}")
    log_metric("accuracy", 0.0)  # Log failure metric
    sys.exit(1)
```

### Testing Locally

Always test your training script locally before submitting to Katib:

```bash
# Test with default parameters
python src/train.py

# Test with specific hyperparameters
python src/train.py \
    --n-estimators 100 \
    --max-depth 10 \
    --min-samples-split 2 \
    --criterion gini \
    --model-type random_forest

# Verify output includes metrics in correct format:
# accuracy=0.9561
# f1_score=0.9489
```

---

## Building the Training Container

### Dockerfile Best Practices

Our Dockerfile follows container best practices for Katib:

```dockerfile
FROM python:3.11-slim

# CRITICAL: Ensure Python output is unbuffered for metrics collection
ENV PYTHONUNBUFFERED=1

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY src/train.py .

# Run as non-root user (Kubernetes security best practice)
RUN useradd -m trainer
USER trainer

# Entry point - Katib passes hyperparameters as arguments
ENTRYPOINT ["python", "train.py"]
```

**Key Point**: `PYTHONUNBUFFERED=1` is essential! Without it, Python buffers stdout and Katib's metrics collector may miss metrics.

### Build Commands

```bash
# Navigate to project root
cd katib-example

# Build the image
docker build -t your-registry/katib-sklearn-example:v1.0 .

# Test locally
docker run your-registry/katib-sklearn-example:v1.0 \
    --n-estimators 100 \
    --max-depth 10 \
    --criterion gini

# Push to registry
docker push your-registry/katib-sklearn-example:v1.0
```

---

## Experiment Configuration Deep Dive

Every Katib Experiment YAML has five key sections. Understanding each is essential for effective hyperparameter tuning.

### Section 1: Objective

```yaml
objective:
  type: maximize           # or "minimize" for loss metrics
  goal: 0.98               # Stop when this value is reached (optional)
  objectiveMetricName: accuracy
  additionalMetricNames:
    - f1_score
    - precision
    - recall
```

| Field | Description |
|-------|-------------|
| `type` | `maximize` for accuracy, F1, AUC; `minimize` for loss, error |
| `goal` | Early termination target - stops experiment when reached |
| `objectiveMetricName` | Primary metric for optimization - must match training script |
| `additionalMetricNames` | Secondary metrics to track (not optimized) |

### Section 2: Algorithm

```yaml
algorithm:
  algorithmName: bayesianoptimization
  algorithmSettings:
    - name: "random_state"
      value: "42"
    - name: "acq_func"
      value: "gp_hedge"
```

Available algorithms:
- `random` - Random search (baseline)
- `grid` - Exhaustive grid search
- `bayesianoptimization` - Gaussian Process-based optimization
- `tpe` - Tree Parzen Estimator
- `hyperband` - Resource-efficient successive halving
- `cmaes` - Covariance Matrix Adaptation Evolution Strategy
- `sobol` - Quasirandom sequence

### Section 3: Trial Limits and Parallelism

```yaml
maxTrialCount: 30          # Total trials to run
parallelTrialCount: 3      # Concurrent trials
maxFailedTrialCount: 3     # Failure threshold
```

| Field | Description |
|-------|-------------|
| `parallelTrialCount` | Set based on cluster capacity. Higher = faster but more resources |
| `maxTrialCount` | Budget control - stop after N trials |
| `maxFailedTrialCount` | Fault tolerance - stop if too many trials fail |

### Section 4: Search Space (Parameters)

#### Integer Parameters

```yaml
- name: n-estimators
  parameterType: int
  feasibleSpace:
    min: "50"
    max: "300"
    step: "50"              # Required for grid search only
```

#### Float/Double Parameters

```yaml
- name: learning-rate
  parameterType: double
  feasibleSpace:
    min: "0.001"
    max: "0.1"
```

#### Categorical Parameters

```yaml
- name: criterion
  parameterType: categorical
  feasibleSpace:
    list:
      - "gini"
      - "entropy"
      - "log_loss"
```

### Section 5: Trial Template

```yaml
trialTemplate:
  primaryContainerName: training-container
  trialParameters:
    - name: nEstimators        # camelCase for template
      reference: n-estimators  # matches parameter name
  trialSpec:
    apiVersion: batch/v1
    kind: Job
    spec:
      template:
        spec:
          containers:
            - name: training-container
              image: your-registry/katib-sklearn-example:v1.0
              command:
                - "python"
                - "/app/train.py"
                - "--n-estimators=${trialParameters.nEstimators}"
          restartPolicy: Never
```

The `${trialParameters.X}` syntax injects suggested hyperparameter values into your training command.

---

## Running Experiments

### Via kubectl (CLI)

#### Submit an Experiment

```bash
# Apply the experiment YAML
kubectl apply -f experiments/01-basic-random-search.yaml

# Check experiment status
kubectl get experiment sklearn-random-search -n kubeflow

# Watch experiment progress
kubectl get experiment sklearn-random-search -n kubeflow -w
```

#### Monitor Trials

```bash
# List all trials for an experiment
kubectl get trials -n kubeflow \
    -l katib.kubeflow.org/experiment=sklearn-random-search

# Get detailed trial information
kubectl describe trial <trial-name> -n kubeflow

# View trial logs
kubectl logs -n kubeflow -l trial-name=<trial-name>
```

#### Get Best Parameters

```bash
# Full experiment details including optimal trial
kubectl describe experiment sklearn-random-search -n kubeflow

# Extract optimal parameters as JSON
kubectl get experiment sklearn-random-search -n kubeflow \
    -o jsonpath='{.status.currentOptimalTrial.parameterAssignments}'
```

### Via Katib UI

1. **Access the UI**:
   ```bash
   # Port-forward the Katib UI
   kubectl port-forward svc/katib-ui -n kubeflow 8080:80
   ```
   Open http://localhost:8080/katib/ in your browser.

2. **Create Experiment**:
   - Click "NEW EXPERIMENT"
   - Fill in the form or paste YAML directly
   - Click "DEPLOY"

3. **Monitor Progress**:
   - View real-time trial status
   - See metric convergence graphs
   - Parallel coordinates plot for parameter relationships

4. **Analyze Results**:
   - Compare trial performances
   - View optimal parameters
   - Export results

### Via Helper Scripts

```bash
# Submit with confirmation and status
./scripts/submit-experiment.sh experiments/01-basic-random-search.yaml

# Real-time monitoring dashboard
./scripts/monitor-experiment.sh sklearn-random-search

# Get optimal parameters in various formats
./scripts/get-optimal-params.sh sklearn-random-search kubeflow json
./scripts/get-optimal-params.sh sklearn-random-search kubeflow yaml
./scripts/get-optimal-params.sh sklearn-random-search kubeflow cli

# Clean up completed experiments
./scripts/cleanup-experiments.sh kubeflow --completed
```

---

## Monitoring and Logging Trials

### Real-Time Experiment Status

```bash
# Watch experiment status changes
watch -n 2 'kubectl get experiment sklearn-random-search -n kubeflow'

# Output example:
# NAME                    TYPE        STATUS      AGE
# sklearn-random-search   Running     Created     5m
```

### Trial Status Overview

```bash
# Get trial summary with sorting
kubectl get trials -n kubeflow \
    -l katib.kubeflow.org/experiment=sklearn-random-search \
    --sort-by=.metadata.creationTimestamp

# Output example:
# NAME                             STATUS      AGE
# sklearn-random-search-abc123     Succeeded   10m
# sklearn-random-search-def456     Succeeded   8m
# sklearn-random-search-ghi789     Running     2m
```

### Viewing Trial Logs

```bash
# Find the pod for a specific trial
kubectl get pods -n kubeflow -l trial-name=sklearn-random-search-abc123

# Stream logs from training container
kubectl logs -n kubeflow -l trial-name=sklearn-random-search-abc123 -f

# Example output:
# [2024-01-15 10:23:45] Loading Breast Cancer Wisconsin dataset...
# [2024-01-15 10:23:45] Dataset shape: (569, 30)
# [2024-01-15 10:23:45] Training with parameters:
# [2024-01-15 10:23:45]   n_estimators: 150
# [2024-01-15 10:23:45]   max_depth: 12
# epoch=1 validation-accuracy=0.912281
# epoch=2 validation-accuracy=0.929825
# ...
# accuracy=0.956140
# f1_score=0.948936
```

### Metrics Analysis

```bash
# Get metrics for all trials
kubectl get trials -n kubeflow \
    -l katib.kubeflow.org/experiment=sklearn-random-search \
    -o jsonpath='{range .items[*]}{.metadata.name}: accuracy={.status.observation.metrics[0].latest}{"\n"}{end}'

# Output:
# sklearn-random-search-abc123: accuracy=0.956140
# sklearn-random-search-def456: accuracy=0.938596
# sklearn-random-search-ghi789: accuracy=0.947368
```

### Using the Monitor Script

The `monitor-experiment.sh` script provides a comprehensive dashboard:

```bash
./scripts/monitor-experiment.sh sklearn-random-search
```

Shows:
- Experiment status and progress bar
- Current optimal trial with parameters and metrics
- Recent trial list with status
- Running pods and resource usage

---

## Algorithm Examples

### Random Search (`01-basic-random-search.yaml`)

**Best for**: Quick exploration, establishing baselines

```yaml
algorithm:
  algorithmName: random
```

Random search samples hyperparameters uniformly from the search space. It's simple, parallelizes perfectly, and often surprisingly effective.

**When to use**:
- Starting a new optimization problem
- Large search spaces with unknown structure
- Need maximum parallelism

### Grid Search (`02-grid-search.yaml`)

**Best for**: Small search spaces, exhaustive evaluation

```yaml
algorithm:
  algorithmName: grid
```

Requires `step` in parameter definitions. Evaluates every combination systematically.

**When to use**:
- Fewer than 100 total combinations
- Need guaranteed coverage
- Final refinement of narrow ranges

### Bayesian Optimization (`03-bayesian-optimization.yaml`)

**Best for**: Expensive evaluations, continuous parameters

```yaml
algorithm:
  algorithmName: bayesianoptimization
  algorithmSettings:
    - name: "random_state"
      value: "42"
    - name: "acq_func"
      value: "gp_hedge"
    - name: "n_initial_points"
      value: "5"
```

Uses Gaussian Process to model objective function and select promising configurations.

**When to use**:
- Trials are expensive (GPU-hours)
- Mostly continuous parameters
- Low-dimensional space (<15 parameters)

### Hyperband (`04-hyperband.yaml`)

**Best for**: Iterative training, resource efficiency

```yaml
algorithm:
  algorithmName: hyperband
  algorithmSettings:
    - name: "resource_name"
      value: "num-epochs"
    - name: "eta"
      value: "3"
    - name: "r_l"
      value: "1"
```

Requires a resource parameter (like epochs). Aggressively stops poor performers using successive halving.

**When to use**:
- Training is iterative with natural checkpoints
- Early performance predicts final performance
- Need to evaluate many configurations efficiently

### TPE (`05-tpe.yaml`)

**Best for**: Mixed parameter types, moderate dimensions

```yaml
algorithm:
  algorithmName: tpe
  algorithmSettings:
    - name: "random_state"
      value: "42"
    - name: "gamma"
      value: "0.25"
```

Better than Bayesian optimization for high-dimensional or categorical-heavy spaces.

**When to use**:
- Many categorical parameters
- 10-20+ dimensions
- Complex search spaces with mixed types

### CMA-ES (`07-early-stopping-only.yaml`)

**Best for**: Continuous optimization, moderate dimensions

```yaml
algorithm:
  algorithmName: cmaes
  algorithmSettings:
    - name: "random_state"
      value: "42"
```

Evolutionary algorithm that adapts search distribution based on successful configurations.

**When to use**:
- Purely continuous parameters
- Non-convex objective landscapes
- Medium-dimensional spaces (5-30 parameters)

---

## Metrics Collector Configuration

### Default StdOut Collector

The simplest approach - Katib parses stdout for metrics:

```yaml
metricsCollectorSpec:
  collector:
    kind: StdOut
```

Training script logs: `accuracy=0.95`

### Custom Regex Patterns

For non-standard log formats:

```yaml
metricsCollectorSpec:
  collector:
    kind: StdOut
  source:
    filter:
      metricsFormat:
        # Standard: metric_name=value
        - "{metricName: ([\\w|-]+), metricValue: ((-?\\d+)(\\.\\d+)?)}"
        # With epoch: epoch=5 metric_name=value
        - "epoch=(\\d+).+{metricName: ([\\w|-]+), metricValue: ((-?\\d+)(\\.\\d+)?)}"
```

### File Collector

For structured metric files:

```yaml
metricsCollectorSpec:
  collector:
    kind: File
  source:
    fileSystemPath:
      path: /var/log/katib/metrics.log
      kind: File
```

Training script writes:
```python
with open('/var/log/katib/metrics.log', 'a') as f:
    f.write(f"accuracy={accuracy}\n")
```

---

## Troubleshooting

### Common Issues

#### 1. Trials Stuck in Pending

```bash
kubectl describe trial <trial-name> -n kubeflow
# Check Events section for:
# - Insufficient resources
# - Image pull errors
# - Node scheduling issues
```

**Solutions**:
- Check cluster resources: `kubectl top nodes`
- Verify image exists and is accessible
- Check resource requests aren't too high

#### 2. Metrics Not Collected

**Symptoms**: Trial succeeds but shows no metrics

**Check**:
```bash
# View trial logs
kubectl logs -n kubeflow -l trial-name=<trial-name>

# Verify output includes:
# accuracy=0.95
# NOT: Accuracy: 0.95 (wrong format)
```

**Solutions**:
- Ensure metric format matches collector pattern
- Print to stdout, not stderr
- Metric name must match `objectiveMetricName` exactly
- Add `sys.stdout.flush()` after printing

#### 3. All Trials Failing

```bash
kubectl logs -n kubeflow -l trial-name=<trial-name>
# Look for Python errors, missing dependencies
```

**Solutions**:
- Test container locally first
- Check all hyperparameter combinations are valid
- Verify dependencies in requirements.txt

#### 4. Image Pull Errors

```bash
kubectl describe pod <pod-name> -n kubeflow
# Look for: ImagePullBackOff
```

**Solutions**:
- Verify image exists: `docker pull your-image`
- Add imagePullSecrets for private registries
- Check image name spelling in YAML

### Debug Mode

Run training locally to verify metrics output:

```bash
python src/train.py \
    --n-estimators 100 \
    --max-depth 10 \
    --criterion gini

# Verify output includes exactly:
# accuracy=X.XXXXXX
# f1_score=X.XXXXXX
```

Test with Docker:

```bash
docker run --rm your-registry/katib-sklearn-example:v1.0 \
    --n-estimators 100 \
    --max-depth 10 2>&1 | grep -E "^[a-z_]+=.*"
```

---

## Next Steps

### 1. Adapt for Your Model

Replace the Random Forest with your own algorithm:

```python
# In train.py
from your_package import YourModel

model = YourModel(
    learning_rate=args.learning_rate,
    hidden_size=args.hidden_size,
    ...
)
```

### 2. Expand Search Space

Add hyperparameters specific to your model:

```yaml
parameters:
  - name: hidden-size
    parameterType: int
    feasibleSpace:
      min: "64"
      max: "512"
  - name: attention-heads
    parameterType: categorical
    feasibleSpace:
      list: ["4", "8", "12", "16"]
```

### 3. Integrate with Pipelines

Use optimal parameters in downstream training:

```bash
# Get optimal params as CLI arguments
OPTIMAL_PARAMS=$(./scripts/get-optimal-params.sh sklearn-random-search kubeflow cli)

# Use in final training
python train_final.py $OPTIMAL_PARAMS --epochs 100
```

### 4. Scale Up

For production runs:

```yaml
maxTrialCount: 100
parallelTrialCount: 10  # Match cluster capacity
```

Consider cluster autoscaling for large experiments.

---

## References

- [Katib Official Documentation](https://www.kubeflow.org/docs/components/katib/)
- [Katib GitHub Repository](https://github.com/kubeflow/katib)
- [Hyperparameter Optimization Algorithms](https://www.kubeflow.org/docs/components/katib/overview/)
- [Bergstra & Bengio - Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)
- [Hyperband: A Novel Bandit-Based Approach](https://arxiv.org/abs/1603.06560)
