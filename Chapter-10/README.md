# Chapter 10: Production Operations & Integrations

This chapter covers the essential integrations and operational patterns for running Kubeflow in production: storage, CI/CD, monitoring, multi-tenancy, and cost management.

## Directory Structure

```
Chapter-10/
├── storage-integration/  # S3, GCS, Azure storage + database integration
├── cicd/                 # GitHub Actions CI/CD pipeline
├── monitoring/           # Prometheus, Grafana, alerting
├── multi-tenancy/        # Profiles, quotas, isolation
└── cost-management/      # Cost tracking and cleanup automation
```

## Storage Integration

### AWS S3 with IRSA (IAM Roles for Service Accounts)

1. Create IAM role with trust policy and permissions:
   - `iam-trust-policy.json` — Allows EKS pods to assume the role
   - `iam-permissions-policy.json` — Grants S3 bucket access

2. Create the Kubernetes ServiceAccount:
   ```bash
   kubectl apply -f s3-service-account.yaml
   ```

3. Use in pipelines — no credentials in code:
   - `s3_pipeline_components.py` — Load data from / save models to S3
   - `s3_ml_pipeline.py` — Complete pipeline with compilation and submission

### Database Integration

- `database_integration.py` — PostgreSQL feature fetching component
- `postgres-credentials.yaml` — Kubernetes Secret for DB credentials

### Other Cloud Providers

- `gcs-service-account.yaml` — GCP Workload Identity
- `azure-service-account.yaml` — Azure Managed Identity

## CI/CD with GitHub Actions

Automated pipeline compilation, testing, and deployment.

### Repository Structure

```
ml-pipelines/
├── .github/workflows/deploy-pipeline.yaml
├── pipelines/
├── components/
├── tests/
├── config/
├── scripts/
└── requirements.txt
```

### Key Files

| File | Purpose |
|------|---------|
| `deploy-pipeline-workflow.yaml` | GitHub Actions workflow (compile, test, deploy) |
| `scripts/compile_pipeline.py` | Compile Python pipeline definitions to YAML |
| `scripts/deploy_pipeline.py` | Deploy compiled pipelines to Kubeflow |
| `config/prod.yaml` | Production environment configuration |
| `tests/test_components.py` | Unit tests for pipeline components |
| `tests/test_pipelines.py` | Compilation and structure tests |

### Workflow

1. Push to feature branch — CI compiles and tests
2. Open PR — CI checks pass before merge
3. Merge to main — CD deploys to Kubeflow automatically

## Monitoring with Prometheus & Grafana

### Install

```bash
./monitoring/install-prometheus.sh
```

### Access Grafana

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Open http://localhost:3000 (admin / password from values file)
```

### Configuration Files

| File | Purpose |
|------|---------|
| `prometheus-values.yaml` | Helm values for kube-prometheus-stack |
| `alert-rules.yaml` | Kubeflow-specific alert rules |
| `alertmanager-slack-config.yaml` | Slack notification routing |

### Key Alerts

- **HighPipelinePodFailureRate** — Pods restarting frequently (warning)
- **GPUNodeDown** — GPU node unavailable for 5+ minutes (critical)
- **PersistentVolumeFillingUp** — Storage over 85% capacity (warning)

## Multi-Tenancy

### Team Onboarding (Automated)

```bash
./multi-tenancy/onboard-team.sh team-ml-fraud jane@example.com 64 256Gi 8
```

This creates a Profile, namespace, shared storage, and S3 ServiceAccount.

### Configuration Files

| File | Purpose |
|------|---------|
| `team-profile.yaml` | Profile with resource quotas and storage config |
| `resource-quotas-example.yaml` | Quota examples for different team types |
| `onboard-team.sh` | Automated onboarding script |
| `network-policy.yaml` | Deny cross-namespace traffic |
| `pod-security.yaml` | Pod security admission labels |

### Resource Quota Guidelines

- Overcommit CPU and memory by up to 1.5x
- Never overcommit GPUs
- Set pod and PVC count limits to prevent runaway workloads

## Cost Management

### Cost Tracking

`cost-tracking-rules.yaml` — Prometheus recording rules that calculate approximate hourly costs per namespace (GPU, CPU, storage).

### Automated Cleanup

| File | Schedule | Purpose |
|------|----------|---------|
| `cleanup-artifacts-cronjob.yaml` | Sundays 2 AM | Delete S3 artifacts older than 90 days from failed runs |
| `cleanup-pvcs-cronjob.yaml` | Sundays 3 AM | Delete PVCs not mounted for 30+ days |

### GPU Monitoring

Install the NVIDIA GPU exporter:
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-monitoring-tools/master/exporters/prometheus-dcgm/dcgm-exporter.yaml
```

Key Prometheus queries:
- GPU allocation: `sum(kube_pod_info{node=~".*gpu.*"}) by (namespace)`
- GPU utilization: `avg(DCGM_FI_DEV_GPU_UTIL{namespace!=""}) by (namespace) / 100`
