# Chapter 2: Getting Started with Kubeflow

This chapter covers installing and configuring Kubeflow on a local Kubernetes cluster, exploring the dashboard, and understanding each core component.

## Directory Structure

```
Chapter-2/
├── scripts/              # Installation and setup scripts
├── auth/                 # Authentication and user management
├── profiles/             # Kubeflow profile manifests
└── security/             # Production security scripts
```

## Prerequisites

- Kubernetes cluster (v1.22+) — Kind is used for local deployment
- At least 16GB memory and 4 CPU cores
- Docker installed and running
- kubectl command-line tool
- Kustomize v5.4.3+

## Setup Steps

### 1. Install Kustomize

```bash
./scripts/install-kustomize.sh
```

### 2. Create a Kind Cluster

```bash
./scripts/create-kind-cluster.sh
```

### 3. Save Kubeconfig

```bash
./scripts/save-kubeconfig.sh
```

### 4. Create Docker Registry Secret

```bash
./scripts/create-docker-secret.sh
```

### 5. Install Kubeflow

```bash
./scripts/install-kubeflow.sh
```

### 6. Verify Installation

```bash
./scripts/verify-installation.sh
```

### 7. Access the Dashboard

```bash
./scripts/port-forward-dashboard.sh
```

Open http://localhost:8080 with default credentials:
- Email: `user@example.com`
- Password: `12341234`

## User Management

### Adding Static Users

```bash
./auth/add-static-user.sh
```

### Generating Password Hashes

```bash
./auth/generate-password-hash.sh
```

### Enterprise Identity Integration

See `auth/dex-oidc-connector-example.yaml` for an Azure AD OIDC connector example. Dex supports LDAP, OIDC, SAML, GitHub, and GitLab connectors.

## Profiles

Create isolated user/team environments:

```bash
kubectl apply -f profiles/profile-example.yaml
```

## Production Security

Change default credentials before going to production:

```bash
./security/change-default-credentials.sh
```

## Key Components

| Component | Version | Purpose |
|-----------|---------|---------|
| Training Operator | v1.9.1 | Distributed training jobs |
| Notebook Controller | v1.10.0 | Notebook server management |
| Central Dashboard | v1.10.0 | Unified UI |
| Katib | v0.18.0 | Hyperparameter tuning |
| KServe | v0.14.1 | Model serving |
| Kubeflow Pipelines | 2.4.1 | ML workflow orchestration |
| Istio | 1.24.3 | Service mesh |
| Knative | v1.16.2 | Serverless workloads |
| Cert Manager | 1.16.1 | TLS certificates |
