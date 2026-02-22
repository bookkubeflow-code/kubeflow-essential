#!/usr/bin/env python3
"""
Deploy compiled pipeline to Kubeflow.
Usage: python deploy_pipeline.py <pipeline_yaml> <environment>
"""

import sys
import yaml
from pathlib import Path
import kfp
from typing import Optional


def load_config(environment: str) -> dict:
    """Load environment-specific configuration."""
    config_path = Path(f"config/{environment}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def deploy_pipeline(pipeline_path: str, environment: str):
    """Deploy pipeline to Kubeflow."""

    # Load configuration
    config = load_config(environment)

    # Connect to Kubeflow
    client = kfp.Client(
        host=config['kubeflow']['host'],
        namespace=config['kubeflow']['namespace']
    )

    # Extract pipeline name from the YAML
    with open(pipeline_path) as f:
        pipeline_spec = yaml.safe_load(f)
        pipeline_name = pipeline_spec['metadata']['name']

    print(f"Deploying pipeline: {pipeline_name}")
    print(f"Environment: {environment}")
    print(f"Kubeflow host: {config['kubeflow']['host']}")

    # Check if pipeline already exists
    try:
        existing_pipeline = client.get_pipeline(pipeline_name)
        print(f"Found existing pipeline (id: {existing_pipeline.id})")

        # Upload a new version
        version = client.upload_pipeline_version(
            pipeline_package_path=pipeline_path,
            pipeline_version_name=f"{pipeline_name}-{config.get('version', 'latest')}",
            pipeline_id=existing_pipeline.id
        )
        print(f"Created new version (id: {version.id})")

    except Exception:
        # Pipeline doesn't exist, create it
        print("Pipeline doesn't exist yet, creating new pipeline...")
        pipeline = client.upload_pipeline(
            pipeline_package_path=pipeline_path,
            pipeline_name=pipeline_name
        )
        print(f"Created new pipeline (id: {pipeline.id})")

    print(f"Deployment successful")

    # Optionally create a run if specified in config
    if config.get('auto_run', False):
        print("Auto-run enabled, creating pipeline run...")
        experiment_name = config.get('experiment_name', 'Default')
        experiment = client.create_experiment(experiment_name)

        run = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=f"{pipeline_name}-auto-run",
            pipeline_package_path=pipeline_path
        )
        print(f"Created run (id: {run.id})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: deploy_pipeline.py <pipeline_yaml> <environment>")
        sys.exit(1)

    try:
        deploy_pipeline(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)
