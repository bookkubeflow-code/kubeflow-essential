# tests/test_pipelines.py
import yaml
from pathlib import Path


def test_fraud_detection_pipeline_compiles():
    """Test that fraud detection pipeline compiles successfully."""
    compiled_path = Path("compiled/fraud_detection_pipeline.yaml")

    assert compiled_path.exists(), "Pipeline should be compiled"

    with open(compiled_path) as f:
        pipeline_spec = yaml.safe_load(f)

    # Verify basic structure
    assert 'metadata' in pipeline_spec
    assert 'name' in pipeline_spec['metadata']
    assert pipeline_spec['metadata']['name'] == 'fraud-detection-pipeline'

    # Verify expected tasks exist
    tasks = pipeline_spec['spec']['templates']
    task_names = [t['name'] for t in tasks]

    expected_tasks = ['load-data', 'preprocess', 'train-model', 'evaluate']
    for task in expected_tasks:
        assert task in task_names, f"Expected task {task} not found in pipeline"
