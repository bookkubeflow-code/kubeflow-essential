# S3 Pipeline Components for Kubeflow Pipelines
from kfp import dsl
from kfp.dsl import Dataset, Output, Input
import os


@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=["boto3==1.28.0", "pandas==2.0.3"]
)
def load_training_data(
    s3_bucket: str,
    s3_key: str,
    output_dataset: Output[Dataset]
):
    """Load training data from S3 bucket."""
    import boto3
    import pandas as pd

    # No credentials needed - automatically uses pod's IAM role
    s3_client = boto3.client('s3')

    # Download file from S3
    local_path = '/tmp/training_data.csv'
    s3_client.download_file(s3_bucket, s3_key, local_path)

    # Load and validate
    df = pd.read_csv(local_path)
    print(f"Loaded {len(df)} rows from s3://{s3_bucket}/{s3_key}")

    # Save to output artifact
    df.to_csv(output_dataset.path, index=False)
    output_dataset.metadata['row_count'] = len(df)
    output_dataset.metadata['source'] = f"s3://{s3_bucket}/{s3_key}"


@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=["boto3==1.28.0", "scikit-learn==1.3.0", "joblib==1.3.2"]
)
def train_and_save_model(
    training_data: Input[Dataset],
    s3_bucket: str,
    model_s3_key: str
) -> str:
    """Train model and save to S3."""
    import boto3
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    # Load training data
    df = pd.read_csv(training_data.path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model locally first
    model_path = '/tmp/model.joblib'
    joblib.dump(model, model_path)

    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(model_path, s3_bucket, model_s3_key)

    model_uri = f"s3://{s3_bucket}/{model_s3_key}"
    print(f"Model saved to {model_uri}")
    return model_uri
