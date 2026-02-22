# Complete S3 ML Pipeline with compilation and submission
from kfp import dsl, compiler
import kfp
from s3_pipeline_components import load_training_data, train_and_save_model


@dsl.pipeline(
    name="S3 ML Pipeline",
    description="Complete pipeline with S3 integration"
)
def ml_pipeline_with_s3(
    data_bucket: str = "my-ml-pipeline-data",
    data_key: str = "datasets/training/data.csv",
    model_bucket: str = "my-ml-models",
    model_key: str = "models/random-forest/model.joblib"
):
    """End-to-end pipeline with S3 storage."""

    # Load data from S3
    load_task = load_training_data(
        s3_bucket=data_bucket,
        s3_key=data_key
    )

    # Train and save model to S3
    train_task = train_and_save_model(
        training_data=load_task.outputs['output_dataset'],
        s3_bucket=model_bucket,
        model_s3_key=model_key
    )

    # Configure pipeline to use our ServiceAccount
    load_task.set_env_variable('AWS_REGION', 'us-west-2')
    train_task.set_env_variable('AWS_REGION', 'us-west-2')


# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=ml_pipeline_with_s3,
    package_path='s3_pipeline.yaml'
)

# Submit with ServiceAccount configuration
client = kfp.Client(host='http://localhost:8080')

# Create run with ServiceAccount
run = client.create_run_from_pipeline_package(
    pipeline_file='s3_pipeline.yaml',
    arguments={
        'data_bucket': 'my-ml-pipeline-data',
        'data_key': 'datasets/fraud-detection/training.csv',
        'model_bucket': 'my-ml-models',
        'model_key': 'fraud-detection/v1/model.joblib'
    },
    service_account='s3-pipeline-sa'  # This is the critical line
)
