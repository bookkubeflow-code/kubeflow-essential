# TFJob Python SDK example for programmatic job submission
from kubeflow import training

client = training.TrainingClient()

# Submit TFJob
tfjob_name = "mnist-sdk-training"
client.create_job(
    name=tfjob_name,
    job_kind="TFJob",
    base_image="tensorflow/tensorflow:2.13.0-gpu",
    num_workers=3,
    command=["python", "/workspace/train_mnist_tf.py", "--batch-size=64", "--epochs=15"],
    resources_per_replica={
        "requests": {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
        "limits": {"cpu": "8", "memory": "16Gi", "nvidia.com/gpu": "1"}
    }
)

# Wait for completion
client.wait_for_job(tfjob_name, job_kind="TFJob", timeout=3600)

if client.is_job_succeeded(tfjob_name, job_kind="TFJob"):
    print("Training completed successfully!")
