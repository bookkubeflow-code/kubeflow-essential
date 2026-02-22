# submit_training_v2.py
from kubernetes import client, config
import time

# Load kubeconfig
config.load_kube_config()

# Define training parameters
training_params = {
    'batch_size': 128,
    'epochs': 20,
    'lr': 0.001,
    'checkpoint_dir': '/workspace/checkpoints'
}

# Create custom object API
api = client.CustomObjectsApi()

print("Submitting CIFAR-10 training job...")

# Define TrainJob
trainjob = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainJob",
    "metadata": {
        "generateName": "cifar10-training-",
        "namespace": "default"  # Using default namespace - no need to create
    },
    "spec": {
        "runtimeRef": {
            "kind": "ClusterTrainingRuntime",
            "name": "torch-distributed"
        },
        "trainer": {
            "numNodes": 3,

            # Replace with your registry image
            # Examples:
            # - Docker Hub: "username/cifar10-training:v1"
            # - AWS ECR: "account-id.dkr.ecr.region.amazonaws.com/cifar10-training:v1"
            # - GCR: "gcr.io/project-id/cifar10-training:v1"
            # - Azure ACR: "myregistry.azurecr.io/cifar10-training:v1"
            "image": "<your-registry>/cifar10-training:v1",

            "command": [
                "python",
                "/workspace/train_cifar10.py",
                f"--batch-size={training_params['batch_size']}",
                f"--epochs={training_params['epochs']}",
                f"--lr={training_params['lr']}",
                f"--checkpoint-dir={training_params['checkpoint_dir']}"
            ],

            # Resource specifications
            "resourcesPerNode": {
                "requests": {
                    "cpu": "2",
                    "memory": "4Gi"
                },
                "limits": {
                    "cpu": "4",
                    "memory": "8Gi"
                }
            },

            "env": [
                {"name": "NCCL_DEBUG", "value": "INFO"},
                {"name": "PYTHONUNBUFFERED", "value": "1"}
            ]
        }
    }
}

# Submit the TrainJob
try:
    response = api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace="default",  # Matches metadata.namespace
        plural="trainjobs",
        body=trainjob
    )

    job_id = response['metadata']['name']
    namespace = response['metadata']['namespace']

    print(f"Training job created: {job_id}")
    print(f"\nMonitor with:")
    print(f"  kubectl get trainjobs {job_id} -n {namespace} -w")
    print(f"  kubectl logs -n {namespace} -l trainer.kubeflow.org/trainjob-name={job_id} -f")

    # Optional: Monitor job status
    print(f"\nMonitoring job (Ctrl+C to stop monitoring, job continues)...\n")

    try:
        while True:
            # Get job status
            job = api.get_namespaced_custom_object(
                group="trainer.kubeflow.org",
                version="v1alpha1",
                namespace=namespace,
                plural="trainjobs",
                name=job_id
            )

            status = job.get('status', {}).get('phase', 'Unknown')
            print(f"Status: {status}")

            if status in ["Succeeded", "Failed", "Completed"]:
                print(f"\nJob finished with status: {status}")
                break

            time.sleep(10)

    except KeyboardInterrupt:
        print(f"\nStopped monitoring. Job continues running in cluster.")
        print(f"Check status: kubectl get trainjobs {job_id} -n {namespace}")

except Exception as e:
    print(f"Error creating TrainJob: {e}")
