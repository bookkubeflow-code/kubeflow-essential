# DeepSpeed training example using Training Operator v2
# For training models with billions of parameters
from kubeflow.trainer import TrainerClient, CustomTrainer

client = TrainerClient()

job_name = client.train(
    name="large-model-deepspeed",
    trainer=CustomTrainer(
        image="deepspeed/deepspeed:latest",
        command=["deepspeed", "/workspace/train_large_model.py"],
        num_nodes=8,
        resources_per_node={
            "cpu": "16",
            "memory": "64Gi",
            "gpu": "4"
        }
    ),
    runtime=client.get_runtime("deepspeed-distributed")
)

print(f"DeepSpeed training job created: {job_name}")
