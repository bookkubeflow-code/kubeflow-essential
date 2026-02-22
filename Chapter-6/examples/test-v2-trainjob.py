# Test v2 installation with a simple TrainJob via Python SDK
from kubeflow.trainer import TrainerClient, CustomTrainer


def train():
    print('TrainJob test successful')


job_id = TrainerClient().train(
    trainer=CustomTrainer(
        func=train,
        num_nodes=1,
        resources_per_node={
            'cpu': '2',
            'memory': '4Gi',
        }
    ),
    runtime=TrainerClient().get_runtime('torch-distributed')
)
print(f'TrainJob created with ID: {job_id}')
