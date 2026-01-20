import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfp import dsl
from kfp.compiler import Compiler
from pipelines.components.ml_ops_components import load_data, train_model, evaluate_model

@dsl.pipeline(
    name="ml-training-pipeline",
    description="A machine learning pipeline that trains and stores a model on the Iris dataset"
)
def ml_training_pipeline(
    dataset_url: str = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
):
    # Load data
    load_data_task = load_data(dataset_url=dataset_url)
    
    # Train model using the loaded dataset
    train_model_task = train_model(input_dataset=load_data_task.outputs['output_dataset'])
    
    # Evaluate model with rich visualizations
    evaluate_model_task = evaluate_model(
        test_dataset=load_data_task.outputs['output_dataset'],
        model_input=train_model_task.outputs['model_output']
    )
    
    # Set dependencies (explicit ordering)
    train_model_task.after(load_data_task)
    evaluate_model_task.after(train_model_task)

if __name__ == '__main__':
    # Compile the pipeline
    Compiler().compile(
        pipeline_func=ml_training_pipeline,
        package_path='ml_pipeline.yaml'
    )
    print("Pipeline compiled successfully to ml_pipeline.yaml") 