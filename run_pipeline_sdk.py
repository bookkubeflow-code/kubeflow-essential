#!/usr/bin/env python3
"""
Run the pipeline using KFP SDK instead of UI to bypass MySQL encoding issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.advanced_parameter_pipeline import advanced_parameter_pipeline
from config import config

def run_pipeline_via_sdk():
    """Run pipeline using KFP SDK instead of UI"""
    
    print("🚀 Running pipeline via KFP SDK to bypass UI encoding issues...")
    
    try:
        # Get KFP client
        client = config.get_client()
        
        # Create experiment
        experiment_name = "simple-ml-experiment"
        try:
            experiment = client.create_experiment(name=experiment_name)
            print(f"✅ Created experiment: {experiment_name}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"📋 Using existing experiment: {experiment_name}")
                experiment = client.get_experiment(experiment_name=experiment_name)
            else:
                raise e
        
        # Run pipeline directly
        run_result = client.create_run_from_pipeline_func(
            pipeline_func=advanced_parameter_pipeline,
            arguments={
                "dataset_path": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
                "model_output_path": "./outputs/models/"
            },
            experiment_name=experiment_name,
            run_name="simple-ml-run"
        )
        
        print(f"✅ Pipeline run created successfully!")
        print(f"📋 Run ID: {run_result.run_id}")
        print(f"🔗 Run URL: {run_result.run_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ SDK run failed: {e}")
        return False

if __name__ == "__main__":
    success = run_pipeline_via_sdk()
    if success:
        print("\n🎉 Pipeline submitted successfully via SDK!")
        print("Check the Kubeflow UI to monitor the run progress.")
    else:
        print("\n💡 The MySQL encoding issue affects both UI and SDK.")
        print("This confirms the problem is with your Kubeflow database configuration.")


