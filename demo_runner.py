# demo_runner.py
# Demo script to show PipelineRunner capabilities with manual pipeline execution

from pipeline_runner import PipelineRunner, quick_run
import time
import json
from datetime import datetime

def demo_pipeline_monitoring():
    """
    Demo script that shows how PipelineRunner would work.
    Since you already have the pipeline uploaded in Kubeflow UI, this demonstrates
    what the monitoring and tracking would look like.
    """
    
    print("🚀 PipelineRunner Demo")
    print("=" * 50)
    
    # Simulate what would happen when running a pipeline
    print("📊 This is what PipelineRunner would do if authentication was set up:")
    print()
    
    # Example 1: Basic pipeline run
    print("1️⃣ Basic Pipeline Submission:")
    print("   runner = PipelineRunner()")
    print("   result = runner.run_pipeline(")
    print("       pipeline_path='compiled_pipelines/ml_training_latest.yaml',")
    print("       experiment_name='ML Training',")
    print("       arguments={'dataset_url': 'https://...'},")
    print("       wait_for_completion=True")
    print("   )")
    print()
    
    # Simulate monitoring output
    print("2️⃣ Real-time Monitoring Output:")
    print("   ✅ Connected to Kubeflow at http://localhost:8080")
    print("   📁 Using existing experiment: ML Training")
    print("   🚀 Submitting run 'ml_training_latest_20250726_162500'")
    print("   ✅ Pipeline submitted successfully!")
    print("   📊 Run ID: abc123-def456-ghi789")
    print("   🔗 View at: http://localhost:8080/#/runs/details/abc123-def456-ghi789")
    print("   ⏳ Waiting for pipeline completion...")
    
    # Simulate status updates
    statuses = ["Running", "Running", "Running", "Succeeded"]
    for i, status in enumerate(statuses):
        elapsed = (i + 1) * 30
        print(f"   📊 Status: {status} (elapsed: {elapsed}s)")
        time.sleep(0.5)  # Just for demo effect
    
    print("   ✅ Pipeline completed successfully!")
    print()
    
    # Example result
    result = {
        'run_id': 'abc123-def456-ghi789',
        'run_name': 'ml_training_latest_20250726_162500',
        'experiment_name': 'ML Training',
        'url': 'http://localhost:8080/#/runs/details/abc123-def456-ghi789',
        'submitted_at': datetime.now().isoformat(),
        'pipeline_path': 'compiled_pipelines/ml_training_latest.yaml',
        'arguments': {'dataset_url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'},
        'status': 'Succeeded',
        'completed_at': datetime.now().isoformat()
    }
    
    print("3️⃣ Saved Metadata:")
    print(json.dumps(result, indent=2))
    print()
    
    # Additional features
    print("4️⃣ Additional PipelineRunner Features:")
    print("   📋 List recent runs: runner.list_runs(limit=10)")
    print("   🔍 Get run status: runner.get_run_status('run-id')")
    print("   ⚡ Quick run utility: quick_run('pipeline.yaml')")
    print("   🔄 Automatic retries with exponential backoff")
    print("   ⏰ Timeout handling for long-running pipelines")
    print("   📝 Comprehensive error reporting and logging")
    print()
    
    print("💡 To use PipelineRunner with your uploaded pipeline:")
    print("   1. Set up credentials: python setup_credentials.py")
    print("   2. Source credentials: source kubeflow.env")
    print("   3. Run: python pipeline_runner.py")
    print()
    print("   Or create a new run directly in Kubeflow UI and use PipelineRunner")
    print("   to monitor existing runs!")

def show_current_setup():
    """Show what files are available for pipeline execution."""
    import os
    
    print("📁 Current Pipeline Files:")
    print("-" * 30)
    
    if os.path.exists('compiled_pipelines'):
        files = os.listdir('compiled_pipelines')
        for file in sorted(files):
            if file.endswith('.yaml'):
                print(f"   ✅ {file}")
    else:
        print("   ❌ No compiled_pipelines directory found")
    
    print()
    print("🔧 Available Tools:")
    print("   ✅ pipeline_runner.py - Production runner with monitoring")
    print("   ✅ kfp_client.py - Direct API client")
    print("   ✅ compile_and_run.py - Advanced compiler")
    print("   ✅ dex_auth.py - Authentication helper")
    print()
    
    print("🎯 Since you already have 'ML Training Pipeline-1' uploaded:")
    print("   1. Go to Kubeflow UI: http://localhost:8080")
    print("   2. Navigate to Runs → Create run")
    print("   3. Select your 'ML Training Pipeline-1'")
    print("   4. Click 'Start' and watch it run!")
    print()
    print("   PipelineRunner would automate this entire process! 🚀")

if __name__ == "__main__":
    print("🎉 Welcome to PipelineRunner Demo!")
    print()
    
    show_current_setup()
    print()
    demo_pipeline_monitoring()
    
    print("✨ This demonstrates what PipelineRunner brings to your Kubeflow setup!")
    print("🔥 Ready to make your pipeline execution production-ready!") 