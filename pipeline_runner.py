# pipeline_runner.py

import kfp
from datetime import datetime
import json
import time
from typing import Dict, Optional, Any
import logging
import os
from pathlib import Path

# Import our custom authentication
try:
    from dex_auth import DexSessionManager
    from kfp_client import KFPClient
    HAS_CUSTOM_AUTH = True
except ImportError:
    HAS_CUSTOM_AUTH = False

class PipelineRunner:
    """Production-ready pipeline runner with monitoring and error handling."""
    
    def __init__(self, host: str = None, use_custom_auth: bool = True):
        self.host = host or os.getenv('KFP_ENDPOINT', 'http://localhost:8080')
        self.client = None
        self.use_custom_auth = use_custom_auth and HAS_CUSTOM_AUTH
        self.logger = logging.getLogger(__name__)
        
        # Set up logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        self._connect()
    
    def _connect(self, retries: int = 3):
        """Connect with retry logic because networks are unreliable."""
        for attempt in range(retries):
            try:
                if self.use_custom_auth:
                    # Use our custom authenticated client
                    self.logger.info("Using custom Dex authentication...")
                    auth_manager = DexSessionManager()
                    self.client = auth_manager.get_authenticated_client()
                    # Also keep a reference to our custom client for direct API calls
                    self.kfp_client = KFPClient()
                else:
                    # Use standard KFP client
                    self.client = kfp.Client(host=self.host)
                
                self.logger.info(f"✅ Connected to Kubeflow at {self.host}")
                return
            except Exception as e:
                if attempt == retries - 1:
                    self.logger.error(f"❌ Failed to connect after {retries} attempts: {e}")
                    raise
                self.logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def run_pipeline(
        self,
        pipeline_path: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        arguments: Dict[str, Any] = None,
        wait_for_completion: bool = False,
        timeout_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Run a pipeline with proper error handling and monitoring.
        
        Returns dict with run_id, status, and other metadata.
        """
        
        # Validate pipeline file exists
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        # Generate run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pipeline_name = Path(pipeline_path).stem
            run_name = f"{pipeline_name}_{timestamp}"
        
        # Ensure experiment exists
        try:
            experiment = self.client.create_experiment(name=experiment_name)
            self.logger.info(f"📁 Created new experiment: {experiment_name}")
        except Exception as e:
            # Experiment already exists or we don't have permissions
            try:
                experiment = self.client.get_experiment(experiment_name=experiment_name)
                self.logger.info(f"📁 Using existing experiment: {experiment_name}")
            except Exception as e2:
                self.logger.warning(f"⚠️ Could not get/create experiment '{experiment_name}': {e2}")
                # Try with Default experiment
                experiment_name = "Default"
                try:
                    experiment = self.client.get_experiment(experiment_name=experiment_name)
                    self.logger.info(f"📁 Falling back to Default experiment")
                except:
                    self.logger.error("❌ Could not access any experiment")
                    raise
        
        # Validate arguments
        arguments = arguments or {}
        self.logger.info(f"🚀 Submitting run '{run_name}' with arguments: {arguments}")
        
        # Submit the run
        try:
            run = self.client.create_run_from_pipeline_package(
                pipeline_file=pipeline_path,
                arguments=arguments,
                run_name=run_name,
                experiment_name=experiment_name,
                enable_caching=True
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to submit pipeline: {e}")
            
            # If using custom auth, try fallback method
            if self.use_custom_auth and hasattr(self, 'kfp_client'):
                self.logger.info("🔄 Trying fallback submission method...")
                try:
                    # Use our custom upload method
                    result = self.kfp_client.upload_pipeline(pipeline_path)
                    if result.get('success'):
                        self.logger.info("✅ Pipeline uploaded successfully via fallback method")
                        # Return partial result since we can't create the run directly
                        return {
                            'pipeline_id': result.get('pipeline_id'),
                            'pipeline_name': run_name,
                            'experiment_name': experiment_name,
                            'submitted_at': datetime.now().isoformat(),
                            'status': 'Uploaded - Manual run required',
                            'message': 'Pipeline uploaded successfully. Please create run manually in Kubeflow UI.'
                        }
                    else:
                        raise Exception(f"Fallback upload failed: {result.get('error')}")
                except Exception as fallback_error:
                    self.logger.error(f"❌ Fallback method also failed: {fallback_error}")
            
            raise
        
        run_id = run.run_id
        run_url = f"{self.host}/#/runs/details/{run_id}"
        
        self.logger.info(f"✅ Pipeline submitted successfully!")
        self.logger.info(f"📊 Run ID: {run_id}")
        self.logger.info(f"🔗 View at: {run_url}")
        
        result = {
            'run_id': run_id,
            'run_name': run_name,
            'experiment_name': experiment_name,
            'url': run_url,
            'submitted_at': datetime.now().isoformat(),
            'pipeline_path': pipeline_path,
            'arguments': arguments
        }
        
        # Wait for completion if requested
        if wait_for_completion:
            self.logger.info("⏳ Waiting for pipeline completion...")
            final_status = self._wait_for_run(run_id, timeout_seconds)
            result['status'] = final_status
            result['completed_at'] = datetime.now().isoformat()
            
            if final_status == 'Failed':
                self.logger.error("❌ Pipeline failed!")
                # Try to get error details
                try:
                    run_details = self.client.get_run(run_id)
                    if hasattr(run_details.run, 'error'):
                        result['error'] = run_details.run.error
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not fetch error details: {e}")
            elif final_status == 'Succeeded':
                self.logger.info("✅ Pipeline completed successfully!")
            elif final_status == 'Timeout':
                self.logger.warning(f"⏰ Pipeline timed out after {timeout_seconds}s")
        
        return result
    
    def _wait_for_run(self, run_id: str, timeout_seconds: int) -> str:
        """Wait for a run to complete with timeout."""
        start_time = time.time()
        last_status = None
        
        while True:
            try:
                run = self.client.get_run(run_id)
                status = run.run.status
                
                # Only log status changes to avoid spam
                if status != last_status:
                    elapsed = time.time() - start_time
                    self.logger.info(f"📊 Status: {status} (elapsed: {elapsed:.0f}s)")
                    last_status = status
                
                if status in ['Succeeded', 'Failed', 'Error', 'Skipped', 'Terminated']:
                    return status
                
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    self.logger.warning(f"⏰ Timeout waiting for run after {timeout_seconds}s")
                    return 'Timeout'
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error checking run status: {e}")
                time.sleep(30)
                
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    return 'Timeout'
    
    def list_runs(self, experiment_name: str = None, limit: int = 10) -> list:
        """List recent pipeline runs."""
        try:
            if experiment_name:
                experiment = self.client.get_experiment(experiment_name=experiment_name)
                runs = self.client.list_runs(experiment_id=experiment.experiment_id, page_size=limit)
            else:
                runs = self.client.list_runs(page_size=limit)
            
            run_list = []
            for run in runs.runs or []:
                run_info = {
                    'run_id': run.run_id,
                    'name': run.name,
                    'status': run.status,
                    'created_at': run.created_at.isoformat() if run.created_at else None,
                    'finished_at': run.finished_at.isoformat() if run.finished_at else None
                }
                run_list.append(run_info)
            
            return run_list
        except Exception as e:
            self.logger.error(f"❌ Failed to list runs: {e}")
            return []
    
    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific run."""
        try:
            run = self.client.get_run(run_id)
            return {
                'run_id': run_id,
                'name': run.run.name,
                'status': run.run.status,
                'created_at': run.run.created_at.isoformat() if run.run.created_at else None,
                'finished_at': run.run.finished_at.isoformat() if run.run.finished_at else None,
                'error': getattr(run.run, 'error', None)
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get run status: {e}")
            return {'error': str(e)}

# Utility function for quick pipeline submission
def quick_run(pipeline_path: str, experiment_name: str = "Default", **kwargs) -> Dict[str, Any]:
    """Quick utility function to run a pipeline with minimal setup."""
    runner = PipelineRunner()
    return runner.run_pipeline(
        pipeline_path=pipeline_path,
        experiment_name=experiment_name,
        **kwargs
    )

# Usage example that actually handles real-world scenarios
if __name__ == "__main__":
    # Create runs directory if it doesn't exist
    os.makedirs('runs', exist_ok=True)
    
    runner = PipelineRunner()
    
    # Example: Run our ML training pipeline
    try:
        result = runner.run_pipeline(
            pipeline_path='compiled_pipelines/ml_training_latest.yaml',
            experiment_name='ML Training',
            arguments={
                'dataset_url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
            },
            wait_for_completion=True,
            timeout_seconds=1800  # 30 minutes
        )
        
        # Save run metadata for tracking
        run_file = f"runs/{result.get('run_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(run_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"📁 Run metadata saved to: {run_file}")
        print(f"🎯 Final result: {result}")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        
    # Example: List recent runs
    print("\n📊 Recent runs:")
    recent_runs = runner.list_runs(limit=5)
    for run in recent_runs:
        print(f"  • {run['name']} - {run['status']} ({run['created_at']})") 