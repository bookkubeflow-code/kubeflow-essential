#!/usr/bin/env python3

import requests
import json
import yaml
import base64
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dex_auth import DexSessionManager

class RawKFPClient:
    """
    Raw HTTP-based KFP client that works with Dex authentication.
    
    This bypasses the official KFP client issues with cookie handling
    and uses direct HTTP requests to the KFP API.
    """
    
    def __init__(
        self,
        endpoint_url: str = None,
        username: str = None,
        password: str = None,
        namespace: str = None
    ):
        # Use environment variables with fallbacks
        endpoint_url = endpoint_url or os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8080')
        username = username or os.getenv('KUBEFLOW_USERNAME', 'user@example.com')
        password = password or os.getenv('KUBEFLOW_PASSWORD', '12341234')
        namespace = namespace or os.getenv('KUBEFLOW_USER_NAMESPACE', 'kubeflow-user-example-com')
        self.endpoint_url = endpoint_url
        self.namespace = namespace
        self.pipeline_base_url = f"{endpoint_url}/pipeline"
        self.api_base_url = f"{self.pipeline_base_url}/apis/v2beta1"
        
        # Initialize authentication
        self.auth_manager = DexSessionManager(
            endpoint_url=endpoint_url,
            dex_username=username,
            dex_password=password
        )
        
        self.session = requests.Session()
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Setup session with Dex authentication cookies"""
        print("🔐 Setting up authentication...")
        cookies = self.auth_manager.get_session_cookies()
        
        if cookies:
            for cookie_pair in cookies.split("; "):
                if "=" in cookie_pair:
                    name, value = cookie_pair.split("=", 1)
                    self.session.cookies.set(name, value)
            print(f"✅ Authentication configured with {len(self.session.cookies)} cookies")
        else:
            print("⚠️  No cookies obtained - proceeding without authentication")
    
    def health_check(self) -> Dict[str, Any]:
        """Check KFP API health"""
        url = f"{self.api_base_url}/healthz"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_pipelines(self) -> Dict[str, Any]:
        """List all pipelines"""
        url = f"{self.api_base_url}/pipelines"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments"""
        url = f"{self.api_base_url}/experiments?namespace={self.namespace}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_runs(self) -> Dict[str, Any]:
        """List all runs"""
        url = f"{self.api_base_url}/runs?namespace={self.namespace}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def create_experiment(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new experiment"""
        url = f"{self.api_base_url}/experiments?namespace={self.namespace}"
        
        payload = {
            "display_name": name,
            "description": description
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def upload_pipeline(self, pipeline_file: str, name: str = None) -> Dict[str, Any]:
        """Upload a pipeline from YAML file"""
        pipeline_path = Path(pipeline_file)
        
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
        
        # Read and encode the pipeline file
        with open(pipeline_path, 'rb') as f:
            pipeline_content = f.read()
        
        # Generate pipeline name if not provided
        if name is None:
            name = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📁 Pipeline file size: {len(pipeline_content)} bytes")
        
        # Try v2 API first, fallback to v1 if it fails
        upload_urls = [
            f"{self.api_base_url}/pipelines/upload",  # v2 API
            f"{self.pipeline_base_url}/apis/v1beta1/pipelines/upload"  # v1 API
        ]
        
        for i, url in enumerate(upload_urls):
            api_version = "v2" if i == 0 else "v1"
            print(f"🔄 Trying {api_version} upload API...")
            
            try:
                # Prepare multipart form data
                files = {
                    'uploadfile': (pipeline_path.name, pipeline_content, 'application/x-yaml')
                }
                
                data = {
                    'name': name,
                    'description': f'Pipeline uploaded via KFP Client at {datetime.now()}'
                }
                
                # Set longer timeout for upload
                response = self.session.post(url, files=files, data=data, timeout=120)
                response.raise_for_status()
                
                print(f"✅ Pipeline uploaded successfully via {api_version} API")
                return response.json()
                
            except requests.exceptions.Timeout:
                print(f"⚠️  {api_version} API upload timed out (120s), trying next method...")
                continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 504:
                    print(f"⚠️  {api_version} API gateway timeout, trying next method...")
                    continue
                else:
                    print(f"⚠️  {api_version} API failed with {e.response.status_code}, trying next method...")
                    continue
            except Exception as e:
                print(f"⚠️  {api_version} API failed: {e}, trying next method...")
                continue
        
        # If both APIs fail, suggest manual upload
        raise Exception(
            f"Pipeline upload failed via both v2 and v1 APIs. "
            f"Please try manual upload via UI: http://localhost:8080"
        )
    
    def create_run(
        self,
        pipeline_id: str,
        experiment_id: str,
        run_name: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new pipeline run"""
        url = f"{self.api_base_url}/runs?namespace={self.namespace}"
        
        # The correct payload format for Kubeflow v2 API
        payload = {
            "display_name": run_name,
            "description": f"Run created via Raw KFP Client at {datetime.now()}",
            "pipeline_version_reference": {
                "pipeline_id": pipeline_id
            },
            "runtime_config": {
                "parameters": parameters or {}
            },
            "experiment_id": experiment_id
        }
        
        response = self.session.post(url, json=payload)
        
        if not response.ok:
            error_details = f"Status: {response.status_code}, URL: {url}"
            try:
                error_body = response.json()
                error_details += f", Body: {error_body}"
            except:
                error_body = response.text
                error_details += f", Text: {error_body[:500]}"
            
            raise Exception(f"Failed to create run: {error_details}")
        
        return response.json()
    
    def submit_pipeline_run(
        self,
        pipeline_file: str,
        run_name: str = None,
        experiment_name: str = "Default",
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline submission: upload pipeline, create experiment, submit run
        """
        print(f"🚀 Starting pipeline submission...")
        print(f"   Pipeline file: {pipeline_file}")
        print(f"   Experiment: {experiment_name}")
        
        # Generate run name if not provided
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"   Run name: {run_name}")
        
        try:
            # 1. Health check
            print("\n🔍 Checking API health...")
            health = self.health_check()
            print(f"✅ API is healthy: {health.get('apiServerReady', 'Unknown')}")
            
            # 2. Upload pipeline
            print("\n📤 Uploading pipeline...")
            pipeline_name = f"pipeline_{run_name}"
            uploaded_pipeline = self.upload_pipeline(pipeline_file, pipeline_name)
            pipeline_id = uploaded_pipeline.get('pipeline_id')
            print(f"✅ Pipeline uploaded: {pipeline_id}")
            
            # 3. Create or get experiment
            print(f"\n🧪 Setting up experiment: {experiment_name}")
            try:
                experiment = self.create_experiment(
                    name=experiment_name,
                    description="Automated experiment via Raw KFP Client"
                )
                experiment_id = experiment.get('experiment_id')
                print(f"✅ Created experiment: {experiment_id}")
            except requests.HTTPError as e:
                if e.response.status_code == 409:  # Experiment already exists
                    print(f"ℹ️  Experiment '{experiment_name}' already exists")
                    # Get existing experiment
                    experiments = self.list_experiments()
                    experiment_id = None
                    for exp in experiments.get('experiments', []):
                        if exp.get('display_name') == experiment_name:
                            experiment_id = exp.get('experiment_id')
                            break
                    
                    if not experiment_id:
                        raise Exception(f"Could not find existing experiment: {experiment_name}")
                    print(f"✅ Using existing experiment: {experiment_id}")
                else:
                    raise
            
            # 4. Create run
            print(f"\n🏃 Creating run: {run_name}")
            run_result = self.create_run(
                pipeline_id=pipeline_id,
                experiment_id=experiment_id,
                run_name=run_name,
                parameters=parameters or {}
            )
            run_id = run_result.get('run_id')
            
            print(f"\n🎉 Pipeline run submitted successfully!")
            print(f"   Run ID: {run_id}")
            print(f"   View at: {self.endpoint_url}/#/runs/details/{run_id}")
            
            return {
                'run_id': run_id,
                'pipeline_id': pipeline_id,
                'experiment_id': experiment_id,
                'run_name': run_name,
                'url': f"{self.endpoint_url}/#/runs/details/{run_id}"
            }
            
        except Exception as e:
            print(f"❌ Pipeline submission failed: {e}")
            raise

def test_raw_client():
    """Test the Raw KFP Client"""
    print("🧪 Testing KFP Client")
    print("=" * 50)
    
    client = RawKFPClient()
    
    try:
        # Test basic operations
        health = client.health_check()
        print(f"✅ Health check: {health.get('apiServerReady')}")
        
        pipelines = client.list_pipelines()
        print(f"✅ Found {len(pipelines.get('pipelines', []))} pipelines")
        
        experiments = client.list_experiments()
        print(f"✅ Found {len(experiments.get('experiments', []))} experiments")
        
        runs = client.list_runs()
        print(f"✅ Found {len(runs.get('runs', []))} runs")
        
        print("\n🎉 KFP Client is working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ KFP Client test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the client
    success = test_raw_client()
    
    if success:
        # Test pipeline submission if available
        pipeline_files = [
            "compiled_pipelines/ml_training_latest.yaml",
            "ml_pipeline.yaml"
        ]
        
        client = RawKFPClient()
        
        for pipeline_file in pipeline_files:
            if Path(pipeline_file).exists():
                print(f"\n📋 Found pipeline: {pipeline_file}")
                choice = input("Submit this pipeline? (y/N): ").lower().strip()
                if choice == 'y':
                    try:
                        result = client.submit_pipeline_run(
                            pipeline_file=pipeline_file,
                            experiment_name="Raw Client Tests",
                            parameters={
                                "dataset_url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
                            }
                        )
                        print(f"\n🎯 Success! View your run at: {result['url']}")
                    except Exception as e:
                        print(f"❌ Submission failed: {e}")
                break
        else:
            print(f"\n📋 No compiled pipelines found. Run compile_and_run.py first.")
    else:
        print(f"\n❌ KFP Client is not working. Check authentication and connectivity.") 