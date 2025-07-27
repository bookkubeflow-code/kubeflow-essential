# config.py - Put this in your project root
import os
from pathlib import Path
import sys

class Config:
    def __init__(self):
        # Verify Python version
        if sys.version_info < (3, 11):
            raise RuntimeError(f"Python 3.11+ required, got {sys.version}")
        
        # Local development
        self.LOCAL_MODE = os.getenv('KFP_LOCAL_MODE', 'false').lower() == 'true'
        
        # Server configuration
        self.KFP_HOST = os.getenv('KFP_HOST', 'http://localhost:8080')
        self.KFP_NAMESPACE = os.getenv('KFP_NAMESPACE', 'kubeflow')
        
        # Storage configuration  
        self.PIPELINE_ROOT = os.getenv(
            'PIPELINE_ROOT', 
            'gs://my-bucket/pipelines' if not self.LOCAL_MODE else './outputs'
        )
        
        # Authentication (for cloud deployments)
        self.USE_AUTH = os.getenv('KFP_USE_AUTH', 'false').lower() == 'true'
        self.AUTH_TOKEN = os.getenv('KFP_AUTH_TOKEN', '')
        
        # Custom Dex authentication settings (for our local setup)
        self.USE_DEX_AUTH = os.getenv('KFP_USE_DEX_AUTH', 'true').lower() == 'true'
        self.KUBEFLOW_ENDPOINT = os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8080')
        self.KUBEFLOW_USERNAME = os.getenv('KUBEFLOW_USERNAME', 'user@example.com')
        self.KUBEFLOW_PASSWORD = os.getenv('KUBEFLOW_PASSWORD', '')
        self.KUBEFLOW_USER_NAMESPACE = os.getenv('KUBEFLOW_USER_NAMESPACE', 'kubeflow-user-example-com')
        
    def get_client(self):
        """Get KFP client with appropriate authentication method."""
        import kfp
        
        # Try custom Dex authentication first (for local development)
        if self.USE_DEX_AUTH and os.path.exists('dex_auth.py'):
            try:
                print("🔐 Attempting custom Dex authentication...")
                from dex_auth import DexSessionManager
                
                # Create DexSessionManager with proper parameters
                auth_manager = DexSessionManager(
                    endpoint_url=self.KUBEFLOW_ENDPOINT,
                    dex_username=self.KUBEFLOW_USERNAME,
                    dex_password=self.KUBEFLOW_PASSWORD
                )
                client = auth_manager.get_authenticated_client()
                
                # Verify connection with our custom client
                print(f"✓ Connected to Kubeflow Pipelines at {self.KFP_HOST} (Dex Auth)")
                return client
                
            except Exception as e:
                print(f"⚠️ Dex authentication failed: {e}")
                print("Falling back to standard authentication...")
        
        # Standard KFP client authentication
        try:
            if self.USE_AUTH:
                client = kfp.Client(
                    host=self.KFP_HOST,
                    namespace=self.KFP_NAMESPACE,
                    existing_token=self.AUTH_TOKEN
                )
            else:
                client = kfp.Client(
                    host=self.KFP_HOST,
                    namespace=self.KFP_NAMESPACE
                )
            
            # Verify connection
            _ = client.list_pipelines(page_size=1)
            print(f"✓ Connected to Kubeflow Pipelines at {self.KFP_HOST}")
            return client
            
        except Exception as e:
            print(f"✗ Failed to connect to {self.KFP_HOST}")
            print(f"  Error: {e}")
            print("\n  Troubleshooting:")
            print("  1. Check if port-forward is running: ps aux | grep port-forward")
            print("  2. Verify Kubeflow is running: kubectl get pods -n kubeflow")
            print("  3. Try accessing UI: http://localhost:8080")
            print("  4. Set up credentials: python setup_credentials.py")
            print("  5. Source credentials: source kubeflow.env")
            raise
    
    def get_custom_client(self):
        """Get our custom KFP client with direct API access."""
        try:
            from kfp_client import KFPClient
            print("🔧 Using custom KFP client...")
            return KFPClient()
        except Exception as e:
            print(f"⚠️ Custom client failed: {e}")
            return None
    
    def print_config(self):
        """Print current configuration for debugging."""
        print("📋 Current Configuration:")
        print("=" * 40)
        print(f"Python Version: {sys.version}")
        print(f"Local Mode: {self.LOCAL_MODE}")
        print(f"KFP Host: {self.KFP_HOST}")
        print(f"KFP Namespace: {self.KFP_NAMESPACE}")
        print(f"Pipeline Root: {self.PIPELINE_ROOT}")
        print(f"Use Auth: {self.USE_AUTH}")
        print(f"Use Dex Auth: {self.USE_DEX_AUTH}")
        print(f"Kubeflow Endpoint: {self.KUBEFLOW_ENDPOINT}")
        print(f"Username: {self.KUBEFLOW_USERNAME}")
        print(f"User Namespace: {self.KUBEFLOW_USER_NAMESPACE}")
        print("=" * 40)
    
    def test_connection(self):
        """Test connection to Kubeflow with detailed diagnostics."""
        print("🧪 Testing Kubeflow Connection...")
        print("=" * 50)
        
        # Test 1: Environment variables
        print("1️⃣ Environment Variables:")
        env_vars = ['KUBEFLOW_ENDPOINT', 'KUBEFLOW_USERNAME', 'KUBEFLOW_USER_NAMESPACE']
        for var in env_vars:
            value = os.getenv(var, 'Not set')
            print(f"   {var}: {value}")
        print()
        
        # Test 2: Custom authentication
        if self.USE_DEX_AUTH:
            try:
                print("2️⃣ Testing Custom Dex Authentication:")
                from dex_auth import DexSessionManager
                auth_manager = DexSessionManager(
                    endpoint_url=self.KUBEFLOW_ENDPOINT,
                    dex_username=self.KUBEFLOW_USERNAME,
                    dex_password=self.KUBEFLOW_PASSWORD
                )
                client = auth_manager.get_authenticated_client()
                print("   ✅ Dex authentication successful")
                print()
                
                # Test 3: API access
                print("3️⃣ Testing API Access:")
                from kfp_client import KFPClient
                kfp_client = KFPClient()
                
                health = kfp_client.health_check()
                print(f"   Health check: {'✅ Pass' if health else '❌ Fail'}")
                
                try:
                    pipelines = kfp_client.list_pipelines()
                    print(f"   Found {len(pipelines.get('pipelines', []))} pipelines")
                    
                    experiments = kfp_client.list_experiments()
                    print(f"   Found {len(experiments.get('experiments', []))} experiments")
                    
                    runs = kfp_client.list_runs()
                    print(f"   Found {len(runs.get('runs', []))} runs")
                    
                    print("   ✅ All API tests passed")
                    
                except Exception as api_error:
                    print(f"   ❌ API test failed: {api_error}")
                
            except Exception as auth_error:
                print(f"   ❌ Authentication failed: {auth_error}")
        
        # Test 4: Standard KFP client (fallback)
        print("\n4️⃣ Testing Standard KFP Client:")
        try:
            client = self.get_client()
            if client:
                pipelines = client.list_pipelines(page_size=1)
                print("   ✅ Standard client working")
            else:
                print("   ❌ Standard client failed")
        except Exception as std_error:
            print(f"   ❌ Standard client error: {std_error}")
        
        print("\n🎯 Connection test completed!")

# Create global config instance
config = Config()

if __name__ == '__main__':
    # When run directly, perform configuration test
    config.test_connection()
    config.print_config() 