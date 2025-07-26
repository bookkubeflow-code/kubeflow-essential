# compile_and_run.py

from kfp import compiler
from kfp import dsl
import kfp
from pathlib import Path
import yaml
import json
from datetime import datetime
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PipelineCompiler:
    """Smart pipeline compilation with validation and versioning."""
    
    def __init__(self, output_dir: str = "compiled_pipelines"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def compile_pipeline(
        self, 
        pipeline_func, 
        pipeline_name: str = None,
        validate: bool = True,
        add_metadata: bool = True
    ):
        """Compile with bells and whistles."""
        
        # Generate versioned filename
        pipeline_name = pipeline_name or pipeline_func.__name__
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{pipeline_name}_{timestamp}.yaml"
        
        # Compile
        print(f"🔨 Compiling {pipeline_name}...")
        try:
            compiler.Compiler().compile(
                pipeline_func=pipeline_func,
                package_path=str(output_path)
            )
            print(f"✅ Compiled to: {output_path}")
        except Exception as e:
            print(f"❌ Compilation failed: {e}")
            return None
            
        # Validate if requested
        if validate:
            if self._validate_pipeline(output_path):
                print("✅ Pipeline validation passed")
            else:
                print("⚠️  Pipeline validation warnings (see above)")
        
        # Add metadata
        if add_metadata:
            self._add_compilation_metadata(output_path, pipeline_func)
            
        # Create a 'latest' symlink for convenience
        latest_link = self.output_dir / f"{pipeline_name}_latest.yaml"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_path.name)
        
        return output_path
    
    def _validate_pipeline(self, pipeline_path: Path) -> bool:
        """Validate the compiled pipeline."""
        with open(pipeline_path, 'r') as f:
            pipeline_spec = yaml.safe_load(f)
        
        warnings = []
        
        # Check for common issues
        if 'spec' in pipeline_spec:
            spec = pipeline_spec['spec']
            
            # Check for missing container images
            templates = spec.get('templates', [])
            for template in templates:
                if 'container' in template:
                    image = template['container'].get('image', '')
                    if not image:
                        warnings.append(f"Template '{template.get('name')}' has no image")
                    elif image == 'python:3.7':  # Common mistake
                        warnings.append(f"Template '{template.get('name')}' uses outdated Python 3.7")
                    elif image == 'python:3.8':
                        warnings.append(f"Template '{template.get('name')}' uses outdated Python 3.8")
            
            # Check for resource limits
            for template in templates:
                if 'container' in template:
                    container = template['container']
                    if 'resources' not in container:
                        warnings.append(f"Template '{template.get('name')}' has no resource limits")
                    else:
                        resources = container.get('resources', {})
                        if 'requests' not in resources:
                            warnings.append(f"Template '{template.get('name')}' has no resource requests")
                        if 'limits' not in resources:
                            warnings.append(f"Template '{template.get('name')}' has no resource limits")
        
        # Print warnings
        for warning in warnings:
            print(f"⚠️  {warning}")
            
        return len(warnings) == 0
    
    def _add_compilation_metadata(self, pipeline_path: Path, pipeline_func):
        """Add useful metadata to the compiled pipeline."""
        
        # Read the compiled pipeline
        with open(pipeline_path, 'r') as f:
            content = f.read()
        
        # Handle different pipeline object types
        func_name = getattr(pipeline_func, '__name__', 
                           getattr(pipeline_func, 'name', 'unknown_pipeline'))
        module_name = getattr(pipeline_func, '__module__', 'unknown_module')
        description = getattr(pipeline_func, '__doc__', 'No description')
        
        # Add metadata as comments
        metadata = f"""# Pipeline Compilation Metadata
# Compiled at: {datetime.now().isoformat()}
# Function: {func_name}
# Module: {module_name}
# Description: {description or 'No description'}
# Python version: 3.11.13
# KFP SDK version: 2.12.1
# 
"""
        
        # Write back with metadata
        with open(pipeline_path, 'w') as f:
            f.write(metadata + content)
    
    def try_submit_pipeline(self, pipeline_path: Path, pipeline_name: str = None):
        """Attempt to submit pipeline with advanced Dex authentication, graceful fallback to UI."""
        pipeline_name = pipeline_name or "ml-training-run"
        
        print("\n🚀 Attempting to submit pipeline...")
        
        # Method 1: Try Dex authentication
        try:
            print("🔐 Trying advanced Dex authentication...")
            from dex_auth import AuthenticatedKFPManager
            
            auth_manager = AuthenticatedKFPManager()
            
            run = auth_manager.submit_pipeline(
                pipeline_path=str(pipeline_path),
                run_name=f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                experiment_name="ML Training Experiments",
                arguments={
                    'dataset_url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
                }
            )
            
            print(f"✅ Pipeline submitted via Dex authentication!")
            return True
            
        except ImportError:
            print("⚠️  Dex auth module not available, trying basic client...")
        except Exception as e:
            print(f"⚠️  Dex authentication failed: {e}")
            print("   Falling back to basic client...")
        
        # Method 2: Try basic client (will likely fail with 401)
        try:
            from config import config
            client = config.get_client()
            
            # Create experiment
            experiment_name = "ML Training Experiments"
            try:
                experiment = client.create_experiment(
                    name=experiment_name,
                    description="Automated ML pipeline experiments"
                )
                print(f"✅ Created experiment: {experiment_name}")
            except Exception as e:
                # Experiment might already exist
                print(f"ℹ️  Using existing experiment: {experiment_name}")
            
            # Submit run
            run_name = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = client.create_run_from_pipeline_package(
                pipeline_file=str(pipeline_path),
                arguments={
                    'dataset_url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
                },
                experiment_name=experiment_name,
                run_name=run_name
            )
            
            print(f"✅ Run submitted successfully!")
            print(f"🔗 Run ID: {run.run_id}")
            print(f"🌐 View at: http://localhost:8080/#/runs/details/{run.run_id}")
            return True
            
        except Exception as e:
            print(f"❌ All programmatic submission methods failed: {e}")
            print("\n📋 Manual Submission Instructions:")
            print("=" * 50)
            print("1. Open browser: http://localhost:8080")
            print("2. Login with any email/password (e.g., user@example.com / 12341234)")
            print("3. Navigate to: Pipelines → Upload Pipeline")
            print(f"4. Upload file: {pipeline_path}")
            print("5. Click 'Create Run' and name it")
            print("6. Set parameters if needed and submit")
            print("=" * 50)
            return False

# Usage example and main execution
if __name__ == "__main__":
    # Import your actual pipeline
    try:
        from pipelines.ml_training_pipeline import ml_training_pipeline
        pipeline_func = ml_training_pipeline
        pipeline_name = "ml_training"
        
        # Handle different pipeline object types
        func_name = getattr(pipeline_func, '__name__', 
                           getattr(pipeline_func, 'name', 'unknown_pipeline'))
        print(f"✅ Found pipeline: {func_name}")
    except ImportError as e:
        print(f"❌ Could not import pipeline: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    # Create compiler instance
    compiler_instance = PipelineCompiler()
    
    # Compile with all features
    print("🎯 Starting advanced pipeline compilation...")
    compiled_path = compiler_instance.compile_pipeline(
        pipeline_func=pipeline_func,
        pipeline_name=pipeline_name,
        validate=True,
        add_metadata=True
    )
    
    if compiled_path:
        print(f"\n📁 Pipeline files available:")
        print(f"   • Versioned: {compiled_path}")
        print(f"   • Latest: {compiled_path.parent / f'{pipeline_name}_latest.yaml'}")
        
        # Try to submit (will gracefully handle Dex auth issues)
        compiler_instance.try_submit_pipeline(compiled_path, pipeline_name)
        
        print(f"\n🎉 Compilation complete!")
        print(f"📊 Pipeline ready for execution in Kubeflow UI")
    else:
        print("❌ Compilation failed. Please check the errors above.")
        sys.exit(1) 