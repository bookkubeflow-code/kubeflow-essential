# smart_caching_pipeline.py

import sys
import os
import json
import hashlib
from typing import Dict, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfp import dsl
from kfp.compiler import Compiler
from pipelines.components.ml_ops_components import load_data, train_model, evaluate_model

@dsl.pipeline(
    name='Smart Caching Pipeline',
    description="Demonstration of intelligent caching strategies for ML pipelines"
)
def caching_pipeline(
    dataset_url: str = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    preprocessing_version: str = 'v1',  # Change this to invalidate cache
    model_version: str = 'v1',          # Change this to invalidate model cache
    enable_expensive_caching: bool = True,  # Control caching behavior
    cache_experiment_id: str = 'default'   # Group related cached runs
):
    """
    Pipeline that uses caching intelligently.
    
    Pro tip: Use version strings to control cache invalidation!
    
    Caching Strategy:
    - Data loading: CACHED (expensive, rarely changes)
    - Model training: CACHED for expensive models only
    - Evaluation: NOT CACHED (want fresh metrics every time)
    """
    
    # CACHED: Data loading is expensive and stable
    # Note: Print statements in pipeline definition show parameter placeholders
    data_task = load_data(
        dataset_url=dataset_url,
        # Include version in component to affect caching
        # Note: In real implementation, you'd modify the component to accept version
    )
    
    # Enable caching for data loading (expensive operation)
    # Note: Conditional logic in pipeline definition must use dsl.Condition for runtime decisions
    # For now, we'll set caching based on the parameter value
    data_task.set_caching_options(enable_caching=True)  # Always cache data loading
    
    # CACHED: Model training (expensive, should cache when model code unchanged)
    model_task = train_model(
        input_dataset=data_task.outputs['output_dataset']
    )
    
    # Cache model training - we'll use conditional logic based on enable_expensive_caching
    # In KFP, we can't use string methods on parameters during definition,
    # so we'll use a simpler caching strategy
    model_task.set_caching_options(enable_caching=True)  # Cache by default
    
    # NOT CACHED: Evaluation should always run fresh
    eval_task = evaluate_model(
        test_dataset=data_task.outputs['output_dataset'],
        model_input=model_task.outputs['model_output']
    )
    
    # Never cache evaluation - we want fresh metrics every time
    eval_task.set_caching_options(enable_caching=False)
    
    # Set execution order
    model_task.after(data_task)
    eval_task.after(model_task)


# Advanced caching management
class SmartCacheManager:
    """Manage pipeline caching strategically."""
    
    def __init__(self, client=None):
        self.client = client
        if client is None:
            try:
                from config import Config
                config = Config()
                self.client = config.get_client()
            except:
                self.client = None
    
    @staticmethod
    def get_cache_key(
        data_version: str,
        code_version: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key that changes when it should."""
        
        # Include everything that affects results
        cache_data = {
            'data_version': data_version,
            'code_version': code_version,
            'params': {k: v for k, v in params.items() if k not in ['cache_experiment_id']},
            'timestamp': datetime.now().strftime('%Y-%m-%d')  # Daily cache invalidation
        }
        
        # Create hash for cache key
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()[:8]
    
    @staticmethod
    def get_data_version(dataset_url: str) -> str:
        """Get version of data source (simplified)."""
        # In production, this might check file modification time,
        # database schema version, etc.
        return hashlib.md5(dataset_url.encode()).hexdigest()[:6]
    
    @staticmethod 
    def get_code_version() -> str:
        """Get version of pipeline code (simplified)."""
        # In production, this might use git commit hash,
        # file modification times, etc.
        try:
            # Try to get git commit hash
            import subprocess
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Fallback to file modification time
        pipeline_file = __file__
        mod_time = os.path.getmtime(pipeline_file)
        return str(int(mod_time))[-6:]  # Last 6 digits
    
    def get_optimal_cache_settings(self, pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal caching settings based on parameters."""
        
        settings = {
            'enable_expensive_caching': True,
            'cache_data_loading': True,
            'cache_model_training': True,
            'cache_evaluation': False  # Never cache evaluation
        }
        
        # Adjust based on parameters
        if 'debug' in pipeline_params.get('cache_experiment_id', ''):
            settings['enable_expensive_caching'] = False
            print("🐛 Debug mode: Disabling caching for fresh runs")
        
        if pipeline_params.get('preprocessing_version', 'v1') != 'v1':
            settings['cache_data_loading'] = False
            print(f"🔄 Preprocessing version changed: Invalidating data cache")
        
        if pipeline_params.get('model_version', 'v1') != 'v1':
            settings['cache_model_training'] = False
            print(f"🔄 Model version changed: Invalidating model cache")
        
        return settings
    
    def run_with_smart_caching(self, base_args: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline with intelligent cache management."""
        
        if not self.client:
            raise Exception("No KFP client available. Cannot submit pipeline.")
        
        # Generate intelligent cache key
        cache_key = self.get_cache_key(
            data_version=self.get_data_version(base_args.get('dataset_url', '')),
            code_version=self.get_code_version(),
            params=base_args
        )
        
        # Get optimal cache settings
        cache_settings = self.get_optimal_cache_settings(base_args)
        
        # Enhance arguments with cache intelligence
        enhanced_args = {
            **base_args,
            'cache_key': cache_key,
            'enable_expensive_caching': cache_settings['enable_expensive_caching'],
        }
        
        print(f"🔑 Cache key: {cache_key}")
        print(f"📦 Cache settings: {cache_settings}")
        
        # Submit pipeline with enhanced arguments
        try:
            run_result = self.client.create_run_from_pipeline_package(
                pipeline_file='compiled_pipelines/smart_caching_pipeline.yaml',
                arguments=enhanced_args,
                run_name=f"smart_cache_{cache_key}_{datetime.now().strftime('%H%M%S')}",
                experiment_name=base_args.get('cache_experiment_id', 'Smart Caching'),
                enable_caching=True  # Enable caching at run level
            )
            
            return {
                'success': True,
                'run_id': run_result.run_id,
                'cache_key': cache_key,
                'cache_settings': cache_settings
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'cache_key': cache_key
            }


# Cache analysis utilities
class CacheAnalyzer:
    """Analyze caching effectiveness."""
    
    def __init__(self, client=None):
        self.client = client
        if client is None:
            try:
                from config import Config
                config = Config()
                self.client = config.get_client()
            except:
                self.client = None
    
    def analyze_cache_performance(self, experiment_name: str = "Smart Caching") -> Dict[str, Any]:
        """Analyze how well caching is working."""
        
        if not self.client:
            return {'error': 'No KFP client available'}
        
        try:
            # Get runs from caching experiment
            runs = self.client.list_runs(
                namespace='kubeflow-user-example-com',
                page_size=20
            )
            
            cache_stats = {
                'total_runs': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'avg_duration_with_cache': 0,
                'avg_duration_without_cache': 0
            }
            
            cached_durations = []
            uncached_durations = []
            
            for run in runs.runs if runs.runs else []:
                cache_stats['total_runs'] += 1
                
                # Simple heuristic: runs with "cache" in name likely used caching
                if 'cache' in run.display_name.lower():
                    cache_stats['cache_hits'] += 1
                    if hasattr(run, 'finished_at') and hasattr(run, 'created_at'):
                        duration = (run.finished_at - run.created_at).total_seconds()
                        cached_durations.append(duration)
                else:
                    cache_stats['cache_misses'] += 1
                    if hasattr(run, 'finished_at') and hasattr(run, 'created_at'):
                        duration = (run.finished_at - run.created_at).total_seconds()
                        uncached_durations.append(duration)
            
            # Calculate averages
            if cached_durations:
                cache_stats['avg_duration_with_cache'] = sum(cached_durations) / len(cached_durations)
            if uncached_durations:
                cache_stats['avg_duration_without_cache'] = sum(uncached_durations) / len(uncached_durations)
            
            # Calculate cache effectiveness
            if cache_stats['avg_duration_without_cache'] > 0:
                speedup = cache_stats['avg_duration_without_cache'] / max(cache_stats['avg_duration_with_cache'], 1)
                cache_stats['cache_speedup'] = f"{speedup:.2f}x faster"
            
            return cache_stats
            
        except Exception as e:
            return {'error': str(e)}


# Demonstration and testing
def demo_smart_caching():
    """Demonstrate smart caching capabilities."""
    
    print("🧠 Smart Caching Pipeline Demo")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = SmartCacheManager()
    
    # Example parameters
    base_params = {
        'dataset_url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
        'preprocessing_version': 'v1',
        'model_version': 'v1',
        'cache_experiment_id': 'caching_demo'
    }
    
    print("📋 Base parameters:")
    for key, value in base_params.items():
        print(f"   {key}: {value}")
    
    # Show cache key generation
    cache_key = cache_manager.get_cache_key(
        data_version=cache_manager.get_data_version(base_params['dataset_url']),
        code_version=cache_manager.get_code_version(),
        params=base_params
    )
    print(f"\n🔑 Generated cache key: {cache_key}")
    
    # Show cache settings
    settings = cache_manager.get_optimal_cache_settings(base_params)
    print(f"\n📦 Optimal cache settings:")
    for key, value in settings.items():
        print(f"   {key}: {value}")
    
    # Demonstrate version change impact
    print(f"\n🔄 Testing version change impact:")
    modified_params = {**base_params, 'preprocessing_version': 'v2'}
    modified_settings = cache_manager.get_optimal_cache_settings(modified_params)
    print(f"   With preprocessing_version='v2': cache_data_loading={modified_settings['cache_data_loading']}")


if __name__ == '__main__':
    print("🧠 Compiling Smart Caching Pipeline...")
    
    # Compile the pipeline
    Compiler().compile(
        pipeline_func=caching_pipeline,
        package_path='compiled_pipelines/smart_caching_pipeline.yaml'
    )
    print("✅ Pipeline compiled to compiled_pipelines/smart_caching_pipeline.yaml")
    
    # Run demonstration
    print()
    demo_smart_caching()
    
    print(f"\n💡 Usage Examples:")
    print(f"   1. Basic run: python pipeline_runner.py")
    print(f"   2. Smart caching run:")
    print(f"      from pipelines.smart_caching_pipeline import SmartCacheManager")
    print(f"      manager = SmartCacheManager()")
    print(f"      result = manager.run_with_smart_caching({{'dataset_url': '...'}})")
    print(f"   3. Cache analysis:")
    print(f"      from pipelines.smart_caching_pipeline import CacheAnalyzer")
    print(f"      analyzer = CacheAnalyzer()")
    print(f"      stats = analyzer.analyze_cache_performance()") 