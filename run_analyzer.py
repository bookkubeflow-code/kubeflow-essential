# run_analyzer.py

from enum import Enum
from typing import List, Dict, Optional, Any
import pandas as pd
import json
import kfp
from datetime import datetime, timedelta
import re
import logging
from pathlib import Path

class RunStatus(Enum):
    """The real statuses you'll see in production."""
    PENDING = "Pending"           # Waiting to be scheduled
    RUNNING = "Running"           # At least one component is running
    SUCCEEDED = "Succeeded"       # All components completed successfully
    FAILED = "Failed"            # At least one component failed
    ERROR = "Error"              # System error (not your fault!)
    SKIPPED = "Skipped"          # Conditional execution skipped this
    TERMINATED = "Terminated"     # Someone killed it
    OMITTED = "Omitted"          # Parent condition was false

class RunAnalyzer:
    """Analyze pipeline runs to understand what's really happening."""
    
    def __init__(self, client: kfp.Client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # Set up logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def get_run_timeline(self, run_id: str) -> pd.DataFrame:
        """Get detailed timeline of a run."""
        try:
            run = self.client.get_run(run_id)
            
            # Handle different KFP API versions and response structures
            timeline = []
            run_obj = None
            
            # Try to access run data - KFP v2 API returns different structures
            if hasattr(run, 'run'):
                run_obj = run.run
            elif hasattr(run, 'display_name'):
                run_obj = run
            else:
                self.logger.warning(f"Unknown run object structure: {type(run)}")
                run_obj = run
            
            # Try to parse the workflow status if available
            if run_obj and hasattr(run_obj, 'pipeline_runtime'):
                if hasattr(run_obj.pipeline_runtime, 'workflow_manifest'):
                    try:
                        workflow = json.loads(run_obj.pipeline_runtime.workflow_manifest)
                        nodes = workflow.get('status', {}).get('nodes', {})
                        
                        for node_id, node in nodes.items():
                            timeline.append({
                                'component': node.get('displayName', node_id),
                                'type': node.get('type'),
                                'status': node.get('phase'),
                                'started': node.get('startedAt'),
                                'finished': node.get('finishedAt'),
                                'duration': self._calculate_duration(node),
                                'message': node.get('message', ''),
                                'node_id': node_id
                            })
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(f"Could not parse workflow manifest: {e}")
            
            # Fallback: create basic timeline from run info
            if not timeline and run_obj:
                # Get run attributes safely
                name = getattr(run_obj, 'display_name', getattr(run_obj, 'name', 'Unknown'))
                status = getattr(run_obj, 'state', getattr(run_obj, 'status', 'Unknown'))
                created_at = getattr(run_obj, 'created_at', None)
                finished_at = getattr(run_obj, 'finished_at', None)
                
                timeline.append({
                    'component': name,
                    'type': 'Pipeline',
                    'status': status,
                    'started': created_at.isoformat() if created_at else None,
                    'finished': finished_at.isoformat() if finished_at else None,
                    'duration': self._calculate_run_duration(run_obj),
                    'message': getattr(run_obj, 'error', ''),
                    'node_id': run_id
                })
            
            df = pd.DataFrame(timeline)
            if not df.empty and 'started' in df.columns:
                df['started'] = pd.to_datetime(df['started'], errors='coerce')
                df['finished'] = pd.to_datetime(df['finished'], errors='coerce')
                # Sort with NaN values at the end
                df = df.sort_values('started', na_position='last')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get timeline for run {run_id}: {e}")
            return pd.DataFrame()
    
    def _calculate_duration(self, node: Dict[str, Any]) -> Optional[float]:
        """Calculate duration in seconds from node info."""
        try:
            started = node.get('startedAt')
            finished = node.get('finishedAt')
            
            if started and finished:
                start_time = pd.to_datetime(started)
                end_time = pd.to_datetime(finished)
                return (end_time - start_time).total_seconds()
        except Exception:
            pass
        return None
    
    def _calculate_run_duration(self, run) -> Optional[float]:
        """Calculate run duration from run object."""
        try:
            created_at = getattr(run, 'created_at', None)
            finished_at = getattr(run, 'finished_at', None)
            if created_at and finished_at:
                return (finished_at - created_at).total_seconds()
        except Exception:
            pass
        return None
    
    def diagnose_failure(self, run_id: str) -> Dict[str, Any]:
        """Figure out why a run failed."""
        timeline = self.get_run_timeline(run_id)
        
        # Find failed components
        failed = timeline[timeline['status'].isin(['Failed', 'Error'])] if not timeline.empty else pd.DataFrame()
        
        diagnosis = {
            'run_id': run_id,
            'total_components': len(timeline),
            'failed_components': len(failed),
            'failures': [],
            'overall_status': 'Unknown'
        }
        
        # Get overall run status
        try:
            run = self.client.get_run(run_id)
            # Handle different API response structures
            run_obj = run.run if hasattr(run, 'run') else run
            
            diagnosis['overall_status'] = getattr(run_obj, 'state', getattr(run_obj, 'status', 'Unknown'))
            diagnosis['run_name'] = getattr(run_obj, 'display_name', getattr(run_obj, 'name', 'Unknown'))
            
            created_at = getattr(run_obj, 'created_at', None)
            finished_at = getattr(run_obj, 'finished_at', None)
            
            diagnosis['created_at'] = created_at.isoformat() if created_at else None
            diagnosis['finished_at'] = finished_at.isoformat() if finished_at else None
        except Exception as e:
            self.logger.warning(f"Could not get run details: {e}")
        
        # Analyze failures
        for _, failure in failed.iterrows():
            # Get component logs (simplified for now)
            logs = self._get_component_logs(run_id, failure.get('node_id', ''))
            
            failure_info = {
                'component': failure['component'],
                'message': failure.get('message', ''),
                'duration_before_failure': failure.get('duration'),
                'likely_cause': self._guess_failure_cause(logs, failure.get('message', '')),
                'started_at': failure.get('started'),
                'failed_at': failure.get('finished')
            }
            
            diagnosis['failures'].append(failure_info)
        
        # If no component-level failures but run failed, analyze run-level
        if len(failed) == 0 and diagnosis['overall_status'] in ['Failed', 'Error']:
            try:
                run = self.client.get_run(run_id)
                run_obj = run.run if hasattr(run, 'run') else run
                error_msg = getattr(run_obj, 'error', '')
                diagnosis['failures'].append({
                    'component': 'Pipeline',
                    'message': error_msg,
                    'likely_cause': self._guess_failure_cause('', error_msg),
                    'duration_before_failure': self._calculate_run_duration(run_obj)
                })
            except Exception:
                pass
        
        return diagnosis
    
    def _get_component_logs(self, run_id: str, node_id: str) -> str:
        """Get component logs (simplified implementation)."""
        # Note: In a full implementation, this would query the actual logs
        # For now, return empty string as log access varies by Kubeflow setup
        try:
            # This would typically involve calling the Kubeflow logs API
            # or Kubernetes API to get pod logs
            return ""
        except Exception:
            return ""
    
    def _guess_failure_cause(self, logs: str, error_message: str) -> str:
        """Pattern match common failures."""
        
        error_text = (logs + " " + error_message).lower()
        
        # OOM kills
        if any(term in error_text for term in ['oomkilled', 'memory limit', 'out of memory']):
            return "Out of memory - increase memory limits"
        
        # Missing files
        if any(term in error_text for term in ['filenotfounderror', 'no such file', 'file not found']):
            return "Missing input file - check artifact paths"
        
        # Import errors
        if 'modulenotfounderror' in error_text:
            # Try to extract module name
            module_match = re.search(r"modulenotfounderror.*'([^']+)'", error_text)
            if module_match:
                return f"Missing dependency: {module_match.group(1)}"
            return "Missing Python dependency - check requirements"
        
        # Timeout
        if any(term in error_text for term in ['timeout', 'timed out', 'deadline exceeded']):
            return "Component timeout - increase timeout or optimize code"
        
        # Permission errors
        if any(term in error_text for term in ['permission denied', 'accessdenied', 'unauthorized']):
            return "Permission error - check service account and IAM roles"
        
        # Network errors
        if any(term in error_text for term in ['connection refused', 'network unreachable', 'dns resolution']):
            return "Network connectivity issue - check network policies"
        
        # Resource constraints
        if any(term in error_text for term in ['insufficient cpu', 'insufficient memory', 'resource quotas']):
            return "Resource constraints - check cluster capacity"
        
        # Container errors
        if any(term in error_text for term in ['imagepullbackoff', 'image not found', 'pull access denied']):
            return "Container image issue - check image name and registry access"
        
        return "Unknown - check component logs for details"
    
    def get_run_cost_estimate(self, run_id: str) -> Dict[str, float]:
        """Estimate the cost of a run (if you're on cloud)."""
        timeline = self.get_run_timeline(run_id)
        
        # Simple cost model - adjust for your cloud provider
        CPU_HOUR_COST = 0.05  # dollars per CPU hour
        GPU_HOUR_COST = 2.50  # dollars per GPU hour
        MEMORY_GB_HOUR_COST = 0.01  # dollars per GB-hour
        
        total_cpu_hours = 0
        total_gpu_hours = 0
        total_memory_gb_hours = 0
        
        for _, component in timeline.iterrows():
            duration_hours = component.get('duration', 0) / 3600 if component.get('duration') else 0
            
            # This is simplified - in reality you'd parse resource requests from the workflow
            component_name = component.get('component', '').lower()
            
            if any(gpu_term in component_name for gpu_term in ['gpu', 'cuda', 'nvidia']):
                total_gpu_hours += duration_hours
                total_memory_gb_hours += duration_hours * 16  # Assume 16GB per GPU component
            else:
                total_cpu_hours += duration_hours * 2  # Assume 2 CPU per component
                total_memory_gb_hours += duration_hours * 8   # Assume 8GB per CPU component
        
        estimated_cost = (
            total_cpu_hours * CPU_HOUR_COST + 
            total_gpu_hours * GPU_HOUR_COST +
            total_memory_gb_hours * MEMORY_GB_HOUR_COST
        )
        
        return {
            'cpu_hours': round(total_cpu_hours, 2),
            'gpu_hours': round(total_gpu_hours, 2),
            'memory_gb_hours': round(total_memory_gb_hours, 2),
            'estimated_cost': round(estimated_cost, 2)
        }
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get comprehensive run summary."""
        timeline = self.get_run_timeline(run_id)
        diagnosis = self.diagnose_failure(run_id)
        cost = self.get_run_cost_estimate(run_id)
        
        # Calculate statistics
        if not timeline.empty:
            durations = timeline['duration'].dropna()
            stats = {
                'total_components': len(timeline),
                'completed_components': len(timeline[timeline['status'] == 'Succeeded']),
                'failed_components': len(timeline[timeline['status'].isin(['Failed', 'Error'])]),
                'avg_component_duration': round(durations.mean(), 2) if len(durations) > 0 else 0,
                'max_component_duration': round(durations.max(), 2) if len(durations) > 0 else 0,
                'total_execution_time': round(durations.sum(), 2) if len(durations) > 0 else 0
            }
        else:
            stats = {
                'total_components': 0,
                'completed_components': 0,
                'failed_components': 0,
                'avg_component_duration': 0,
                'max_component_duration': 0,
                'total_execution_time': 0
            }
        
        return {
            'run_id': run_id,
            'status': diagnosis['overall_status'],
            'statistics': stats,
            'cost_estimate': cost,
            'failure_analysis': diagnosis,
            'timeline_available': not timeline.empty
        }
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs."""
        comparisons = []
        
        for run_id in run_ids:
            summary = self.get_run_summary(run_id)
            comparisons.append({
                'run_id': run_id[:8] + '...',  # Shortened for display
                'status': summary['status'],
                'components': summary['statistics']['total_components'],
                'failures': summary['statistics']['failed_components'],
                'duration_minutes': round(summary['statistics']['total_execution_time'] / 60, 1),
                'estimated_cost': summary['cost_estimate']['estimated_cost']
            })
        
        return pd.DataFrame(comparisons)

def analyze_run(run_id: str, client: kfp.Client = None) -> None:
    """Convenience function to analyze a single run."""
    if client is None:
        # Try to get client from config
        try:
            from config import Config
            config = Config()
            client = config.get_client()
        except ImportError:
            print("❌ No KFP client provided and config.py not available")
            return
    
    analyzer = RunAnalyzer(client)
    
    print(f"🔍 Analyzing Run: {run_id}")
    print("=" * 60)
    
    # Get comprehensive summary
    summary = analyzer.get_run_summary(run_id)
    
    print(f"📊 Run Summary:")
    print(f"   Status: {summary['status']}")
    print(f"   Components: {summary['statistics']['total_components']}")
    print(f"   Completed: {summary['statistics']['completed_components']}")
    print(f"   Failed: {summary['statistics']['failed_components']}")
    print(f"   Total Duration: {summary['statistics']['total_execution_time']:.1f}s")
    print(f"   Estimated Cost: ${summary['cost_estimate']['estimated_cost']}")
    
    # Show failures if any
    if summary['failure_analysis']['failures']:
        print(f"\n❌ Failure Analysis:")
        for i, failure in enumerate(summary['failure_analysis']['failures'], 1):
            print(f"   {i}. Component: {failure['component']}")
            print(f"      Cause: {failure['likely_cause']}")
            if failure.get('message'):
                print(f"      Message: {failure['message'][:100]}...")
    
    # Show timeline if available
    if summary['timeline_available']:
        print(f"\n⏱️ Component Timeline:")
        timeline = analyzer.get_run_timeline(run_id)
        for _, component in timeline.iterrows():
            status_emoji = {'Succeeded': '✅', 'Failed': '❌', 'Running': '🔄', 'Pending': '⏳'}.get(component['status'], '❓')
            duration = f"({component['duration']:.1f}s)" if component['duration'] else "(duration unknown)"
            print(f"   {status_emoji} {component['component']} {duration}")

# Real-world usage examples
if __name__ == "__main__":
    print("🔬 Kubeflow Run Analyzer")
    print("=" * 40)
    
    try:
        # Initialize with our config
        from config import Config
        config = Config()
        client = config.get_client()
        
        analyzer = RunAnalyzer(client)
        
        print("✅ Analyzer initialized successfully!")
        print("\nAvailable commands:")
        print("1. analyzer.get_run_summary('run-id')")
        print("2. analyzer.diagnose_failure('failed-run-id')")
        print("3. analyzer.get_run_cost_estimate('run-id')")
        print("4. analyze_run('run-id')  # Comprehensive analysis")
        
        # Get recent runs for demonstration
        try:
            # Get runs directly using the client (avoiding pipeline_runner bug)
            runs = client.list_runs(namespace='kubeflow-user-example-com', page_size=5)
            
            if runs.runs:
                print(f"\n📋 Recent Runs (for testing):")
                for i, run in enumerate(runs.runs[:3], 1):
                    status = getattr(run, 'state', 'Unknown')
                    print(f"   {i}. {run.display_name} ({run.run_id[:8]}...) - {status}")
                
                # Analyze the most recent run
                latest_run = runs.runs[0]
                print(f"\n🔍 Analyzing most recent run...")
                analyze_run(latest_run.run_id, client)
            else:
                print("\n📋 No recent runs found")
        
        except Exception as e:
            print(f"\n⚠️ Could not get recent runs: {e}")
            print("💡 Usage: analyze_run('your-run-id-here')")
    
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Make sure you have a working KFP client configuration") 