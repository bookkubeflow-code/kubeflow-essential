# monitor_runs.py
# Script to monitor your Kubeflow pipeline runs in real-time

import time
import json
from datetime import datetime
from pipeline_runner import PipelineRunner

def monitor_recent_runs():
    """Monitor and display recent pipeline runs."""
    print("🔍 Initializing Pipeline Monitoring...")
    print("=" * 50)
    
    try:
        # Initialize with custom authentication
        runner = PipelineRunner(use_custom_auth=True)
        print("✅ Connected successfully!")
        print()
        
        # Get recent runs
        print("📋 Recent Pipeline Runs:")
        print("-" * 30)
        recent_runs = runner.list_runs(limit=10)
        
        if not recent_runs:
            print("   No runs found")
            return
        
        # Display runs with status
        for i, run in enumerate(recent_runs, 1):
            status_emoji = {
                'Running': '🔄',
                'Succeeded': '✅', 
                'Failed': '❌',
                'Pending': '⏳',
                'Skipped': '⏭️',
                'Unknown': '❓'
            }.get(run.get('status', 'Unknown'), '❓')
            
            print(f"   {i:2d}. {status_emoji} {run['name']}")
            print(f"       Status: {run['status']}")
            print(f"       Created: {run.get('created_at', 'Unknown')}")
            print(f"       Run ID: {run['run_id']}")
            print()
        
        return recent_runs
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def monitor_specific_run(run_id):
    """Monitor a specific run in detail."""
    print(f"🔍 Monitoring Run: {run_id}")
    print("=" * 50)
    
    try:
        runner = PipelineRunner(use_custom_auth=True)
        
        # Get detailed status
        status = runner.get_run_status(run_id)
        
        if 'error' in status:
            print(f"❌ Error getting run status: {status['error']}")
            return
        
        print(f"📊 Run Details:")
        print(f"   Name: {status.get('name', 'Unknown')}")
        print(f"   Status: {status.get('status', 'Unknown')}")
        print(f"   Created: {status.get('created_at', 'Unknown')}")
        print(f"   Finished: {status.get('finished_at', 'Not finished')}")
        
        if status.get('error'):
            print(f"   Error: {status['error']}")
        
        return status
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def interactive_monitor():
    """Interactive monitoring session."""
    print("🎯 Interactive Pipeline Monitor")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. 📋 List recent runs")
        print("2. 🔍 Monitor specific run")
        print("3. 🔄 Real-time monitoring (every 30s)")
        print("4. 🚪 Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            print()
            runs = monitor_recent_runs()
            
        elif choice == '2':
            run_id = input("\nEnter Run ID: ").strip()
            if run_id:
                print()
                monitor_specific_run(run_id)
            
        elif choice == '3':
            print("\n🔄 Starting real-time monitoring (Ctrl+C to stop)...")
            try:
                while True:
                    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Checking runs...")
                    runs = monitor_recent_runs()
                    
                    # Show any running pipelines
                    running = [r for r in runs if r.get('status') == 'Running']
                    if running:
                        print(f"🔄 {len(running)} pipeline(s) currently running")
                    
                    time.sleep(30)  # Check every 30 seconds
                    
            except KeyboardInterrupt:
                print("\n⏹️ Monitoring stopped")
                
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option")

def quick_status():
    """Quick status check - perfect for demonstrating monitoring."""
    print("⚡ Quick Status Check")
    print("=" * 30)
    
    try:
        runner = PipelineRunner(use_custom_auth=True) 
        runs = runner.list_runs(limit=5)
        
        total_runs = len(runs)
        running = len([r for r in runs if r.get('status') == 'Running'])
        succeeded = len([r for r in runs if r.get('status') == 'Succeeded'])
        failed = len([r for r in runs if r.get('status') == 'Failed'])
        
        print(f"📊 Pipeline Summary:")
        print(f"   Total runs: {total_runs}")
        print(f"   🔄 Running: {running}")
        print(f"   ✅ Succeeded: {succeeded}")
        print(f"   ❌ Failed: {failed}")
        print()
        
        if running > 0:
            print("🔄 Currently Running:")
            for run in runs:
                if run.get('status') == 'Running':
                    print(f"   • {run['name']}")
        else:
            print("💤 No pipelines currently running")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Kubeflow Pipeline Monitor")
    print("Using custom Dex authentication")
    print()
    
    # Show quick status first
    quick_status()
    print()
    
    # Ask user what they want to do
    print("What would you like to do?")
    print("1. 👁️ Quick status check (done above)")
    print("2. 📋 List all recent runs") 
    print("3. 🎯 Interactive monitoring")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '2':
        print()
        monitor_recent_runs()
    elif choice == '3':
        interactive_monitor()
    else:
        print("\n✅ Quick status completed! Use options 2 or 3 for more features.")
    
    print("\n💡 To monitor a running pipeline:")
    print("   1. Start a run in Kubeflow UI: http://localhost:8080")
    print("   2. Copy the Run ID from the URL")
    print("   3. Run: python monitor_runs.py")
    print("   4. Choose option 2 and paste the Run ID")
    print("\n🎯 This demonstrates the real power of Option 2!") 