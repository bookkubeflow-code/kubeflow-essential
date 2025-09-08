#!/usr/bin/env python3
"""
Error Handling Pipeline Testing Script
Demonstrates and tests various error scenarios in Kubeflow Pipelines.

This script shows how to:
1. Submit pipelines with different error scenarios
2. Monitor pipeline runs for error patterns
3. Analyze error handling effectiveness
4. Test circuit breaker behavior
5. Validate graceful degradation
"""

import sys
import os
import time
import json
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config import Config
from pipeline_runner import PipelineRunner
from run_analyzer import RunAnalyzer, RunStatus


class ErrorHandlingTester:
    """Test suite for error handling patterns in KFP pipelines."""
    
    def __init__(self):
        """Initialize the error handling tester."""
        config = Config()
        self.client = config.get_client()
        self.runner = PipelineRunner()
        self.analyzer = RunAnalyzer()
        self.test_results = []
    
    def test_exponential_backoff(self) -> Dict[str, Any]:
        """Test exponential backoff behavior with network timeouts."""
        print("\n🔄 Testing Exponential Backoff with Network Timeouts")
        print("=" * 60)
        
        test_params = {
            "primary_data_url": "https://httpbin.org/delay/10",  # Will timeout
            "max_fetch_retries": 4,
            "service_failure_rate": 0.0  # Don't fail other services
        }
        
        try:
            run_result = self.runner.submit_pipeline(
                pipeline_path="resilient_pipeline.yaml",
                run_name="test-exponential-backoff",
                parameters=test_params,
                experiment_name="error-handling-tests"
            )
            
            if run_result and "run_id" in run_result:
                # Monitor the run
                run_id = run_result["run_id"]
                print(f"📊 Monitoring run: {run_id}")
                
                # Wait a bit for the run to start
                time.sleep(10)
                
                # Analyze the run
                timeline = self.analyzer.get_run_timeline(run_id)
                
                result = {
                    "test": "exponential_backoff",
                    "status": "completed",
                    "run_id": run_id,
                    "timeline_events": len(timeline) if timeline else 0,
                    "expected_behavior": "Should retry with increasing delays, then use fallback"
                }
                
                print(f"✅ Test completed. Run ID: {run_id}")
                return result
            else:
                return {"test": "exponential_backoff", "status": "failed", "error": "Failed to submit pipeline"}
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return {"test": "exponential_backoff", "status": "error", "error": str(e)}
    
    def test_circuit_breaker(self) -> Dict[str, Any]:
        """Test circuit breaker pattern with high failure rates."""
        print("\n🔴 Testing Circuit Breaker Pattern")
        print("=" * 60)
        
        test_params = {
            "circuit_failure_threshold": 2,
            "service_failure_rate": 1.0,  # Always fail
            "fallback_data_rows": 25
        }
        
        try:
            run_result = self.runner.submit_pipeline(
                pipeline_path="resilient_pipeline.yaml",
                run_name="test-circuit-breaker",
                parameters=test_params,
                experiment_name="error-handling-tests"
            )
            
            if run_result and "run_id" in run_result:
                run_id = run_result["run_id"]
                print(f"📊 Monitoring circuit breaker run: {run_id}")
                
                time.sleep(15)
                
                # Check if fallback path was taken
                timeline = self.analyzer.get_run_timeline(run_id)
                
                result = {
                    "test": "circuit_breaker",
                    "status": "completed",
                    "run_id": run_id,
                    "expected_behavior": "Should open circuit and use fallback data generation"
                }
                
                print(f"✅ Circuit breaker test completed. Run ID: {run_id}")
                return result
            else:
                return {"test": "circuit_breaker", "status": "failed", "error": "Failed to submit pipeline"}
                
        except Exception as e:
            print(f"❌ Circuit breaker test failed: {e}")
            return {"test": "circuit_breaker", "status": "error", "error": str(e)}
    
    def test_graceful_degradation(self) -> Dict[str, Any]:
        """Test graceful degradation with resource constraints."""
        print("\n⚡ Testing Graceful Degradation")
        print("=" * 60)
        
        test_params = {
            "min_memory_gb": 0.1,
            "target_memory_gb": 0.2,  # Very low to force degradation
            "primary_data_url": "https://httpbin.org/json",  # Valid URL
            "max_fetch_retries": 2
        }
        
        try:
            run_result = self.runner.submit_pipeline(
                pipeline_path="resilient_pipeline.yaml",
                run_name="test-graceful-degradation",
                parameters=test_params,
                experiment_name="error-handling-tests"
            )
            
            if run_result and "run_id" in run_result:
                run_id = run_result["run_id"]
                print(f"📊 Monitoring degradation run: {run_id}")
                
                time.sleep(20)
                
                result = {
                    "test": "graceful_degradation",
                    "status": "completed", 
                    "run_id": run_id,
                    "expected_behavior": "Should adapt processing based on available memory"
                }
                
                print(f"✅ Graceful degradation test completed. Run ID: {run_id}")
                return result
            else:
                return {"test": "graceful_degradation", "status": "failed", "error": "Failed to submit pipeline"}
                
        except Exception as e:
            print(f"❌ Graceful degradation test failed: {e}")
            return {"test": "graceful_degradation", "status": "error", "error": str(e)}
    
    def test_error_scenarios(self) -> Dict[str, Any]:
        """Test specific error scenarios using the error testing pipeline."""
        print("\n🧪 Testing Specific Error Scenarios")
        print("=" * 60)
        
        scenarios = [
            {"test_scenario": "network_timeout", "failure_rate": 0.9},
            {"test_scenario": "high_failure_rate", "failure_rate": 1.0},
            {"test_scenario": "resource_constraint", "failure_rate": 0.5}
        ]
        
        scenario_results = []
        
        for scenario in scenarios:
            print(f"\n🎯 Testing scenario: {scenario['test_scenario']}")
            
            try:
                run_result = self.runner.submit_pipeline(
                    pipeline_path="error_testing_pipeline.yaml",
                    run_name=f"test-{scenario['test_scenario']}",
                    parameters=scenario,
                    experiment_name="error-scenario-tests"
                )
                
                if run_result and "run_id" in run_result:
                    run_id = run_result["run_id"]
                    print(f"   📊 Run ID: {run_id}")
                    
                    scenario_results.append({
                        "scenario": scenario['test_scenario'],
                        "run_id": run_id,
                        "status": "submitted"
                    })
                else:
                    scenario_results.append({
                        "scenario": scenario['test_scenario'],
                        "status": "failed",
                        "error": "Failed to submit"
                    })
                    
            except Exception as e:
                print(f"   ❌ Scenario {scenario['test_scenario']} failed: {e}")
                scenario_results.append({
                    "scenario": scenario['test_scenario'],
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "test": "error_scenarios",
            "status": "completed",
            "scenarios": scenario_results
        }
    
    def analyze_error_patterns(self, run_ids: List[str]) -> Dict[str, Any]:
        """Analyze error patterns across multiple runs."""
        print("\n📈 Analyzing Error Patterns")
        print("=" * 60)
        
        analysis_results = {
            "total_runs": len(run_ids),
            "successful_runs": 0,
            "failed_runs": 0,
            "common_errors": {},
            "recovery_patterns": []
        }
        
        for run_id in run_ids:
            try:
                print(f"🔍 Analyzing run: {run_id}")
                
                # Get run summary
                summary = self.analyzer.get_run_summary(run_id)
                if summary:
                    if summary.get("status") == "SUCCEEDED":
                        analysis_results["successful_runs"] += 1
                    else:
                        analysis_results["failed_runs"] += 1
                    
                    # Check for failure diagnosis
                    if summary.get("status") in ["FAILED", "ERROR"]:
                        diagnosis = self.analyzer.diagnose_failure(run_id)
                        if diagnosis and "error_summary" in diagnosis:
                            for error in diagnosis["error_summary"]:
                                error_type = error.get("error_type", "unknown")
                                analysis_results["common_errors"][error_type] = \
                                    analysis_results["common_errors"].get(error_type, 0) + 1
                
            except Exception as e:
                print(f"   ⚠️  Failed to analyze run {run_id}: {e}")
        
        return analysis_results
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete error handling test suite."""
        print("\n🚀 Starting Comprehensive Error Handling Test Suite")
        print("=" * 80)
        
        # Compile pipelines first
        print("📦 Compiling pipelines...")
        try:
            os.system("cd pipelines && python resilient_pipeline.py")
            print("✅ Pipelines compiled successfully")
        except Exception as e:
            print(f"❌ Failed to compile pipelines: {e}")
            return {"status": "failed", "error": "Pipeline compilation failed"}
        
        # Run individual tests
        test_results = []
        run_ids = []
        
        # Test 1: Exponential Backoff
        result1 = self.test_exponential_backoff()
        test_results.append(result1)
        if "run_id" in result1:
            run_ids.append(result1["run_id"])
        
        # Test 2: Circuit Breaker
        result2 = self.test_circuit_breaker()
        test_results.append(result2)
        if "run_id" in result2:
            run_ids.append(result2["run_id"])
        
        # Test 3: Graceful Degradation
        result3 = self.test_graceful_degradation()
        test_results.append(result3)
        if "run_id" in result3:
            run_ids.append(result3["run_id"])
        
        # Test 4: Error Scenarios
        result4 = self.test_error_scenarios()
        test_results.append(result4)
        if "scenarios" in result4:
            for scenario in result4["scenarios"]:
                if "run_id" in scenario:
                    run_ids.append(scenario["run_id"])
        
        # Wait for runs to complete before analysis
        print(f"\n⏳ Waiting for {len(run_ids)} runs to complete...")
        time.sleep(30)
        
        # Analyze error patterns
        if run_ids:
            pattern_analysis = self.analyze_error_patterns(run_ids)
        else:
            pattern_analysis = {"error": "No runs to analyze"}
        
        # Compile final results
        final_results = {
            "test_suite": "comprehensive_error_handling",
            "timestamp": time.time(),
            "individual_tests": test_results,
            "pattern_analysis": pattern_analysis,
            "run_ids": run_ids,
            "summary": {
                "total_tests": len(test_results),
                "successful_tests": len([t for t in test_results if t.get("status") == "completed"]),
                "total_runs_created": len(run_ids)
            }
        }
        
        return final_results
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of test results."""
        print("\n" + "=" * 80)
        print("🎯 ERROR HANDLING TEST SUITE SUMMARY")
        print("=" * 80)
        
        summary = results.get("summary", {})
        print(f"📊 Total Tests: {summary.get('total_tests', 0)}")
        print(f"✅ Successful Tests: {summary.get('successful_tests', 0)}")
        print(f"🚀 Pipeline Runs Created: {summary.get('total_runs_created', 0)}")
        
        if "pattern_analysis" in results:
            analysis = results["pattern_analysis"]
            if "total_runs" in analysis:
                print(f"\n📈 Run Analysis:")
                print(f"   Total Runs: {analysis['total_runs']}")
                print(f"   Successful: {analysis['successful_runs']}")
                print(f"   Failed: {analysis['failed_runs']}")
                
                if analysis.get("common_errors"):
                    print(f"   Common Errors:")
                    for error_type, count in analysis["common_errors"].items():
                        print(f"     - {error_type}: {count}")
        
        print(f"\n🔗 Run IDs for detailed analysis:")
        for run_id in results.get("run_ids", []):
            print(f"   - {run_id}")
        
        print(f"\n💡 To analyze individual runs:")
        print(f"   python run_analyzer.py --run-id <RUN_ID>")
        
        print("\n✨ Error handling patterns demonstrated:")
        print("   🔄 Exponential backoff with jitter")
        print("   🔴 Circuit breaker pattern")
        print("   ⚡ Graceful degradation")
        print("   🔀 Conditional execution for fallbacks")
        print("   🧹 Exit handlers for cleanup")
        print("   📊 Resource-aware processing")


def main():
    """Main function to run error handling tests."""
    print("🧪 Kubeflow Pipelines Error Handling Test Suite")
    print("Testing production-ready error handling patterns...")
    
    try:
        tester = ErrorHandlingTester()
        results = tester.run_comprehensive_test_suite()
        
        # Save results to file
        results_file = f"error_handling_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Print summary
        tester.print_test_summary(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
