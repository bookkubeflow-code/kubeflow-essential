#!/usr/bin/env python3
"""
Simple pipeline upload script - Backup method when main client times out
"""

import os
import sys
from pathlib import Path

def upload_via_ui_instructions():
    """Provide clear manual upload instructions"""
    print("📋 Manual Pipeline Upload Instructions")
    print("=" * 50)
    print("The automated upload is experiencing timeouts.")
    print("Please use the manual method which always works:")
    print()
    print("1. 🌐 Open browser: http://localhost:8080")
    print("2. 🔑 Login with:")
    print("   Username: user@example.com")
    print("   Password: 12341234")
    print()
    print("3. 📁 Navigate to: Pipelines → Upload Pipeline")
    print()
    print("4. 📤 Upload this file:")
    
    # Find the latest pipeline file
    pipeline_files = [
        "compiled_pipelines/ml_training_latest.yaml",
        "ml_pipeline.yaml"
    ]
    
    for pipeline_file in pipeline_files:
        if Path(pipeline_file).exists():
            full_path = os.path.abspath(pipeline_file)
            file_size = Path(pipeline_file).stat().st_size
            print(f"   📄 File: {pipeline_file}")
            print(f"   📍 Full path: {full_path}")
            print(f"   📊 Size: {file_size} bytes")
            break
    else:
        print("   ❌ No compiled pipeline found. Run 'python compile_and_run.py' first.")
        return False
    
    print()
    print("5. ✍️  Set pipeline name: ml-training-pipeline")
    print("6. 📝 Add description: ML training pipeline for Iris dataset")
    print("7. ✅ Click 'Create'")
    print()
    print("8. 🚀 After upload, click 'Create Run' to execute:")
    print("   - Run name: iris-training-run")
    print("   - Experiment: Create new or use existing")
    print("   - Parameters: dataset_url (use default)")
    print()
    print("🎯 Your pipeline will then execute in Kubeflow!")
    return True

def check_file_size():
    """Check if pipeline file might be too large"""
    pipeline_file = "compiled_pipelines/ml_training_latest.yaml"
    if Path(pipeline_file).exists():
        size = Path(pipeline_file).stat().st_size
        print(f"📊 Pipeline file size: {size} bytes ({size/1024:.1f} KB)")
        
        if size > 50000:  # 50KB
            print("⚠️  File is relatively large, which might cause upload timeouts")
        else:
            print("✅ File size is reasonable for upload")
        
        return size
    return 0

def suggest_alternatives():
    """Suggest alternative upload methods"""
    print("\n🔧 Alternative Upload Methods:")
    print("=" * 40)
    
    print("\n1. 🌐 Manual UI Upload (Recommended)")
    print("   - Always works")
    print("   - Full feature support")
    print("   - Easy to use")
    
    print("\n2. 🔄 Retry Automated Upload")
    print("   - Restart Kubeflow services:")
    print("   kubectl rollout restart deployment ml-pipeline -n kubeflow")
    print("   kubectl rollout restart deployment ml-pipeline-ui -n kubeflow")
    
    print("\n3. 📋 kubectl Direct (Advanced)")
    print("   - For experts only")
    print("   - Bypasses UI entirely")
    
    print("\n💡 Tip: The manual UI method is actually preferred by many teams")
    print("   because it provides better visibility and control!")

if __name__ == "__main__":
    print("🚀 Pipeline Upload Helper")
    print("=" * 30)
    
    # Check file
    size = check_file_size()
    if size == 0:
        print("❌ No pipeline file found. Run 'python compile_and_run.py' first.")
        sys.exit(1)
    
    # Provide instructions
    if upload_via_ui_instructions():
        suggest_alternatives()
        print("\n🎉 Ready to upload! Use the manual method for guaranteed success.")
    else:
        print("❌ Setup incomplete. Please compile your pipeline first.")
        sys.exit(1) 