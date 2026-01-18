import os
import sys
import subprocess
import pandas as pd
import torch
import json
from datetime import datetime

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_cystic_fibrosis_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cf_tabular_central.pt")
ONNX_PATH = os.path.join(PROJECT_ROOT, "models", "cf_tabular_edge.onnx")

def run_command(command, description):
    print(f"\n[STEP] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}:")
        print(e.stderr)
        return False

def automate_upgrade(new_data_path=None):
    print("="*60)
    print(f"CFVision-FL: Automated Model Upgrade Pipeline")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. Data Integration
    if new_data_path and os.path.exists(new_data_path):
        print(f"\n[STEP] Integrating new data from: {new_data_path}")
        old_df = pd.read_csv(DATA_PATH)
        new_df = pd.read_csv(new_data_path)
        
        # Simple validation: check if columns match
        if set(new_df.columns) != set(old_df.columns):
            print("ERROR: New data columns do not match existing dataset columns!")
            print(f"Expected: {list(old_df.columns)}")
            print(f"Received: {list(new_df.columns)}")
            return

        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        combined_df.to_csv(DATA_PATH, index=False)
        print(f"SUCCESS: Dataset updated. Total samples: {len(combined_df)}")
    else:
        print("\n[INFO] No new data path provided. Refreshing existing model.")

    # 2. Retrain Centralized Baseline
    if not run_command("python experiments/baselines.py --epochs 25", "Retraining Model"):
        return

    # 3. Export to ONNX for Edge Devices
    if not run_command("python export_to_onnx.py", "Exporting to ONNX"):
        return

    # 4. Git Synchronization (The "Auto-Deploy" Part)
    print("\n[STEP] Synchronizing with Cloud (GitHub/Render)...")
    commit_msg = f"Auto-upgrade: Model updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Run git commands
    git_success = (
        run_command("git add .", "Staging changes") and
        run_command(f'git commit -m "{commit_msg}"', "Committing updates") and
        run_command("git push", "Pushing to Cloud")
    )

    # 5. Final Summary
    print("\n" + "="*60)
    if git_success:
        print("UPGRADE & DEPLOYMENT COMPLETE")
        print("Your live dashboard will update automatically in 2-3 minutes.")
    else:
        print("UPGRADE COMPLETE (Local Only)")
        print("Git synchronization failed. Please push manually.")
    print("="*60)
    print(f"Main Model: {MODEL_PATH}")
    print(f"Edge Model: {ONNX_PATH}")
    print("="*60)

if __name__ == "__main__":
    # If a path is passed as an argument, use it as the new data source
    path = sys.argv[1] if len(sys.argv) > 1 else None
    automate_upgrade(path)
