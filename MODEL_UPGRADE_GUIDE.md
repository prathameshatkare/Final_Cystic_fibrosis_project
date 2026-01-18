# CFVision-FL: Model Upgrade & Data Evolution Guide

This guide provides a comprehensive technical roadmap for evolving your Cystic Fibrosis diagnosis model as new clinical data becomes available.

---

## Table of Contents
1. [Scenario 1: Increasing Sample Size (More Patients)](#scenario-1-increasing-sample-size)
2. [Scenario 2: Expanding Feature Scope (New Symptoms/Columns)](#scenario-2-expanding-feature-scope)
3. [Scenario 3: Privacy-Preserving Federated Upgrades](#scenario-3-federated-upgrades)
4. [Scenario 4: Genetic Database Updates (CFTR2)](#scenario-4-genetic-updates)
5. [Deployment & Syncing (Edge Devices)](#deployment--syncing)
6. [Best Practices for Data Quality](#best-practices)

---

## Scenario 1: Increasing Sample Size
**Use Case:** You received 5,000 new patient records that use the *exact same* columns as your current dataset.

### Step 1: Data Integration
Append the new data to your existing master CSV file.
```python
import pandas as pd

# Load existing and new data
old_df = pd.read_csv("data/synthetic_cystic_fibrosis_dataset.csv")
new_df = pd.read_csv("data/new_clinical_records.csv")

# Combine and save
combined_df = pd.concat([old_df, new_df], ignore_index=True)
combined_df.to_csv("data/synthetic_cystic_fibrosis_dataset.csv", index=False)
```

### Step 2: Retraining the Baseline
Run the centralized training script. The script automatically detects the increased sample size and recalculates the normalization parameters.
```bash
python experiments/baselines.py --epochs 25 --batch_size 64
```

---

## Scenario 2: Expanding Feature Scope
**Use Case:** Medical research identifies a new biomarker (e.g., "Lung Clearance Index") that you want to add to the model.

### Step 1: Update Data Preparation
Modify `experiments/baselines.py` or `api/main.py` if the new feature requires specific preprocessing (like custom scaling or one-hot encoding).

### Step 2: Update Model Architecture
If you add a new column, the `input_dim` of the neural network must change. 
1. Open `models/cf_tabular.py`.
2. The `input_dim` is passed during initialization, so ensure your training script calculates it correctly:
```python
# In training script
input_dim = X_train.shape[1] 
model = CFTabularNet(input_dim=input_dim)
```

### Step 3: Update the UI Form
To allow doctors to enter the new data in the "Diagnose" tab:
1. Open `api/main.py`.
2. Locate the `groups` dictionary in the `get_metadata` function.
3. Add your new feature to the appropriate clinical group:
```python
groups = {
    "Clinical Tests": ["sweat_test_simulated", "new_biomarker_name"]
}
```

---

## Scenario 3: Privacy-Preserving Federated Upgrades
**Use Case:** A new hospital wants to contribute data but cannot share the raw CSV due to HIPAA/GDPR.

1. **Deploy a Node:** Provide the hospital with the `federated/client.py` script.
2. **Local Preprocessing:** The hospital ensures their local CSV matches your feature names.
3. **Run Simulation/Training:**
   * Start your central server.
   * The hospital runs: `python federated/client.py --data_path local_data.csv`
4. **Aggregate:** The global model weights will update to reflect the "knowledge" of the new hospital without any data leaving their premises.

---

## Scenario 4: Genetic Database Updates
**Use Case:** The CFTR2 project releases a new `.xlsx` report with newly discovered mutations.

### Step 1: Parsing the New Data
Since the system uses `mutations.json` for speed, you must rebuild this file whenever you get new CFTR2 Excel files.

### Step 2: Verify `mutations.json`
Ensure the new mutations are categorized into:
* `CF-causing`
* `Varying clinical consequence`
* `Non CF-causing`

The API will automatically pick up these changes and adjust the "Risk Weighting" in the diagnosis logic.

---

## Deployment & Syncing

### 1. Update the Live Cloud App
After retraining, you must push the new model weights to Render/GitHub:
```bash
git add models/cf_tabular_central.pt
git commit -m "Upgrade: Model updated with new clinical data (N=15,000)"
git push
```

### 2. Syncing Edge Devices (Raspberry Pi/Tablets)
If you have deployed offline edge devices, you **must** regenerate the ONNX file. The ONNX file contains both the new weights and the new normalization logic.
```bash
python export_to_onnx.py
```
**Action:** Replace the `cf_tabular_edge.onnx` and `edge_config.json` files on the Raspberry Pi/Tablet with the newly generated ones.

---

## Best Practices
1. **Versioning:** Save old versions of the model (e.g., `model_v1.pt`, `model_v2.pt`) before overwriting.
2. **Evaluation:** Always compare the **F1-Score** of the new model against the old one. If the F1-Score drops, the new data might be noisy or incorrectly labeled.
3. **Data Balancing:** CF is a rare disease. When adding new data, try to keep the prevalence of CF cases around 10-15% to avoid the "Accuracy Paradox" where the model ignores rare cases.
4. **Audit Trail:** Keep a log of where the new data came from (Hospital A, Clinic B, etc.) for clinical accountability.

---

**Your CFVision system is designed to grow. By following this guide, you ensure that your AI remains at the cutting edge of clinical research.**
