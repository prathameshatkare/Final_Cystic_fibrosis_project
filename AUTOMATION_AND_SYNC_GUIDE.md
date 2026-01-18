# CFVision-FL: Data Automation & Edge-to-Cloud Sync Guide

This guide explains the automated data lifecycle of the CFVision project, from patient diagnosis on a remote edge device to global model updates in the cloud.

---

## 1. The Autonomous Data Loop

The system is designed as a "Closed-Loop" clinical AI. It follows these 4 stages:

1.  **Ingestion**: Edge devices diagnose patients and automatically upload the clinical data.
2.  **Aggregation**: The cloud backend validates and stores the new data in the master dataset.
3.  **Evolution**: The developer runs the automation script to retrain the AI on the newly gathered real-world data.
4.  **Deployment**: The updated "smarter" model is auto-deployed back to the cloud and edge network.

---

## 2. Edge-to-Cloud Synchronization

Every edge device (Raspberry Pi, tablet, or laptop) running the `edge_inference.py` script now acts as a data probe.

### How it Works:
*   **Trigger**: When `predictor.predict(patient_data)` is called.
*   **Process**: 
    1.  The local AI performs the diagnosis (works offline).
    2.  If an internet connection is detected, the script creates a JSON packet containing the symptoms and the AI's diagnosis.
    3.  The packet is sent to `https://final-cystic-fibrosis-project-1.onrender.com/api/data/ingest`.
*   **Resilience**: If the device is in a remote area with no internet, the sync is skipped silently to prevent interrupting the clinician's workflow.

---

## 3. Backend Data Management

The FastAPI backend now features a dedicated clinical data receiver.

### Key Components:
*   **Endpoint**: `/api/data/ingest`
*   **Security**: Validates that incoming data matches the required clinical columns.
*   **Storage**: Appends the new record to `data/synthetic_cystic_fibrosis_dataset.csv`.
*   **Real-time Feedback**: Automatically clears the dashboard cache so the "Total Samples" and "Disease Prevalence" charts update instantly for all users.

---

## 4. The One-Command Upgrade Pipeline

To update the global model with the latest data gathered from clinics, use the `automate_upgrade.py` script.

### Command:
```bash
python automate_upgrade.py
```

### What this command does:
1.  **Model Training**: Uses your local CPU/GPU to retrain the AI on the updated CSV file (including the new data from edge devices).
2.  **Edge Optimization**: Generates a new `cf_tabular_edge.onnx` file.
3.  **Auto-Push**: Commits the new weights and pushes them to GitHub.
4.  **Cloud Trigger**: GitHub notifies Render, which automatically restarts your backend with the smarter model.

---

## 5. Cloud Integrity (GitHub Actions)

We have implemented a "Cloud Guardian" workflow in `.github/workflows/model_check.yml`.

Every time the automation script pushes a new model:
1.  GitHub starts a clean virtual machine.
2.  It loads your new model.
3.  It performs a **Sanity Check** to ensure the model is not corrupted.
4.  It ensures the input dimensions (36 features) match the API requirements.

---

## 6. Summary of Automation Files

| File | Role |
| :--- | :--- |
| `api/main.py` | Receives and stores data from clinics via the `/api/data/ingest` endpoint. |
| `edge_inference.py` | Automatically uploads patient data to the cloud after every diagnosis. |
| `automate_upgrade.py` | Performs local retraining and triggers global cloud deployment. |
| `model_check.yml` | Validates model health in the cloud before it goes live. |

---

**By combining Edge Inference with Cloud Automation, CFVision-FL ensures that the diagnostic accuracy improves every time a doctor uses the system, while keeping patient records safe and structured.**
