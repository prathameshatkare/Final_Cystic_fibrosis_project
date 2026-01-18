# CFVision-FL: Mobile Edge Implementation Guide

This guide explains how to transform a standard Android smartphone into a high-performance clinical edge node, functioning identically to the Raspberry Pi workstations in the CFVision network.

---

## 1. Prerequisites
*   **Device**: Android Smartphone.
*   **App**: [Termux](https://f-droid.org/en/packages/com.termux/) (Download from F-Droid for the most up-to-date version).
*   **Storage**: 100MB of free space.

---

## 2. Environment Setup (The "Bedside" Setup)

Open **Termux** on your mobile phone and execute these commands sequentially:

```bash
# Update the internal package manager
pkg update && pkg upgrade -y

# Install Python and essential build tools
pkg install python -y

# Install the Clinical AI engines (Lightweight versions)
pip install numpy onnxruntime requests
```

---

## 3. Deployment of AI Assets

Transfer the following files from your PC to your phone's `Downloads` folder (via USB, Drive, or Telegram):

1.  `edge_inference.py` (The logic)
2.  `cf_tabular_edge.onnx` (The trained model)
3.  `edge_config.json` (The standardization parameters)

**In Termux, link your phone's storage:**
```bash
termux-setup-storage
# Navigate to the folder (assuming it's in Downloads/CFVision)
cd ~/storage/downloads/CFVision
```

---

## 4. Running the Clinical Diagnosis

To perform a patient diagnosis directly on the phone's CPU:

```bash
python edge_inference.py
```

### What happens inside the phone:
*   **Inference**: The `onnxruntime` engine calculates the CF risk in < 20ms.
*   **Offline Mode**: The diagnosis works perfectly even in "Airplane Mode."
*   **Auto-Sync**: If Wi-Fi/4G is active, the phone automatically transmits the data to:
    `https://final-cystic-fibrosis-project-1.onrender.com/api/data/ingest`

---

## 5. Integration Summary: Mobile vs. Raspberry Pi

| Feature | Mobile (Termux) | Raspberry Pi |
| :--- | :--- | :--- |
| **Logic** | Same `edge_inference.py` | Same `edge_inference.py` |
| **AI Engine** | `onnxruntime` | `onnxruntime` |
| **Data Sync** | Automatic (Wi-Fi/5G) | Automatic (Ethernet/Wi-Fi) |
| **Retraining** | Linked to Cloud CSV | Linked to Cloud CSV |

---

## 6. Pro-Tip: "One-Tap" Quick Launch

To make the phone behave like a dedicated medical device, you can create a "boot" script.

1.  In Termux, type: `nano ~/.bashrc`
2.  Add this line at the bottom: 
    `cd ~/storage/downloads/CFVision && python edge_inference.py`
3.  Save (Ctrl+O) and Exit (Ctrl+X).

**Result**: Every time you open Termux, the CFVision diagnostic tool will start immediately, ready for the next patient.

---

**This mobile node implementation ensures that your Federated Learning network can reach patients anywhere in the world, directly in the palm of a clinician's hand.**
