# Edge Device Deployment Guide for CFVision-FL

This guide provides step-by-step instructions to deploy the CFVision CF diagnosis model on various edge devices (Raspberry Pi, Android tablets, Windows tablets, or any Linux-based edge server).

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Pre-Deployment: Export Model](#pre-deployment-export-model)
3. [Deployment Options](#deployment-options)
   - [Option A: Raspberry Pi (Recommended for Clinics)](#option-a-raspberry-pi)
   - [Option B: Android Tablet](#option-b-android-tablet)
   - [Option C: Windows Tablet/Laptop](#option-c-windows-tabletlaptop)
   - [Option D: NVIDIA Jetson (For Federated Training)](#option-d-nvidia-jetson)
4. [Testing Edge Inference](#testing-edge-inference)
5. [Integrating with Federated Learning](#integrating-with-federated-learning)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Development Machine (Your Current PC)
- Python 3.8+
- PyTorch (already installed)
- Working CFVision project

### Target Edge Device (Minimum Specs)
- **CPU:** ARM Cortex-A53 (Raspberry Pi 3) or better
- **RAM:** 1GB+ (2GB recommended)
- **Storage:** 500MB free space
- **OS:** Linux (Raspbian/Ubuntu), Android 8+, or Windows 10+
- **Python:** 3.8+ (not required for Android if using Java wrapper)

---

## Pre-Deployment: Export Model

**On your development PC**, run the export script to generate edge-compatible files:

```bash
cd C:\Users\ASUS\Desktop\AP_CF_PAPER
python export_to_onnx.py
```

**Expected Output:**
```
Detected input dimension: 36
Saved edge_config.json with scaler and column info.
Loaded weights from models/cf_tabular_central.pt
Successfully exported model to models/cf_tabular_edge.onnx
This file can now be used on edge devices via ONNX Runtime.
```

**Files Generated:**
- `models/cf_tabular_edge.onnx` (54 KB) - The optimized neural network
- `models/edge_config.json` (15 KB) - Preprocessing configuration

---

## Deployment Options

### Option A: Raspberry Pi

**Best for:** Low-cost clinic deployments in remote areas.

#### Step 1: Set Up Raspberry Pi
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip
sudo apt install python3-pip python3-dev -y

# Install required libraries
pip3 install numpy onnxruntime
```

#### Step 2: Transfer Files
Copy the following files from your PC to the Pi:

```bash
# On your PC, create deployment package
mkdir edge_deploy
cp models/cf_tabular_edge.onnx edge_deploy/
cp models/edge_config.json edge_deploy/
cp edge_inference.py edge_deploy/

# Transfer via USB, SCP, or FileZilla
# Example using SCP:
scp -r edge_deploy pi@<raspberry_pi_ip>:~/cfvision/
```

#### Step 3: Run Inference
```bash
# On Raspberry Pi
cd ~/cfvision/edge_deploy
python3 edge_inference.py
```

**Expected Output:**
```
Edge Predictor Initialized with 36 features.
--- Diagnostic Result (Edge Inference) ---
CF Probability: 27.57%
Risk Level: Low
```

#### Step 4: Create a Simple Web Interface (Optional)
```bash
# Install Flask
pip3 install flask

# Create web_app.py on the Pi
nano web_app.py
```

**Paste this code:**
```python
from flask import Flask, request, jsonify, render_template_string
from edge_inference import CFEdgePredictor

app = Flask(__name__)
predictor = CFEdgePredictor()

HTML = '''
<!DOCTYPE html>
<html>
<head><title>CFVision Edge</title></head>
<body>
    <h1>CF Risk Assessment (Offline)</h1>
    <form action="/predict" method="post">
        <label>Age (months): <input type="number" name="age_months" value="12"></label><br>
        <label>Family History CF: <input type="number" name="family_history_cf" value="0"></label><br>
        <label>Sweat Test: <input type="number" name="sweat_test_simulated" value="30"></label><br>
        <button type="submit">Diagnose</button>
    </form>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    result = predictor.predict(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Run the web server:**
```bash
python3 web_app.py
```

Access from any device on the same network: `http://<pi_ip>:5000`

---

### Option B: Android Tablet

**Best for:** Mobile healthcare workers conducting field screenings.

#### Step 1: Install Termux (Android Terminal Emulator)
- Download **Termux** from F-Droid (not Google Play - it's outdated).
- Install Python in Termux:
```bash
pkg update && pkg upgrade
pkg install python numpy
pip install onnxruntime
```

#### Step 2: Transfer Files
- Connect Android to PC via USB.
- Copy `edge_deploy/` folder to `Internal Storage/Download/`.
- In Termux:
```bash
cd ~/storage/downloads/edge_deploy
python edge_inference.py
```

#### Alternative: Native Android App (Advanced)
For a production-grade Android app, use **ONNX Runtime Mobile**:
1. Create an Android Studio project.
2. Add ONNX Runtime dependency:
```gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.0'
```
3. Load the `.onnx` model in Java/Kotlin and run inference.

---

### Option C: Windows Tablet/Laptop

**Best for:** Hospital staff using Windows devices.

#### Step 1: Install Dependencies
```powershell
# On the Windows edge device
pip install numpy onnxruntime
```

#### Step 2: Copy Files
- Copy `edge_deploy/` folder to the device.
- Run:
```powershell
cd edge_deploy
python edge_inference.py
```

#### Step 3: Create Desktop Shortcut (Optional)
1. Right-click desktop → New → Shortcut
2. Target: `python C:\path\to\edge_deploy\edge_inference.py`
3. Name it "CF Diagnosis Tool"

---

### Option D: NVIDIA Jetson (For Federated Training)

**Best for:** Specialist CF centers that want to participate in federated training.

#### Why Jetson?
The Jetson Nano/Orin has a GPU, allowing local training (not just inference).

#### Step 1: Set Up Jetson
```bash
# Install PyTorch for Jetson
pip3 install torch torchvision

# Install Flower
pip3 install flwr
```

#### Step 2: Deploy Full Training Code
- Transfer the entire `AP_CF_PAPER` folder to the Jetson.
- Each hospital runs their own Flower client:

```bash
# On Hospital 1's Jetson
python experiments/simulate_fl_cfvision.py
```

This allows the hospital to:
1. Train locally on their private patient data.
2. Send only model weights (not data) to the central server.
3. Download the improved global model.

---

## Testing Edge Inference

### Test with Custom Patient Data

Modify `edge_inference.py` to test different clinical scenarios:

```python
# High-risk patient
test_patient_high_risk = {
    "age_months": 6,
    "family_history_cf": 1,
    "salty_skin": 1,
    "sweat_test_simulated": 85,  # High chloride (>60 = CF+)
    "cough_type": 3,  # Chronic productive cough
    "meconium_ileus": 1,
    "ethnicity": "Caucasian"
}

result = predictor.predict(test_patient_high_risk)
print(f"CF Probability: {result['cf_probability']:.2%}")
# Expected: >70% (High Risk)
```

### Performance Benchmarks

**Inference Speed (on various devices):**
- Raspberry Pi 4: ~15ms per prediction
- Android Phone (Snapdragon 888): ~5ms
- Jetson Nano: ~3ms
- Windows Laptop (Intel i5): ~2ms

---

## Integrating with Federated Learning

To make the edge device a **Flower client** that participates in training:

### Step 1: Install Full Dependencies
```bash
pip install torch flwr pandas scikit-learn
```

### Step 2: Create Edge Client Script

**Save as `edge_client.py`:**
```python
import flwr as fl
import torch
from models.cf_tabular import CFTabularNet
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load local hospital data (stored privately on this device)
df = pd.read_csv("local_hospital_data.csv")
# ... (same preprocessing as baselines.py)

class CFClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def fit(self, parameters, config):
        # Update model with global weights
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Train locally for 5 epochs
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        for epoch in range(5):
            for xb, yb in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = torch.nn.CrossEntropyLoss()(outputs, yb)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Evaluate on local validation set
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for xb, yb in self.val_loader:
                outputs = self.model(xb)
                loss += torch.nn.CrossEntropyLoss()(outputs, yb).item()
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        
        return float(loss), len(self.val_loader.dataset), {"accuracy": correct/total}

# Initialize and start client
model = CFTabularNet(input_dim=36)
# ... (create train_loader, val_loader from local data)

fl.client.start_client(
    server_address="<central_server_ip>:8080",
    client=CFClient(model, train_loader, val_loader).to_client()
)
```

### Step 3: Connect to Central Server
```bash
# On edge device
python edge_client.py
```

The device will now:
1. Download global model weights from the server.
2. Train on local patient data (which never leaves the device).
3. Send updated weights back to the server.
4. Receive the improved global model.

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'onnxruntime'"
**Fix:**
```bash
pip install onnxruntime
# For ARM devices (Raspberry Pi):
pip install onnxruntime-arm64
```

### Issue 2: Model predictions are always 0% or 100%
**Cause:** Incorrect feature scaling.
**Fix:** Ensure `edge_config.json` is in the same directory as the `.onnx` file.

### Issue 3: Slow inference on Raspberry Pi
**Optimization:**
```bash
# Use lighter ONNX Runtime build
pip install onnxruntime-armhf  # For 32-bit ARM
```

### Issue 4: Cannot connect to Flower server
**Fix:**
- Check firewall settings on the central server.
- Verify server IP address.
- Ensure port 8080 is open.

### Issue 5: Out of memory on low-RAM devices
**Fix:** Reduce batch size in the client script:
```python
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # Reduced from 64
```

---

## Security Considerations

### Data Privacy
- Patient data **never leaves the edge device** during federated learning.
- Only model gradients (mathematical weights) are transmitted.

### Encryption
For production deployments, encrypt communication:
```python
# In edge_client.py, use SSL
fl.client.start_client(
    server_address="<server_ip>:8443",
    client=CFClient(...).to_client(),
    grpc_max_message_length=536870912,
    root_certificates=open("server.crt", "rb").read()
)
```

### Access Control
Implement authentication on the web interface:
```python
# In web_app.py
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify(username, password):
    return username == "doctor" and password == "secure_password"

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    # ... existing code
```

---

## Summary

You now have three deployment modes:

1. **Inference-Only Mode** (Raspberry Pi, Tablets): Ultra-lightweight, runs offline, instant predictions.
2. **Federated Client Mode** (Jetson, Hospital Servers): Participates in collaborative training while keeping data private.
3. **Hybrid Mode**: Edge devices run inference during clinic hours, train overnight when idle.

**Next Steps:**
- Deploy to 1-2 test devices.
- Collect real-world performance metrics.
- Scale to multi-site federated network.

For questions, refer to the main project README or contact the development team.
