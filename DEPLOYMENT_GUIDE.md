# CFVision-FL Complete Deployment Guide

This guide covers all deployment scenarios for the CFVision-FL project, from local development to production edge devices and federated learning networks.

---

## Table of Contents

1. [Quick Start (Local Development)](#quick-start-local-development)
2. [Production Deployment Options](#production-deployment-options)
   - [Option A: Single-Site Web Application](#option-a-single-site-web-application)
   - [Option B: Edge Device Deployment](#option-b-edge-device-deployment)
   - [Option C: Multi-Site Federated Network](#option-c-multi-site-federated-network)
3. [Cloud Deployment](#cloud-deployment)
4. [Monitoring & Maintenance](#monitoring--maintenance)
5. [Security Hardening](#security-hardening)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start (Local Development)

### Prerequisites
- **Python:** 3.8 or higher
- **Node.js:** 18+ (for React frontend)
- **RAM:** 4GB minimum
- **Storage:** 2GB free space

### Step 1: Clone and Setup Backend

```bash
cd C:\Users\ASUS\Desktop\AP_CF_PAPER

# Install Python dependencies
pip install torch torchvision pandas scikit-learn fastapi uvicorn flwr numpy

# Generate synthetic data (if not already present)
python data/generate_data.py

# Train the baseline model
python experiments/baselines.py --epochs 20
```

**Expected Output:**
```
Epoch 20/20 | train_loss=0.1234 | acc=0.9520 | f1=0.7957
Model saved to models/cf_tabular_central.pt
```

### Step 2: Start Backend API

```bash
# Start FastAPI server
python api/main.py
```

**Access:**
- Backend UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Step 3: Start Frontend Dashboard

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**Access:**
- Dashboard: http://localhost:5173

---

## Production Deployment Options

### Option A: Single-Site Web Application

**Best for:** Hospitals deploying the system internally for their own staff.

#### Architecture
```
┌─────────────────┐
│  Nginx (HTTPS)  │ Port 443
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│React │  │FastAPI│
│(Vite)│  │Backend│
└──────┘  └───┬───┘
              │
          ┌───▼────┐
          │PyTorch │
          │ Model  │
          └────────┘
```

#### Step 1: Build Frontend for Production

```bash
cd frontend
npm run build
```

This creates a `dist/` folder with optimized static files.

#### Step 2: Configure Nginx

Create `/etc/nginx/sites-available/cfvision`:

```nginx
server {
    listen 443 ssl http2;
    server_name cfvision.hospital.local;

    ssl_certificate /etc/ssl/certs/cfvision.crt;
    ssl_certificate_key /etc/ssl/private/cfvision.key;

    # Frontend
    location / {
        root /var/www/cfvision/frontend/dist;
        try_files $uri /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/cfvision /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### Step 3: Run Backend as System Service

Create `/etc/systemd/system/cfvision-api.service`:

```ini
[Unit]
Description=CFVision FastAPI Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/cfvision
Environment="PATH=/opt/cfvision/venv/bin"
ExecStart=/opt/cfvision/venv/bin/python api/main.py

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable cfvision-api
sudo systemctl start cfvision-api
sudo systemctl status cfvision-api
```

#### Step 4: Deploy Files

```bash
# Create deployment directory
sudo mkdir -p /opt/cfvision
sudo chown www-data:www-data /opt/cfvision

# Copy project files
sudo cp -r . /opt/cfvision/

# Copy frontend build
sudo cp -r frontend/dist /var/www/cfvision/frontend/

# Create virtual environment
cd /opt/cfvision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Option B: Edge Device Deployment

**Best for:** Remote clinics, mobile health units, or offline diagnostics.

#### Supported Devices
- Raspberry Pi 3/4/5
- NVIDIA Jetson Nano/Orin
- Android tablets (via Termux)
- Windows tablets/laptops

#### Step 1: Export Model to ONNX

On your development machine:

```bash
python export_to_onnx.py
```

**Files Generated:**
- `models/cf_tabular_edge.onnx` (54 KB)
- `models/edge_config.json` (15 KB)

#### Step 2: Prepare Deployment Package

```bash
# Create deployment folder
mkdir edge_package
cp models/cf_tabular_edge.onnx edge_package/
cp models/edge_config.json edge_package/
cp edge_inference.py edge_package/
cp data/mutations.json edge_package/  # Optional: for genetic context
```

#### Step 3: Deploy to Raspberry Pi

**Transfer files:**
```bash
# Via SCP
scp -r edge_package pi@192.168.1.100:~/cfvision/

# Or via USB drive
# Copy edge_package/ to USB, then plug into Pi
```

**On Raspberry Pi:**
```bash
cd ~/cfvision/edge_package

# Install dependencies
pip3 install numpy onnxruntime

# Test inference
python3 edge_inference.py
```

**Expected Output:**
```
Edge Predictor Initialized with 36 features.
--- Diagnostic Result (Edge Inference) ---
CF Probability: 27.57%
Risk Level: Low
```

#### Step 4: Create Web Interface (Optional)

Create `web_server.py` on the Pi:

```python
from flask import Flask, request, jsonify, render_template
from edge_inference import CFEdgePredictor
import json

app = Flask(__name__)
predictor = CFEdgePredictor()

# Load mutations for genetic context
with open("mutations.json", "r") as f:
    mutations = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', mutations=mutations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = predictor.predict(data)
        
        # Apply genetic weighting if mutation provided
        mutation = data.get('mutation')
        if mutation:
            mutation_info = next((m for m in mutations if m["name"] == mutation), None)
            if mutation_info:
                cf_prob = result['cf_probability']
                det = mutation_info['determination']
                if det == "CF-causing":
                    cf_prob = min(0.99, cf_prob + 0.4)
                elif det == "Varying clinical consequence":
                    cf_prob = min(0.95, cf_prob + 0.2)
                result['cf_probability'] = cf_prob
                result['mutation_impact'] = det
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Install Flask:**
```bash
pip3 install flask
```

**Run server:**
```bash
python3 web_server.py
```

**Access from any device on same network:**
```
http://<raspberry_pi_ip>:5000
```

#### Step 5: Auto-Start on Boot

Create `/etc/systemd/system/cfvision-edge.service`:

```ini
[Unit]
Description=CFVision Edge Inference Server
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/cfvision/edge_package
ExecStart=/usr/bin/python3 web_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable cfvision-edge
sudo systemctl start cfvision-edge
```

---

### Option C: Multi-Site Federated Network

**Best for:** Collaborative research networks or hospital systems sharing model improvements without sharing patient data.

#### Architecture

```
                    ┌─────────────────┐
                    │ Central Server  │
                    │ (Flower Server) │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼──────┐  ┌──────▼─────┐  ┌──────▼─────┐
    │ Hospital A   │  │ Hospital B │  │ Hospital C │
    │ (FL Client)  │  │ (FL Client)│  │ (FL Client)│
    └──────────────┘  └────────────┘  └────────────┘
         Local             Local           Local
         Data              Data            Data
```

#### Step 1: Set Up Central Server

**On a cloud server (AWS/Azure/GCP):**

```bash
# Install Flower
pip install flwr

# Start Flower server
flwr-server --port 8080 --rounds 10
```

**Or use custom strategy:**

Create `server.py`:

```python
import flwr as fl
from typing import List, Tuple, Optional
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Configure strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.4,  # Sample 40% of clients per round
    fraction_evaluate=0.4,
    min_fit_clients=2,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

Run:
```bash
python server.py
```

#### Step 2: Configure Hospital Clients

**On each hospital's server/edge device:**

Create `client.py`:

```python
import flwr as fl
import torch
from models.cf_tabular import CFTabularNet
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load local hospital data (stays on-premise)
df = pd.read_csv("local_hospital_data.csv")
df = df.dropna(subset=["cf_diagnosis"])

# Preprocessing (same as training pipeline)
y = df["cf_diagnosis"].astype(int).values
feature_df = df.drop(columns=["cf_diagnosis", "age_at_diagnosis", "diagnostic_confidence"], errors="ignore")
feature_df = pd.get_dummies(feature_df)
scaler = StandardScaler()
X = scaler.fit_transform(feature_df.values.astype(float))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CFTabularNet(input_dim=X_train.shape[1]).to(device)

class CFClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]
    
    def fit(self, parameters, config):
        # Update model with global weights
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Train locally
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(5):  # 5 local epochs per round
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = torch.nn.CrossEntropyLoss()(outputs, yb)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Update model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss += torch.nn.CrossEntropyLoss()(outputs, yb).item()
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        
        return float(loss), len(val_loader.dataset), {"accuracy": correct/total}

# Connect to server
fl.client.start_client(
    server_address="<central_server_ip>:8080",
    client=CFClient().to_client()
)
```

#### Step 3: Start Federated Training

**On central server:**
```bash
python server.py
```

**On each hospital (in parallel):**
```bash
# Hospital A
python client.py

# Hospital B (on different machine)
python client.py

# Hospital C (on different machine)
python client.py
```

#### Step 4: Monitor Training Progress

The server will output:
```
[ROUND 1] fit progress: 2/5 clients
[ROUND 1] aggregate_fit: received 2 results
[ROUND 1] evaluate_metrics: accuracy 0.8234
[ROUND 2] fit progress: 2/5 clients
...
[ROUND 10] Final global accuracy: 0.9520
```

#### Step 5: Save Global Model

Add to `server.py`:

```python
def save_model_callback(server_round, parameters, metrics):
    if server_round == 10:  # After final round
        # Convert parameters to state_dict
        model = CFTabularNet(input_dim=36)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)
        
        # Save
        torch.save(model.state_dict(), f"global_model_round_{server_round}.pt")
        print(f"Global model saved!")

# Add to strategy
strategy = fl.server.strategy.FedAvg(
    # ... existing config ...
    on_fit_config_fn=save_model_callback,
)
```

---

## Cloud Deployment

### AWS Deployment

#### Step 1: Launch EC2 Instance
- **Instance Type:** t3.medium (2 vCPU, 4GB RAM)
- **OS:** Ubuntu 22.04 LTS
- **Storage:** 20GB SSD
- **Security Group:** Allow ports 80, 443, 8000

#### Step 2: Install Dependencies
```bash
# SSH into instance
ssh -i keypair.pem ubuntu@<ec2-public-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip nginx -y
pip3 install torch torchvision pandas scikit-learn fastapi uvicorn
```

#### Step 3: Deploy Application
```bash
# Clone or upload project
git clone <your-repo-url> /opt/cfvision
cd /opt/cfvision

# Train model
python experiments/baselines.py

# Configure Nginx (same as Option A)
sudo nano /etc/nginx/sites-available/cfvision

# Start backend
python api/main.py &
```

#### Step 4: Configure Domain (Optional)
- Point your domain to EC2 public IP
- Install Let's Encrypt SSL:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d cfvision.yourdomain.com
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Train model (or copy pre-trained)
RUN python experiments/baselines.py

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "api/main.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: always
  
  frontend:
    image: node:18
    working_dir: /app/frontend
    volumes:
      - ./frontend:/app/frontend
    ports:
      - "5173:5173"
    command: npm run dev
    restart: always
```

**Deploy:**
```bash
docker-compose up -d
```

---

## Monitoring & Maintenance

### Health Checks

Create `health_check.py`:

```python
import requests
import time
from datetime import datetime

def check_backend():
    try:
        response = requests.get("http://localhost:8000/api/dashboard-metrics", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_frontend():
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        return response.status_code == 200
    except:
        return False

while True:
    backend_status = "✓" if check_backend() else "✗"
    frontend_status = "✓" if check_frontend() else "✗"
    
    print(f"[{datetime.now()}] Backend: {backend_status} | Frontend: {frontend_status}")
    time.sleep(60)  # Check every minute
```

### Logging

Configure structured logging in `api/main.py`:

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/cfvision_{datetime.now().date()}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("CFVision")

# Log predictions
@app.post("/api/predict")
async def predict_api(data: Dict[str, Any]):
    logger.info(f"Prediction request received: {data}")
    result = # ... existing code
    logger.info(f"Prediction result: {result}")
    return result
```

### Performance Monitoring

Track inference time:

```python
import time

@app.post("/api/predict")
async def predict_api(data: Dict[str, Any]):
    start_time = time.time()
    
    # ... existing prediction code ...
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.3f}s")
    
    return {
        **result,
        "inference_time_ms": int(inference_time * 1000)
    }
```

---

## Security Hardening

### 1. API Authentication

Add JWT authentication:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()
SECRET_KEY = "your-secret-key-here"  # Use environment variable in production

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/predict")
async def predict_api(data: Dict[str, Any], user=Depends(verify_token)):
    # ... existing code
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/predict")
@limiter.limit("10/minute")  # Max 10 predictions per minute
async def predict_api(request: Request, data: Dict[str, Any]):
    # ... existing code
```

### 3. Input Validation

```python
from pydantic import BaseModel, Field

class PatientData(BaseModel):
    age_months: int = Field(ge=0, le=240)
    sweat_test_simulated: float = Field(ge=0, le=200)
    family_history_cf: int = Field(ge=0, le=1)
    # ... other fields

@app.post("/api/predict")
async def predict_api(data: PatientData):
    # Data is automatically validated
    result = predictor.predict(data.dict())
    return result
```

### 4. HTTPS Enforcement

In `api/main.py`:

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# Force HTTPS in production
if os.getenv("ENV") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
```

---

## Troubleshooting

### Issue 1: Backend fails to start

**Symptom:**
```
ModuleNotFoundError: No module named 'torch'
```

**Fix:**
```bash
pip install torch torchvision
```

### Issue 2: Model predictions always return 50%

**Cause:** Model file not found or corrupted.

**Fix:**
```bash
# Retrain model
python experiments/baselines.py --epochs 20

# Verify file exists
ls -lh models/cf_tabular_central.pt
```

### Issue 3: Frontend can't connect to backend

**Symptom:** CORS error in browser console.

**Fix:** Check CORS configuration in `api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue 4: Federated client can't connect to server

**Fix:**
```bash
# Check firewall on server
sudo ufw allow 8080

# Verify server is listening
netstat -tuln | grep 8080

# Test connection from client
telnet <server-ip> 8080
```

### Issue 5: Out of memory on edge device

**Fix:** Use quantized model:
```bash
# Export int8 version
python export_to_onnx.py --quantize
```

Or reduce batch size in `client.py`:
```python
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Train and validate model (F1 > 0.75)
- [ ] Test backend API locally
- [ ] Test frontend locally
- [ ] Export ONNX model for edge deployment
- [ ] Prepare deployment documentation

### Backend Deployment
- [ ] Install dependencies on target server
- [ ] Copy project files
- [ ] Configure environment variables
- [ ] Set up systemd service
- [ ] Configure Nginx/reverse proxy
- [ ] Enable HTTPS with SSL certificate
- [ ] Test API endpoints

### Frontend Deployment
- [ ] Build production bundle (`npm run build`)
- [ ] Copy dist/ to web server
- [ ] Configure routing (try_files for SPA)
- [ ] Test dashboard access
- [ ] Verify API integration

### Edge Deployment
- [ ] Export model to ONNX
- [ ] Test edge inference locally
- [ ] Transfer files to edge device
- [ ] Install minimal dependencies
- [ ] Test offline inference
- [ ] Configure auto-start on boot

### Federated Network
- [ ] Deploy central Flower server
- [ ] Configure client hospitals
- [ ] Test FL training round
- [ ] Set up model versioning
- [ ] Configure secure communication

### Monitoring
- [ ] Set up logging
- [ ] Configure health checks
- [ ] Monitor performance metrics
- [ ] Set up alerts for failures
- [ ] Document incident response

---

## Summary

You now have multiple deployment options:

1. **Development:** Local FastAPI + React dev servers
2. **Single-Site Production:** Nginx + systemd service
3. **Edge Devices:** ONNX Runtime on Raspberry Pi/tablets
4. **Federated Network:** Multi-site collaborative training
5. **Cloud:** AWS/Docker containerized deployment

Choose the option that best fits your use case and scale as needed.

For additional support, refer to:
- [EDGE_DEPLOYMENT_GUIDE.md](EDGE_DEPLOYMENT_GUIDE.md) - Detailed edge device instructions
- [HOW_EDGE_DEVICES_PROCESS.md](HOW_EDGE_DEVICES_PROCESS.md) - Technical deep dive
- [report.md](report.md) - Research paper and methodology

