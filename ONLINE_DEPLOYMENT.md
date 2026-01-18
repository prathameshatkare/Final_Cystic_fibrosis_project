# CFVision Online Deployment Guide

This guide will help you deploy CFVision to the cloud so it's accessible from anywhere on the internet.

---

## Table of Contents

1. [Deployment Options Comparison](#deployment-options-comparison)
2. [Option 1: Render (Easiest - FREE)](#option-1-render-easiest---free)
3. [Option 2: Railway (Fast - FREE Tier)](#option-2-railway-fast---free-tier)
4. [Option 3: Heroku (Popular - Paid)](#option-3-heroku-popular---paid)
5. [Option 4: AWS EC2 (Full Control)](#option-4-aws-ec2-full-control)
6. [Option 5: DigitalOcean (Simple VPS)](#option-5-digitalocean-simple-vps)
7. [Custom Domain Setup](#custom-domain-setup)
8. [SSL Certificate (HTTPS)](#ssl-certificate-https)

---

## Deployment Options Comparison

| Platform | Cost | Setup Time | Best For |
|----------|------|------------|----------|
| **Render** | FREE | 10 min | Quick demos, testing |
| **Railway** | FREE (500h/mo) | 5 min | Development, staging |
| **Heroku** | $7/mo | 15 min | Production, reliability |
| **AWS EC2** | $3-20/mo | 30 min | Full control, scaling |
| **DigitalOcean** | $6/mo | 20 min | Simple VPS, predictable cost |

**Recommendation:** Start with **Render (FREE)** for testing, then move to **DigitalOcean** for production.

---

## Option 1: Render (Easiest - FREE)

**Best for:** Quick online demo, no credit card required.

### Step 1: Prepare Your Code

Create `render.yaml` in your project root:

```yaml
services:
  - type: web
    name: cfvision-backend
    env: python
    buildCommand: pip install -r requirements.txt && python experiments/baselines.py
    startCommand: python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
    disk:
      name: model-storage
      mountPath: /opt/render/project/src/models
      sizeGB: 1

  - type: web
    name: cfvision-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/dist
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
```

Create `requirements.txt` in project root:

```txt
torch==2.1.0
torchvision==0.16.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
scikit-learn==1.3.2
numpy==1.26.2
pydantic==2.5.0
python-multipart==0.0.6
onnxruntime==1.16.3
```

### Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit for Render deployment"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/cfvision.git
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to https://render.com and sign up (FREE)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml`
5. Click **"Create Web Service"**

**Your app will be live at:**
```
https://cfvision-backend.onrender.com
https://cfvision-frontend.onrender.com
```

### Step 4: Update Frontend API URL

Edit `frontend/src/App.tsx`:

```typescript
const API_URL = import.meta.env.PROD 
  ? 'https://cfvision-backend.onrender.com/api'
  : 'http://localhost:8000/api';
```

Commit and push:
```bash
git add frontend/src/App.tsx
git commit -m "Update API URL for production"
git push
```

Render will auto-redeploy!

---

## Option 2: Railway (Fast - FREE Tier)

**Best for:** Fastest deployment, 500 hours/month free.

### Step 1: Install Railway CLI

```powershell
# Install via npm
npm install -g @railway/cli

# Login
railway login
```

### Step 2: Create Railway Project

```bash
# Initialize
railway init

# Link to new project
railway link
```

### Step 3: Deploy Backend

Create `Procfile`:

```
web: python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Create `railway.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Step 4: Deploy

```bash
# Deploy backend
railway up

# Get your URL
railway domain
```

**Your backend will be live at:**
```
https://cfvision-production.up.railway.app
```

### Step 5: Deploy Frontend

```bash
cd frontend

# Build
npm run build

# Deploy (Railway will auto-detect Vite)
railway up
```

---

## Option 3: Heroku (Popular - Paid)

**Best for:** Production deployments, $7/month.

### Step 1: Install Heroku CLI

Download from https://devcenter.heroku.com/articles/heroku-cli

```powershell
# Verify installation
heroku --version

# Login
heroku login
```

### Step 2: Create Heroku Apps

```bash
# Create backend app
heroku create cfvision-backend

# Create frontend app
heroku create cfvision-frontend
```

### Step 3: Configure Backend

Create `Procfile`:

```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Create `runtime.txt`:

```
python-3.12.0
```

### Step 4: Deploy Backend

```bash
# Add buildpack for Python
heroku buildpacks:set heroku/python -a cfvision-backend

# Deploy
git push heroku main

# Open in browser
heroku open -a cfvision-backend
```

### Step 5: Deploy Frontend

For frontend, use Heroku's static buildpack:

```bash
cd frontend

# Create separate git repo
git init
git add .
git commit -m "Frontend deployment"

# Add buildpack
heroku buildpacks:set https://github.com/heroku/heroku-buildpack-static -a cfvision-frontend

# Create static.json
echo '{"root": "dist/"}' > static.json

# Deploy
git push heroku main
```

---

## Option 4: AWS EC2 (Full Control)

**Best for:** Production with full customization, ~$10/month.

### Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2
2. Click **"Launch Instance"**
3. Choose **Ubuntu 22.04 LTS** (Free tier eligible)
4. Instance type: **t2.micro** (1 vCPU, 1GB RAM)
5. Create key pair (save `.pem` file)
6. Security group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)
7. Launch instance

### Step 2: Connect to Instance

```powershell
# Convert .pem to .ppk (for PuTTY on Windows)
# Or use Windows Terminal:
ssh -i "cfvision-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
```

### Step 3: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip python3-venv nginx -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs -y
```

### Step 4: Deploy Application

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/cfvision.git
cd cfvision

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Train model (or copy pre-trained)
python experiments/baselines.py

# Install and build frontend
cd frontend
npm install
npm run build
cd ..
```

### Step 5: Configure Nginx

Create `/etc/nginx/sites-available/cfvision`:

```nginx
server {
    listen 80;
    server_name <YOUR_EC2_PUBLIC_IP>;

    # Frontend
    location / {
        root /home/ubuntu/cfvision/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/cfvision /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 6: Create Systemd Service

Create `/etc/systemd/system/cfvision.service`:

```ini
[Unit]
Description=CFVision FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/cfvision
Environment="PATH=/home/ubuntu/cfvision/venv/bin"
ExecStart=/home/ubuntu/cfvision/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable cfvision
sudo systemctl start cfvision
sudo systemctl status cfvision
```

### Step 7: Access Your Application

```
http://<YOUR_EC2_PUBLIC_IP>
```

---

## Option 5: DigitalOcean (Simple VPS)

**Best for:** Simple setup, predictable pricing, $6/month.

### Step 1: Create Droplet

1. Go to https://www.digitalocean.com
2. Click **"Create"** → **"Droplets"**
3. Choose **Ubuntu 22.04 LTS**
4. Plan: **Basic** ($6/mo - 1GB RAM, 1 vCPU)
5. Add SSH key or password
6. Click **"Create Droplet"**

### Step 2: Deploy Using Docker (Easiest)

**Create `Dockerfile`:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Train model
RUN python experiments/baselines.py

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Create `docker-compose.yml`:**

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
    environment:
      - ENV=production

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend/dist:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    restart: always
    depends_on:
      - backend
```

**Create `nginx.conf`:**

```nginx
server {
    listen 80;
    server_name _;

    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Step 3: Deploy to Droplet

```bash
# SSH into droplet
ssh root@<YOUR_DROPLET_IP>

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose -y

# Clone your repository
git clone https://github.com/YOUR_USERNAME/cfvision.git
cd cfvision

# Build frontend
cd frontend
npm install
npm run build
cd ..

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

**Access your app at:**
```
http://<YOUR_DROPLET_IP>
```

---

## Custom Domain Setup

Once deployed, connect a custom domain (e.g., `cfvision.yourdomain.com`):

### Step 1: Update DNS Records

Add these records to your domain registrar:

```
Type    Name        Value
A       @           <YOUR_SERVER_IP>
A       www         <YOUR_SERVER_IP>
```

### Step 2: Update Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/cfvision
```

Change:
```nginx
server_name <YOUR_EC2_PUBLIC_IP>;
```

To:
```nginx
server_name cfvision.yourdomain.com www.cfvision.yourdomain.com;
```

Restart Nginx:
```bash
sudo systemctl restart nginx
```

---

## SSL Certificate (HTTPS)

### Using Let's Encrypt (FREE)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d cfvision.yourdomain.com -d www.cfvision.yourdomain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

Certbot will automatically:
- Obtain SSL certificate
- Configure Nginx for HTTPS
- Set up auto-renewal

**Your app is now accessible at:**
```
https://cfvision.yourdomain.com
```

---

## Environment Variables for Production

Create `.env` file (DO NOT commit to Git):

```env
# Production settings
ENV=production
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=cfvision.yourdomain.com,www.cfvision.yourdomain.com

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/cfvision

# API Keys (if needed)
CFTR2_API_KEY=your-key-here
```

Update `api/main.py` to use environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Example usage
SECRET_KEY = os.getenv("SECRET_KEY", "default-dev-key")
ENV = os.getenv("ENV", "development")
```

---

## Monitoring & Maintenance

### Check Application Status

```bash
# Check systemd service
sudo systemctl status cfvision

# Check Docker containers
docker-compose ps

# View logs
docker-compose logs -f

# Or for systemd:
sudo journalctl -u cfvision -f
```

### Update Deployment

```bash
# Pull latest code
git pull origin main

# Rebuild (if using Docker)
docker-compose down
docker-compose up -d --build

# Or restart systemd service
sudo systemctl restart cfvision
```

### Backup

```bash
# Backup model and data
tar -czf cfvision-backup-$(date +%Y%m%d).tar.gz models/ data/

# Transfer to local machine
scp ubuntu@<SERVER_IP>:/home/ubuntu/cfvision-backup-*.tar.gz .
```

---

## Cost Comparison (Monthly)

| Platform | Free Tier | Production Cost | Scaling |
|----------|-----------|-----------------|---------|
| Render | ✅ Yes | $7 | Auto |
| Railway | ✅ 500h | $5-20 | Auto |
| Heroku | ❌ No | $7 | Manual |
| AWS EC2 | ✅ 12mo | $3-20 | Manual |
| DigitalOcean | ❌ No | $6 | Manual |

---

## Quick Start Recommendation

### For Testing/Demo:
```bash
# Use Render (FREE, no credit card)
# Just push to GitHub and connect!
```

### For Production:
```bash
# Use DigitalOcean + Docker
# $6/month, simple, reliable
```

---

## Troubleshooting

### Issue 1: Port already in use
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

### Issue 2: Model file too large for Git
```bash
# Use Git LFS
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

Or train model on server instead:
```bash
python experiments/baselines.py --epochs 20
```

### Issue 3: Out of memory on small instances
```bash
# Use swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue 4: Frontend can't connect to backend
Update `frontend/src/App.tsx`:
```typescript
const API_URL = window.location.origin + '/api';
```

---

## Security Checklist

- [ ] Enable HTTPS with SSL certificate
- [ ] Set strong SECRET_KEY in environment variables
- [ ] Enable firewall (UFW on Ubuntu)
- [ ] Disable root SSH login
- [ ] Set up automatic security updates
- [ ] Enable rate limiting on API
- [ ] Add authentication for sensitive endpoints
- [ ] Regular backups of model and data

---

## Next Steps

1. **Choose deployment platform** (Render for quick test, DigitalOcean for production)
2. **Push code to GitHub**
3. **Follow platform-specific steps above**
4. **Test the deployed application**
5. **Set up custom domain** (optional)
6. **Enable HTTPS** with Let's Encrypt
7. **Monitor performance** and logs

Your CFVision application will be accessible from anywhere in the world! 🌍

For questions or issues, refer to the platform-specific documentation or create an issue in your GitHub repository.
