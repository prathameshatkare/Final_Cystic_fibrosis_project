# CFVision-FL System Architecture

## Overview
The CFVision-FL project implements a distributed federated learning system for Cystic Fibrosis diagnosis with edge computing capabilities. The architecture is designed to maintain privacy while enabling collaborative model training across multiple clinical sites.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Cloud Layer                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│ │   Render        │  │  FastAPI         │  │  Model Storage              │ │
│ │   Backend       │  │  Application     │  │  (PyTorch/ONNX)             │ │
│ │                 │  │                  │  │                             │ │
│ │  ┌───────────┐  │  │  ┌─────────────┐ │  │  ┌────────────────────────┐ │ │
│ │  │Web Server │  │  │  │API Endpoints│ │  │  │cf_tabular_central.pt   │ │ │
│ │  │(Uvicorn)  │  │  │  │/predict     │ │  │  │cf_tabular_edge.onnx    │ │ │
│ │  └───────────┘  │  │  │/data/ingest  │ │  │  │edge_config.json        │ │ │
│ │                 │  │  │/dashboard    │ │  │  └────────────────────────┘ │ │
│ │  ┌───────────┐  │  │  └─────────────┘ │  │                             │ │
│ │  │Database   │  │  │                  │  │  ┌────────────────────────┐ │ │
│ │  │(CSV File) │  │  │                  │  │  │Synthetic Clinical      │ │ │
│ │  │           │  │  │                  │  │  │Dataset                 │ │ │
│ │  └───────────┘  │  │                  │  │  └────────────────────────┘ │ │
│ └─────────────────┘  └──────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP Requests
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Edge Layer                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│ │   Raspberry Pi  │  │  Mobile (Termux) │  │  Laptop/PC                │ │
│ │   Edge Node     │  │  Edge Node       │  │  Edge Node                │ │
│ │                 │  │                  │  │                             │ │
│ │  ┌───────────┐  │  │  ┌─────────────┐ │  │  ┌──────────────────────┐ │ │
│ │  │Edge Inference│ │  │  │Edge Inference│ │  │  │Edge Inference      │ │ │
│ │  │Engine       │  │  │  │Engine       │ │  │  │Engine               │ │ │
│ │  │(ONNXRuntime)│  │  │  │(ONNXRuntime)│ │  │  │(ONNXRuntime)        │ │ │
│ │  └───────────┘  │  │  └─────────────┘ │  │  └──────────────────────┘ │ │
│ │                 │  │                  │  │                             │ │
│ │  ┌───────────┐  │  │  ┌─────────────┐ │  │  ┌──────────────────────┐ │ │
│ │  │Local Model│  │  │  │Local Model  │ │  │  │Local Model          │ │ │
│ │  │(ONNX)     │  │  │  │(ONNX)       │ │  │  │(ONNX)               │ │ │
│ │  └───────────┘  │  │  └─────────────┘ │  │  └──────────────────────┘ │ │
│ └─────────────────┘  └──────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Edge Layer
- **Edge Computing Nodes**: Raspberry Pi, mobile devices (via Termux), and PCs that perform local inference
- **Edge Inference Engine**: Lightweight ONNX Runtime for executing the neural network model
- **Auto-Sync Capability**: Automatic data transmission to cloud backend when connectivity is available
- **Local Model Storage**: ONNX model files optimized for edge deployment

### 2. Cloud Layer
- **Render Backend Service**: Hosts the FastAPI application serving as the central coordination point
- **API Gateway**: Manages all HTTP requests to various endpoints
- **Model Management**: Stores and manages both central and edge-optimized models
- **Data Persistence**: CSV-based storage for clinical data and model metrics
- **Dashboard Service**: Provides visualization and monitoring capabilities

### 3. Training Infrastructure
- **Centralized Training Pipeline**: Executes training scripts to update the global model
- **Federated Learning Simulation**: Implements Flower framework for distributed training
- **Model Evaluation**: Computes performance metrics across simulated hospital sites

### 4. Frontend Layer
- **Web Dashboard**: React-based interface for visualization and diagnostics
- **Responsive Design**: Optimized for both desktop and mobile access
- **Real-time Metrics**: Displays federated learning performance indicators

## Data Flow Architecture

```
Clinical Data Sources → Edge Preprocessing → Local Inference → Cloud Aggregation
        ↓                    ↓                    ↓                   ↓
Patient Records    →   StandardScaler    →   Neural Network  →   Master Dataset
Genetic Data       →   One-Hot Encoding  →   Probability     →   Model Training
Symptom Reports    →   Normalization     →   Risk Score      →   Performance Metrics
```

## Security & Privacy Considerations
- Data anonymization at the edge
- Secure transmission via HTTPS
- Federated learning preserves data locality
- Differential privacy mechanisms