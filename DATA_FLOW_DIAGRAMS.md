# CFVision-FL Data Flow Diagrams

## Overview
The CFVision-FL system implements a sophisticated data flow architecture that enables secure and efficient processing of clinical data across distributed edge devices and centralized cloud infrastructure. This document details the data flow at multiple levels of abstraction.

## Level 0: Context Diagram

```
                    ┌─────────────────────────────────────┐
                    │        CFVision-FL System         │
                    │                                   │
                    │  ┌─────────────────────────────┐  │
                    │  │    Federated Learning     │  │
                    │  │        Framework          │  │
                    │  └─────────────────────────────┘  │
                    │                                   │
                    │  ┌─────────────────────────────┐  │
                    │  │     Model Management      │  │
                    │  │      & Training           │  │
                    │  └─────────────────────────────┘  │
                    │                                   │
                    │  ┌─────────────────────────────┐  │
                    │  │     Dashboard & API       │  │
                    │  │       Services            │  │
                    │  └─────────────────────────────┘  │
                    └─────────────────────────────────────┘
                              │   │   │
                              │   │   │
                              ▼   ▼   ▼
        Clinicians     Edge Devices   Data Scientists
           │               │             │
           │               │             │
           └───────────────┼─────────────┘
                           │
                    Clinical Data &
                 Model Updates Exchange
```

## Level 1: Primary Data Flows

### 1. Clinical Data Ingestion Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Patient       │───▶│  Edge Device   │───▶│  Cloud Backend  │
│   Clinical      │    │  Preprocessing │    │  Data Ingestion │
│   Data          │    │  & Validation │    │  & Storage      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ 1. Symptom Inputs     │ 2. Normalize &        │ 3. Validate & 
         │                       │    Encode             │    Append to CSV
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │  Local Model    │              │
         │              │  Inference      │              │
         │              │  (ONNX)         │              │
         │              └─────────────────┘              │
         │                       │                       │
         │ 4. Risk Score         │ 5. Auto-sync when     │ 6. Update 
         │                       │    connected          │    Dashboard Cache
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Synthetic      │
                    │  Dataset        │
                    │  (Master CSV)   │
                    └─────────────────┘
```

### 2. Model Training & Distribution Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Master CSV     │───▶│  Training      │───▶│  Model          │
│  (Clinical     │    │  Pipeline      │    │  Optimization   │
│  Data)         │    │  (baselines.py) │    │  (ONNX Export) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ 1. Load Data          │ 2. Train Central      │ 3. Convert to ONNX
         │                       │    Model              │    & Config JSON
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │  Central Model  │───▶│  Edge Model     │
         │              │  (PyTorch)      │    │  Package        │
         │              │  (cf_central.pt)│    │  (ONNX + JSON) │
         │              └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ 4. Save Central       │ 5. Deploy to          │ 6. Distribute to
         │    Model              │    Cloud              │    Edge Devices
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  GitHub Repo    │
                    │  (Version Ctrl) │
                    └─────────────────┘
```

### 3. Real-time Inference Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Clinical       │───▶│  Preprocessing  │───▶│  Neural        │
│  Input Form     │    │  Pipeline       │    │  Network       │
│  (Web/Dashboard)│    │  (Standardizer  │    │  (Forward Pass)│
└─────────────────┘    │  & Encoder)     │    └─────────────────┘
         │              └─────────────────┘            │
         │ 1. Submit    │ 2. Apply       │            │ 3. Compute
         │    Patient   │    Transformations│         │    Probabilities
         │    Data      │                 │            │
         │              ▼                 ▼            ▼
         │      ┌─────────────────┐    ┌─────────────────┐
         │      │  Feature Vector │───▶│  Logits        │
         │      │  (36-dim)      │    │  (2-dim)       │
         │      └─────────────────┘    └─────────────────┘
         │              │                       │
         │ 4. Standardized│ 5. Raw Network     │ 6. Apply Softmax
         │    Features   │    Outputs          │    & Genetic Adjust
         │              │                       │
         └──────────────┼───────────────────────┘
                        │
               ┌─────────────────┐
               │  Final Output   │
               │  (Probability,  │
               │   Risk Level)   │
               └─────────────────┘
```

## Level 2: Detailed Process Flows

### A. Edge Device Data Sync Process

```
Start
  │
  ├─ Receive new clinical data from local diagnosis
  │
  ├─ Validate data format against schema
  │
  ├─ Check internet connectivity
  │
  ├─ Prepare JSON payload with:
  │  ├─ Clinical features (normalized)
  │  ├─ Diagnosis result
  │  └─ Timestamp
  │
  ├─ Send POST request to /api/data/ingest
  │
  ├─ Wait for response confirmation
  │
  ├─ Log sync status (success/failure)
  │
  └─ End
```

### B. Federated Learning Simulation Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Simulated      │───▶│  Client         │───▶│  Parameter      │
│  Hospital      │    │  Training       │    │  Aggregation   │
│  (Partitioned   │    │  (Local Fit)   │    │  (Server)      │
│  Data)         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ 1. Partition Data     │ 2. Local Training    │ 3. Collect Updates
         │    by Hospital        │    on Client         │    from Clients
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │  Local Model    │───▶│  Global Model  │
         │              │  Updates        │    │  Update        │
         │              │  (Δweights)     │    │  (FedAvg)      │
         │              └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ 4. Compute Local      │ 5. Send to Server    │ 6. Average &
         │    Gradients          │    (Flower Protocol) │    Distribute
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Flower         │
                    │  Framework      │
                    │  Orchestration  │
                    └─────────────────┘
```

## Data Transformation Points

### Input Preprocessing
```
Raw Clinical Data → Standardization → One-Hot Encoding → Feature Vector (36-dim)
```

### Genetic Adjustment
```
Base Probability → Mutation Type Lookup → Risk Adjustment → Final Probability
```

### Output Post-processing
```
Logits → Softmax → Probability → Risk Category → Clinical Interpretation
```

## Security & Privacy Data Flow

```
Clinical Data → Anonymization → Encryption → Transmission → Decryption → Processing
      ↑                                                        ↓
   Local Device                                            Cloud Server
      │                                                        │
   → No Raw Data                                           → Access Controls
      │    Leaves Device                                    → Audit Logs
      ↓                                                        ↓
   → Encrypted                                         → Secure Storage
      │    Transmitted                                    → Data Retention
      │                                                   → Policies
```

This comprehensive data flow ensures that clinical information is processed securely and efficiently while maintaining the privacy of patient data through federated learning principles.