# CFVision-FL UML Diagrams

## Overview
The CFVision-FL system implements object-oriented design patterns to support federated learning, edge inference, and cloud-based model management. This document presents the Unified Modeling Language (UML) diagrams for the system architecture.

## Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Neural Network Models                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                              CFTabularNet                                 │ │
│ │                                                                             │ │
│ │ - input_dim: int                                                            │ │
│ │ - net: torch.nn.Sequential                                                  │ │
│ │                                                                             │ │
│ │ + __init__(input_dim: int)                                                  │ │
│ │ + forward(x: torch.Tensor): torch.Tensor                                   │ │
│ │ + get_parameters(): List[torch.Tensor]                                     │ │
│ │ + set_parameters(parameters: List[torch.Tensor])                           │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ inherits
                                          │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Federated Learning                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                                CFClient                                   │ │
│ │                                                                             │ │
│ │ - cid: str                                                                  │ │
│ │ - train_loader: DataLoader                                                  │ │
│ │ - val_loader: DataLoader                                                    │ │
│ │ - model: CFTabularNet                                                       │ │
│ │ - device: torch.device                                                      │ │
│ │ - epochs: int                                                               │ │
│ │ - learning_rate: float                                                      │ │
│ │                                                                             │ │
│ │ + __init__(cid: str, train_loader: DataLoader, ...)                        │ │
│ │ + get_parameters(config: Dict[str, Any]): NDArraysAndInt                    │ │
│ │ + set_parameters(parameters: NDArraysAndInt) -> None                        │ │
│ │ + fit(config: Dict[str, Any]): Tuple[NDArraysAndInt, int, Dict[str, Any]]  │ │
│ │ + evaluate(config: Dict[str, Any]): Tuple[float, int, Dict[str, Any]]      │ │
│ │ + train_one_epoch() -> float                                               │ │
│ │ + validate() -> Dict[str, float]                                           │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ implements
                                          │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Flower Client Interface                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                              flwr.client.NumPyClient                       │ │
│ │                                                                             │ │
│ │ + get_parameters(config: Dict[str, Any]) -> NDArraysAndInt                 │ │
│ │ + set_parameters(parameters: NDArraysAndInt) -> None                       │ │
│ │ + fit(config: Dict[str, Any]) -> Tuple[NDArraysAndInt, int, Dict[str, Any]]│ │
│ │ + evaluate(config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]    │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Edge Inference                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                           CFEdgePredictor                                 │ │
│ │                                                                             │ │
│ │ - model_path: str                                                           │ │
│ │ - config_path: str                                                          │ │
│ │ - session: onnxruntime.InferenceSession                                     │ │
│ │ - input_name: str                                                           │ │
│ │ - scaler_mean: np.ndarray                                                   │ │
│ │ - scaler_scale: np.ndarray                                                  │ │
│ │ - feature_columns: List[str]                                                │ │
│ │ - cat_cols: List[str]                                                       │ │
│ │ - num_cols: List[str]                                                       │ │
│ │                                                                             │ │
│ │ + __init__(model_path: str, config_path: str)                              │ │
│ │ + load_config() -> Dict[str, Any]                                          │ │
│ │ + preprocess(patient_data: Dict[str, Any]) -> np.ndarray                   │ │
│ │ + predict(patient_data: Dict[str, Any]) -> Dict[str, Any]                  │ │
│ │ + manual_one_hot_encode(value: str, col: str) -> np.ndarray                │ │
│ │ + apply_z_score_scaling(values: np.ndarray, col_idx: int) -> np.ndarray    │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API Services                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                                APIService                                 │ │
│ │                                                                             │ │
│ │ + predict_api(data: Dict[str, Any]) -> Dict[str, Any]                      │ │
│ │ + ingest_edge_data(data: Dict[str, Any]) -> Dict[str, Any]                 │ │
│ │ + get_dashboard_metrics() -> Dict[str, Any]                                │ │
│ │ + get_metadata() -> Dict[str, Any]                                         │ │
│ │ + index() -> HTMLResponse                                                  │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Processing                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                           DataProcessor                                   │ │
│ │                                                                             │ │
│ │ - data_path: str                                                            │ │
│ │ - scaler: StandardScaler                                                    │ │
│ │ - dummy_columns: List[str]                                                  │ │
│ │ - cat_cols: List[str]                                                       │ │
│ │ - num_cols: List[str]                                                       │ │
│ │                                                                             │ │
│ │ + load_data() -> pd.DataFrame                                              │ │
│ │ + get_preprocessing_info() -> Tuple[StandardScaler, List[str], ...]        │ │
│ │ + prepare_features(df: pd.DataFrame) -> pd.DataFrame                       │ │
│ │ + get_feature_columns() -> List[str]                                       │ │
│ │ + create_dummy_variables(df: pd.DataFrame) -> pd.DataFrame                 │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Sequence Diagram: Edge Device Diagnosis

```
Actor: Clinician
Object: Edge Device
Object: CFEdgePredictor
Object: ONNX Runtime Session
Object: Cloud Backend
Object: API Service

Clinician->Edge Device: Enter patient clinical data
Edge Device->CFEdgePredictor: Initialize predictor
CFEdgePredictor->ONNX Runtime Session: Load model and config
Edge Device->CFEdgePredictor: Call predict(patient_data)
CFEdgePredictor->CFEdgePredictor: preprocess(patient_data)
CFEdgePredictor->CFEdgePredictor: Apply standardization & encoding
CFEdgePredictor->ONNX Runtime Session: Run inference
ONNX Runtime Session->CFEdgePredictor: Return logits
CFEdgePredictor->CFEdgePredictor: Apply softmax & genetic adjustment
CFEdgePredictor->Edge Device: Return diagnosis result
Edge Device->Cloud Backend: POST /api/data/ingest (auto-sync)
Cloud Backend->API Service: Process ingestion request
API Service->API Service: Validate and append to dataset
API Service->Cloud Backend: Return success response
Cloud Backend->Edge Device: Confirm sync
Edge Device->Clinician: Display diagnosis and sync status
```

## Sequence Diagram: Federated Learning Training

```
Actor: Research Coordinator
Object: Flower Server
Object: CFClient (Hospital 1)
Object: CFClient (Hospital 2)
Object: CFClient (Hospital 3)
Object: CFClient (Hospital 4)
Object: CFClient (Hospital 5)

Research Coordinator->Flower Server: Start federated training
Flower Server->CFClient (Hospital 1): Initialize parameters
Flower Server->CFClient (Hospital 2): Initialize parameters
Flower Server->CFClient (Hospital 3): Initialize parameters
Flower Server->CFClient (Hospital 4): Initialize parameters
Flower Server->CFClient (Hospital 5): Initialize parameters

alt For each federated round
    Flower Server->CFClient (Hospital 1): Request parameters
    CFClient (Hospital 1)->Flower Server: Return local parameters
    
    Flower Server->CFClient (Hospital 2): Request parameters
    CFClient (Hospital 2)->Flower Server: Return local parameters
    
    Flower Server->CFClient (Hospital 3): Request parameters
    CFClient (Hospital 3)->Flower Server: Return local parameters
    
    Flower Server->CFClient (Hospital 4): Request parameters
    CFClient (Hospital 4)->Flower Server: Return local parameters
    
    Flower Server->CFClient (Hospital 5): Request parameters
    CFClient (Hospital 5)->Flower Server: Return local parameters
    
    Flower Server->Flower Server: Aggregate parameters (FedAvg)
    Flower Server->CFClient (Hospital 1): Distribute global parameters
    Flower Server->CFClient (Hospital 2): Distribute global parameters
    Flower Server->CFClient (Hospital 3): Distribute global parameters
    Flower Server->CFClient (Hospital 4): Distribute global parameters
    Flower Server->CFClient (Hospital 5): Distribute global parameters
end

Flower Server->Research Coordinator: Training completed
```

## Activity Diagram: Model Training Process

```
Start: Initialize training parameters
│
├─> Load clinical dataset from CSV
│
├─> Split data into train/test sets (80/20)
│
├─> Apply preprocessing (scaling, encoding)
│
├─> Initialize CFTabularNet model
│
├─> Configure optimizer (AdamW) and loss function
│
├─> Loop: For each epoch (1 to N)
│   │
│   ├─> Perform forward pass
│   │
│   ├─> Calculate loss (cross-entropy)
│   │
│   ├─> Backpropagate gradients
│   │
│   ├─> Update model parameters
│   │
│   ├─> Evaluate on validation set
│   │
│   └─> Log metrics (accuracy, AUC, F1)
│
├─> Save trained model to file
│
├─> Export model to ONNX format
│
├─> Generate model performance report
│
└─> End: Training completed
```

## State Diagram: Edge Device Operation

```
┌─────────────────┐
│   Powered Off   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Initializing  │
│  (Loading Model)│
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Idle State    │
│ (Waiting for    │
│  Input)         │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Processing     │
│  Clinical Data  │
└─────────────────┘
         │
         ├─ Valid Input ──▶ ┌─────────────────┐
         │                  │   Running       │
         │                  │   Inference     │
         │                  └─────────────────┘
         │                           │
         │                  ┌─────────────────┐
         │                  │  Post-processing│
         │                  │  (Genetic Adj)  │
         │                  └─────────────────┘
         │                           │
         │                  ┌─────────────────┐
         │                  │  Results Ready  │
         │                  └─────────────────┘
         │                           │
         │                  ┌─────────────────┐
         │                  │  Sync Attempt   │
         │                  │  (if online)    │
         │                  └─────────────────┘
         │                           │
         │                  ┌─────────────────┐
         │                  │  Sync Success/  │
         │                  │  Failure        │
         │                  └─────────────────┘
         │                           │
         └───────────────────────────┘
         │
    Invalid Input
         │
         ▼
┌─────────────────┐
│   Error State   │
│  (Display Msg)  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Idle State    │
└─────────────────┘
```

## Component Diagram: System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CFVision-FL System                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   Edge Layer    │    │  Cloud Layer    │    │   Training Infrastructure   │ │
│  │                 │    │                 │    │                             │ │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────────────────┐ │ │
│  │ │Edge Device  │ │    │ │FastAPI App  │ │<<import>>│ │Training Pipeline        │ │ │
│  │ │(Raspberry Pi│ │────│ │             │ │<─────────│ │(baselines.py)         │ │ │
│  │ │/Mobile/PC)  │ │    │ │             │ │    │ └─────────────────────────┘ │ │
│  │ └─────────────┘ │    │ └─────────────┘ │    │                             │ │
│  │                 │    │                 │    │ ┌─────────────────────────┐ │ │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │<<import>>│ │Federated Learning      │ │ │
│  │ │ONNX Runtime │ │    │ │Model Storage│ │<─────────│ │Simulation             │ │ │
│  │ │             │ │────│ │             │ │    │ └─────────────────────────┘ │ │
│  │ └─────────────┘ │    │ └─────────────┘ │    │                             │ │
│  │                 │    │                 │    │ ┌─────────────────────────┐ │ │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │<<import>>│ │Evaluation Framework    │ │ │
│  │ │Local Model  │ │    │ │Data Storage │ │<─────────│ │(eval.py)              │ │ │
│  │ │(ONNX)       │ │────│ │(CSV)        │ │    │ └─────────────────────────┘ │ │
│  │ └─────────────┘ │    │ └─────────────┘ │    │                             │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

These UML diagrams provide a comprehensive view of the CFVision-FL system architecture, showing the relationships between classes, the flow of data and operations, and the system's behavioral patterns during various operations such as diagnosis, training, and federated learning.