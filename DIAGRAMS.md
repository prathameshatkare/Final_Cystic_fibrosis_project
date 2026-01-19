# AP_CF_PAPER: Project Diagrams

## 1. System Architecture Diagram

```mermaid
graph TB
    subgraph "Client Devices"
        A1[Mobile Device]
        A2[Raspberry Pi]
        A3[Edge Server]
    end
    
    subgraph "Central Infrastructure"
        B1[Federated Server]
        B2[Model Registry]
        B3[Security Layer]
    end
    
    subgraph "Frontend Interface"
        C1[Web Dashboard]
        C2[Mobile UI]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B1 --> B3
    C1 --> B1
    C2 --> B1
```

## 2. Federated Learning Workflow Diagram

```mermaid
graph TD
    A[Initialize Global Model] --> B[Send Model to Clients]
    B --> C{Clients Train Locally}
    C --> D[Aggregate Model Updates]
    D --> E{Convergence Check}
    E -->|No| F[Update Global Model]
    F --> B
    E -->|Yes| G[Final Model]
```

## 3. Client-Server Communication Diagram

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant Security
    
    Client->>Server: Request Global Model
    Server->>Client: Send Encrypted Model
    Client->>Client: Local Training
    Client->>Security: Encrypt Weights
    Security->>Server: Send Encrypted Updates
    Server->>Server: Aggregate Models
    Server->>Client: Updated Global Model
```

## 4. Knowledge Distillation Process Diagram

```mermaid
graph LR
    subgraph "Teacher Model"
        A[Large Accurate Model]
    end
    
    subgraph "Distillation Process"
        B[Knowledge Transfer]
        C[Loss Function]
    end
    
    subgraph "Student Model"
        D[Compact Efficient Model]
    end
    
    A --> B
    B --> C
    C --> D
    A -.-> D
```

## 5. Data Processing Pipeline Diagram

```mermaid
graph LR
    A[Raw CF Data] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Normalization]
    D --> E[Train/Test Split]
    E --> F[Model Training/Evaluation]
```

## 6. Synthetic Data Generation Flow

```mermaid
graph TD
    A[Real CF Patterns] --> B[Statistical Modeling]
    B --> C[GAN/Simulation]
    C --> D[Synthetic Dataset]
    D --> E[Validation]
    E --> F[Privacy Preservation]
```

## 7. Model Training Data Flow

```mermaid
graph LR
    A[Dataset] --> B[Data Loader]
    B --> C[Preprocessing]
    C --> D[Model Training]
    D --> E[Evaluation]
    E --> F[Metrics]
```

## 8. Inference Data Flow

```mermaid
graph LR
    A[Input Data] --> B[Preprocessing]
    B --> C[Model Inference]
    C --> D[Post-processing]
    D --> E[Prediction Result]
```

## 9. Teacher Model Architecture

```mermaid
graph TB
    A[Input Layer] --> B[Conv Block 1]
    B --> C[Conv Block 2]
    C --> D[Attention Layer]
    D --> E[Feature Extraction]
    E --> F[Classifier]
    F --> G[Output]
```

## 10. Student Model Architecture

```mermaid
graph TB
    A[Input Layer] --> B[Efficient Conv]
    B --> C[Depthwise Separable]
    C --> D[Lightweight Classifier]
    D --> E[Output]
```

## 11. Federated Training Process

```mermaid
graph TD
    A[Server Distributes Model] --> B[Client Downloads Model]
    B --> C[Local Training on Private Data]
    C --> D[Compute Gradients]
    D --> E[Encrypt Model Updates]
    E --> F[Upload Updates to Server]
    F --> G[Aggregate Updates]
    G --> A
```

## 12. Model Evaluation Workflow

```mermaid
graph LR
    A[Test Dataset] --> B[Model Inference]
    B --> C[Calculate Metrics]
    C --> D[Accuracy, Precision, Recall]
    D --> E[Generate Reports]
```

## 13. Deployment Pipeline

```mermaid
graph TD
    A[Model Training] --> B[Model Validation]
    B --> C[Export to ONNX]
    C --> D[Containerization]
    D --> E[Cloud Deployment]
    E --> F[Edge Distribution]
```

## 14. Privacy-Preserving Mechanism Flow

```mermaid
graph LR
    A[Local Data] --> B[Differential Privacy]
    B --> C[Secure Aggregation]
    C --> D[Homomorphic Encryption]
    D --> E[Global Model Update]
```

## 15. Model Accuracy Comparison

```mermaid
graph LR
    A[Traditional ML] --> B[Baseline Accuracy]
    A --> C[Teacher Model] --> D[High Accuracy]
    A --> E[Student Model] --> F[Optimized Accuracy]
    B -.-> F
    D -.-> F
```

## 16. Resource Utilization Chart

```mermaid
graph LR
    A[CPU Usage] --> B[Memory Usage]
    A --> C[Network Bandwidth]
    B --> D[Training Time]
    C --> D
```

## 17. Latency Performance Graph

```mermaid
graph LR
    A[Model Size] --> B[Inference Time]
    A --> C[Edge Device Latency]
    B --> D[Cloud Latency]
```