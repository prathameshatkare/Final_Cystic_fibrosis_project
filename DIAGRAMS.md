# AP_CF_PAPER: Project Diagrams

## 1. System Architecture Diagram

This diagram illustrates the complete system architecture showing how different components interact in the federated learning environment for cystic fibrosis prediction.

```mermaid
graph TB
    subgraph "Client Devices (Edge Nodes)"
        A1[Mobile Device<br/>- Student Model<br/>- Local Inference<br/>- Privacy Preserving]
        A2[Raspberry Pi<br/>- Edge Computing<br/>- CF Data Processing<br/>- Secure Storage]
        A3[Edge Server<br/>- Higher Compute<br/>- Local Aggregation<br/>- Batch Processing]
    end
    
    subgraph "Central Infrastructure"
        B1[Federated Server<br/>- Model Aggregation<br/>- Client Coordination<br/>- Global Model Management]
        B2[Model Registry<br/>- Teacher Model<br/>- Student Model<br/>- Version Control]
        B3[Security Layer<br/>- Encryption<br/>- Authentication<br/>- Access Control]
    end
    
    subgraph "Frontend Interface"
        C1[Web Dashboard<br/>- Model Monitoring<br/>- Performance Metrics<br/>- User Management]
        C2[Mobile UI<br/>- Risk Assessment<br/>- Data Input<br/>- Prediction Results]
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

This diagram shows the complete federated learning process from initialization to convergence, highlighting the iterative nature of distributed model training.

```mermaid
graph TD
    A[Initialize Global Model<br/>- Teacher Model Weights<br/>- Student Model Weights<br/>- Hyperparameters] --> B[Send Model to Clients<br/>- Distribute to K Clients<br/>- Include Privacy Settings<br/>- Set Training Rounds]
    B --> C{Clients Train Locally<br/>- Load Model Weights<br/>- Train on Local Data<br/>- Apply Differential Privacy}
    C --> D[Aggregate Model Updates<br/>- Collect Weight Updates<br/>- Apply Federated Averaging<br/>- Validate Updates]
    D --> E{Convergence Check<br/>- Check Accuracy Threshold<br/>- Verify Convergence Criteria<br/>- Monitor Training Rounds}
    E -->|No| F[Update Global Model<br/>- Apply Aggregated Weights<br/>- Update Model Registry<br/>- Prepare Next Round]
    F --> B
    E -->|Yes| G[Final Model<br/>- Deploy Production Model<br/>- Save Model Artifacts<br/>- Generate Performance Reports]
```

## 3. Client-Server Communication Diagram

This sequence diagram details the secure communication protocol between client devices and the federated server, emphasizing privacy preservation.

```mermaid
sequenceDiagram
    participant Client as Client Device
    participant Server as Federated Server
    participant Security as Security Layer
    
    Note over Client, Server: Phase 1: Model Download
    Client->>+Server: Request Global Model
    Server->>Server: Authenticate Client
    Server->>Security: Encrypt Model Weights
    Security->>Client: Send Encrypted Model
    
    Note over Client: Phase 2: Local Training
    Client->>Client: Decrypt Model Weights
    Client->>Client: Local Training on Private Data
    Client->>Client: Apply Differential Privacy
    Client->>Client: Compute Gradient Updates
    
    Note over Client, Server: Phase 3: Secure Upload
    Client->>Security: Encrypt Gradient Updates
    Security->>Server: Send Encrypted Updates
    Server->>Server: Validate Updates
    Server->>Server: Store Client Updates
    
    Note over Client, Server: Phase 4: Model Update
    Server->>Server: Aggregate All Updates
    Server->>Security: Encrypt New Global Model
    Security->>Client: Send Updated Global Model
    deactivate Server
```

## 4. Knowledge Distillation Process Diagram

This diagram illustrates the knowledge transfer mechanism from the large teacher model to the compact student model, which is crucial for edge deployment.

```mermaid
graph LR
    subgraph "Teacher Model (High Accuracy)"
        A[Large Neural Network<br/>- High Capacity<br/>- High Computational Cost<br/>- Superior Performance]
    end
    
    subgraph "Distillation Process"
        B[Knowledge Transfer<br/>- Soft Targets<br/>- Feature Mapping<br/>- Attention Transfer]
        C[Loss Function<br/>- Cross-Entropy Loss<br/>- Distillation Loss<br/>- Temperature Scaling]
    end
    
    subgraph "Student Model (Efficient)"
        D[Compact Neural Network<br/>- Low Computational Cost<br/>- Small Model Size<br/>- Optimized for Edge]
    end
    
    A --> B
    B --> C
    C --> D
    A -.-> D
    
    style A fill:#ffcccc
    style D fill:#ccffcc
    style B fill:#ffffcc
```

## 5. Data Processing Pipeline Diagram

This diagram outlines the complete data processing pipeline from raw CF data to ready-to-use training data, ensuring privacy and quality.

```mermaid
graph LR
    A[Raw CF Data<br/>- Patient Records<br/>- Genetic Data<br/>- Clinical Measurements] --> B[Data Preprocessing<br/>- Missing Value Imputation<br/>- Outlier Detection<br/>- Data Cleaning]
    B --> C[Feature Engineering<br/>- Genetic Markers<br/>- Demographic Features<br/>- Clinical Indicators]
    C --> D[Normalization<br/>- Min-Max Scaling<br/>- Standardization<br/>- Feature Selection]
    D --> E[Train/Test Split<br/>- Stratified Sampling<br/>- Cross-Validation Folds<br/>- Privacy Preservation]
    E --> F[Model Training/Evaluation<br/>- Local Training<br/>- Federated Learning<br/>- Performance Metrics]
```

## 6. Synthetic Data Generation Flow

This diagram shows the process of creating synthetic CF datasets that preserve privacy while maintaining statistical properties of real data.

```mermaid
graph TD
    A[Real CF Patterns<br/>- Statistical Properties<br/>- Correlation Structures<br/>- Disease Characteristics] --> B[Statistical Modeling<br/>- Probability Distributions<br/>- Generative Models<br/>- Domain Knowledge]
    B --> C[GAN/Simulation<br/>- Generative Adversarial Networks<br/>- Variational Autoencoders<br/>- Monte Carlo Methods]
    C --> D[Synthetic Dataset<br/>- Privacy-Preserved Data<br/>- Realistic Statistics<br/>- Ethical Compliance]
    D --> E[Validation<br/>- Statistical Similarity<br/>- Utility Preservation<br/>- Privacy Verification]
    E --> F[Privacy Preservation<br/>- Differential Privacy<br/>- k-Anonymity<br/>- Data Minimization]
```

## 7. Model Training Data Flow

This diagram details the data flow during model training, from dataset loading to final metrics computation.

```mermaid
graph LR
    A[Dataset<br/>- Training Data<br/>- Validation Data<br/>- Test Data] --> B[Data Loader<br/>- Batch Generation<br/>- Shuffling<br/>- Augmentation]
    B --> C[Preprocessing<br/>- Normalization<br/>- Transformation<br/>- Feature Extraction]
    C --> D[Model Training<br/>- Forward Pass<br/>- Loss Calculation<br/>- Backward Pass<br/>- Parameter Update]
    D --> E[Evaluation<br/>- Validation Metrics<br/>- Overfitting Check<br/>- Performance Tracking]
    E --> F[Metrics<br/>- Accuracy<br/>- Precision/Recall<br/>- AUC-ROC<br/>- F1-Score]
```

## 8. Inference Data Flow

This diagram shows the complete inference process from input data to final prediction, optimized for edge devices.

```mermaid
graph LR
    A[Input Data<br/>- Patient Features<br/>- Genetic Markers<br/>- Clinical Measurements] --> B[Preprocessing<br/>- Data Validation<br/>- Normalization<br/>- Feature Engineering]
    B --> C[Model Inference<br/>- Forward Pass<br/>- Neural Computation<br/>- Probabilistic Output]
    C --> D[Post-processing<br/>- Confidence Scoring<br/>- Uncertainty Quantification<br/>- Result Formatting]
    D --> E[Prediction Result<br/>- Risk Score<br/>- Confidence Level<br/>- Actionable Insights]
```

## 9. Teacher Model Architecture

This diagram illustrates the detailed architecture of the teacher model, highlighting its capacity for high accuracy in CF prediction.

```mermaid
graph TB
    A[Input Layer<br/>- Tabular Data<br/>- Genetic Sequences<br/>- Clinical Features] --> B[Conv Block 1<br/>- Feature Extraction<br/>- Pattern Recognition<br/>- Spatial Relationships]
    B --> C[Conv Block 2<br/>- Deeper Features<br/>- Hierarchical Patterns<br/>- Complex Relationships]
    C --> D[Attention Layer<br/>- Feature Importance<br/>- Context Awareness<br/>- Adaptive Focus]
    D --> E[Feature Extraction<br/>- High-Level Abstractions<br/>- Discriminative Features<br/>- Semantic Representation]
    E --> F[Classifier<br/>- Final Prediction<br/>- Risk Assessment<br/>- Confidence Scoring]
    F --> G[Output<br/>- CF Risk Probability<br/>- Classification Result<br/>- Uncertainty Measure]
```

## 10. Student Model Architecture

This diagram shows the streamlined architecture of the student model, optimized for efficient edge deployment while maintaining predictive capability.

```mermaid
graph TB
    A[Input Layer<br/>- Processed Features<br/>- Normalized Data<br/>- Optimized Format] --> B[Efficient Conv<br/>- Depthwise Separable<br/>- Reduced Parameters<br/>- Faster Computation]
    B --> C[Depthwise Separable<br/>- Channel-wise Convolution<br/>- Point-wise Convolution<br/>- Parameter Efficiency]
    C --> D[Lightweight Classifier<br/>- Reduced Complexity<br/>- Fast Inference<br/>- Memory Efficient]
    D --> E[Output<br/>- Compact Prediction<br/>- Edge-Ready Format<br/>- Low Latency Result]
```

## 11. Federated Training Process

This diagram details the complete federated training cycle, showing how local training contributes to global model improvement.

```mermaid
graph TD
    A[Server Distributes Model<br/>- Global Model Weights<br/>- Training Configuration<br/>- Privacy Parameters] --> B[Client Downloads Model<br/>- Fetch Global Weights<br/>- Configure Local Training<br/>- Set Privacy Levels]
    B --> C[Local Training on Private Data<br/>- Load Patient Data<br/>- Train Model Locally<br/>- Apply Privacy Mechanisms]
    C --> D[Compute Gradients<br/>- Calculate Updates<br/>- Apply Differential Privacy<br/>- Prepare Upload]
    D --> E[Encrypt Model Updates<br/>- Secure Aggregation<br/>- Homomorphic Encryption<br/>- Anonymize Updates]
    E --> F[Upload Updates to Server<br/>- Transmit Securely<br/>- Verify Integrity<br/>- Queue for Aggregation]
    F --> G[Aggregate Updates<br/>- Combine Client Updates<br/>- Apply FedAvg/FedProx<br/>- Validate Consistency]
    G --> A
```

## 12. Model Evaluation Workflow

This diagram outlines the comprehensive evaluation process for assessing model performance in the CF prediction task.

```mermaid
graph LR
    A[Test Dataset<br/>- Unseen Patient Data<br/>- Balanced Classes<br/>- Representative Sample] --> B[Model Inference<br/>- Forward Pass<br/>- Prediction Generation<br/>- Confidence Scoring]
    B --> C[Calculate Metrics<br/>- Accuracy<br/>- Sensitivity/Specificity<br/>- Precision/Recall]
    C --> D[Accuracy, Precision, Recall<br/>- True/False Positives<br/>- True/False Negatives<br/>- Confusion Matrix]
    D --> E[Generate Reports<br/>- Performance Summary<br/>- Comparison Charts<br/>- Clinical Insights]
```

## 13. Deployment Pipeline

This diagram shows the complete deployment pipeline from model training to edge device distribution.

```mermaid
graph TD
    A[Model Training<br/>- Federated Learning<br/>- Knowledge Distillation<br/>- Performance Validation] --> B[Model Validation<br/>- Accuracy Testing<br/>- Privacy Verification<br/>- Edge Compatibility Check]
    B --> C[Export to ONNX<br/>- Format Conversion<br/>- Optimization<br/>- Cross-Platform Ready]
    C --> D[Containerization<br/>- Docker Packaging<br/>- Dependency Management<br/>- Reproducible Environments]
    D --> E[Cloud Deployment<br/>- Server Orchestration<br/>- Load Balancing<br/>- Auto-scaling]
    E --> F[Edge Distribution<br/>- OTA Updates<br/>- Model Sync<br/>- Version Management]
```

## 14. Privacy-Preserving Mechanism Flow

This diagram illustrates the layered privacy preservation approach used throughout the system.

```mermaid
graph LR
    A[Local Data<br/>- Patient Privacy<br/>- Confidential Information<br/>- Sensitive Attributes] --> B[Differential Privacy<br/>- Noise Addition<br/>- Privacy Budget<br/>- Local Randomization]
    B --> C[Secure Aggregation<br/>- Multi-Party Computation<br/>- Secret Sharing<br/>- Masked Updates]
    C --> D[Homomorphic Encryption<br/>- Encrypted Operations<br/>- Secure Computation<br/>- Ciphertext Processing]
    D --> E[Global Model Update<br/>- Aggregated Knowledge<br/>- Improved Performance<br/>- Preserved Privacy]
```

## 15. Model Accuracy Comparison

This diagram compares the performance characteristics of different model approaches in the system.

```mermaid
graph LR
    A[Traditional ML<br/>- Logistic Regression<br/>- Random Forest<br/>- SVM Models] --> B[Baseline Accuracy<br/>- Lower Performance<br/>- Limited Features<br/>- Established Methods]
    A --> C[Teacher Model<br/>- Deep Neural Network<br/>- High Capacity<br/>- Superior Performance]
    C --> D[High Accuracy<br/>- Excellent Results<br/>- High Computational Cost<br/>- Not Edge-Ready]
    A --> E[Student Model<br/>- Compact Architecture<br/>- Optimized for Edge<br/>- Distilled Knowledge]
    E --> F[Optimized Accuracy<br/>- Good Performance<br/>- Low Computational Cost<br/>- Edge-Deployable]
    B -.-> F
    D -.-> F
```

## 16. Resource Utilization Chart

This diagram shows the interdependencies between different resource utilization metrics in the system.

```mermaid
graph LR
    A[CPU Usage<br/>- Training Computation<br/>- Inference Processing<br/>- Model Updates] --> B[Memory Usage<br/>- Model Loading<br/>- Data Buffering<br/>- Intermediate Results]
    A --> C[Network Bandwidth<br/>- Model Transfers<br/>- Updates Upload<br/>- Communication Overhead]
    B --> D[Training Time<br/>- Epoch Duration<br/>- Convergence Speed<br/>- Efficiency Metrics]
    C --> D
```

## 17. Latency Performance Graph

This diagram illustrates the relationship between model size and inference latency across different deployment scenarios.

```mermaid
graph LR
    A[Model Size<br/>- Number of Parameters<br/>- Model Complexity<br/>- Memory Footprint] --> B[Inference Time<br/>- Processing Delay<br/>- Response Speed<br/>- User Experience]
    A --> C[Edge Device Latency<br/>- Mobile CPU<br/>- Battery Constraints<br/>- Real-Time Processing]
    B --> D[Cloud Latency<br/>- Network Delay<br/>- Server Processing<br/>- Total Response Time]
    C --> D
```

## 18. High-Level System Design Diagram

This diagram provides an overview of the complete system architecture with all major components and their interactions.

```mermaid
graph TB
    subgraph "Client Layer"
        CL1[Mobile App<br/>- Patient Interface<br/>- Data Input<br/>- Risk Prediction]
        CL2[Edge Device<br/>- Raspberry Pi<br/>- Local Processing<br/>- Privacy Preservation]
        CL3[Edge Server<br/>- Batch Processing<br/>- Local Aggregation<br/>- Gateway Function]
    end
    
    subgraph "Service Layer"
        SL1[API Gateway<br/>- Request Routing<br/>- Authentication<br/>- Rate Limiting]
        SL2[Federated Service<br/>- Model Distribution<br/>- Weight Aggregation<br/>- Client Coordination]
        SL3[Model Service<br/>- Model Serving<br/>- Inference Engine<br/>- Performance Monitoring]
    end
    
    subgraph "Data Layer"
        DL1[Patient Data Store<br/>- Encrypted Storage<br/>- Privacy Controls<br/>- Access Logs]
        DL2[Model Registry<br/>- Version Management<br/>- Metadata Storage<br/>- Audit Trail]
        DL3[Synthetic Data Generator<br/>- Pattern Synthesis<br/>- Privacy Preservation<br/>- Validation Engine]
    end
    
    subgraph "Security Layer"
        SEC1[Encryption Service<br/>- Data Encryption<br/>- Key Management<br/>- Certificate Authority]
        SEC2[Privacy Manager<br/>- Differential Privacy<br/>- Secure Aggregation<br/>- Anonymization]
        SEC3[Access Control<br/>- Authentication<br/>- Authorization<br/>- Audit Logging]
    end
    
    CL1 --> SL1
    CL2 --> SL1
    CL3 --> SL1
    SL1 --> SL2
    SL1 --> SL3
    SL2 --> DL2
    SL3 --> DL1
    DL3 --> DL1
    SEC1 --> SL1
    SEC2 --> SL2
    SEC3 --> SL1
```

## 19. UML Class Diagram

This UML class diagram shows the main classes and their relationships in the system.

```mermaid
classDiagram
    class CFModel {
        <<abstract>>
        - model_id: str
        - version: str
        - hyperparameters: dict
        + train(data: np.ndarray, epochs: int): None
        + evaluate(test_data: np.ndarray): dict
        + predict(input_data: np.ndarray): np.ndarray
        + save(path: str): None
        + load(path: str): None
    }
    
    class TeacherModel {
        - num_layers: int
        - hidden_size: int
        - attention_heads: int
        + forward(x: torch.Tensor): torch.Tensor
        + compute_loss(predictions: torch.Tensor, targets: torch.Tensor): torch.Tensor
        + extract_features(x: torch.Tensor): torch.Tensor
    }
    
    class StudentModel {
        - compressed_size: int
        - quantized: bool
        - distillation_temperature: float
        + forward(x: torch.Tensor): torch.Tensor
        + distill(teacher_output: torch.Tensor): torch.Tensor
        + compress_weights(): None
        + quantize_model(): None
    }
    
    class FederatedClient {
        - client_id: str
        - is_active: bool
        - local_model: CFModel
        - data_manager: CFDataManager
        - privacy_manager: PrivacyManager
        + local_train(data: np.ndarray): dict
        + update_weights(global_weights: dict): None
        + send_updates(server: FederatedServer): dict
        + receive_global_model(model_state: dict): None
        + validate_local_model(test_data: np.ndarray): dict
    }
    
    class FederatedServer {
        - clients: list[FederatedClient]
        - global_model: CFModel
        - round_number: int
        - aggregation_strategy: str
        - privacy_budget: float
        + aggregate_weights(client_updates: list[dict]): dict
        + distribute_model(clients: list[FederatedClient]): None
        + coordinate_training(num_rounds: int): None
        + select_clients(num_clients: int): list[FederatedClient]
        + broadcast_global_model(): None
    }
    
    class CFDataManager {
        - dataset_path: str
        - preprocessing_config: dict
        - feature_columns: list[str]
        - target_column: str
        + load_data(): pd.DataFrame
        + preprocess(raw_data: pd.DataFrame): np.ndarray
        + split_data(train_ratio: float): tuple
        + validate_data(data: np.ndarray): bool
        + generate_synthetic_data(num_samples: int): pd.DataFrame
    }
    
    class PrivacyManager {
        - epsilon: float
        - delta: float
        - noise_multiplier: float
        - clip_bound: float
        + apply_differential_privacy(gradient: torch.Tensor): torch.Tensor
        + secure_aggregate(weights_list: list[dict]): dict
        + encrypt_data(data: bytes): bytes
        + decrypt_data(encrypted_data: bytes): bytes
        + calculate_privacy_budget(noise_level: float, steps: int): float
    }
    
    class ModelEvaluator {
        - metrics_config: dict
        - confidence_threshold: float
        + calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray): float
        + calculate_precision_recall(y_true: np.ndarray, y_pred: np.ndarray): tuple
        + calculate_auc_roc(y_true: np.ndarray, y_scores: np.ndarray): float
        + generate_report(results: dict): str
        + plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray): plt.Figure
        + plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray): plt.Figure
    }
    
    class ModelRegistry {
        - registry_path: str
        - models: dict[str, dict]
        + register_model(model: CFModel, metadata: dict): str
        + get_model(model_id: str, version: str): CFModel
        + list_models(): list[dict]
        + delete_model(model_id: str, version: str): bool
        + update_model_metadata(model_id: str, version: str, metadata: dict): bool
    }
    
    class APIServer {
        - host: str
        - port: int
        - model_service: ModelRegistry
        - federated_service: FederatedServer
        + start_server(): None
        + stop_server(): None
        + predict_endpoint(request: dict): dict
        + train_endpoint(request: dict): dict
        + register_client_endpoint(client_info: dict): str
    }
    
    CFModel <|-- TeacherModel
    CFModel <|-- StudentModel
    TeacherModel <..> StudentModel : Knowledge Distillation
    FederatedClient o-- CFModel : uses
    FederatedClient --> CFDataManager : uses
    FederatedClient --> PrivacyManager : uses
    FederatedServer o-- CFModel : manages
    FederatedServer --> ModelEvaluator : uses
    FederatedServer }--{ FederatedClient : coordinates
    CFDataManager --> ModelEvaluator : provides_data
    PrivacyManager <-- FederatedClient : applies_privacy
    PrivacyManager <-- FederatedServer : aggregates_securely
    ModelRegistry <-- APIServer : serves_models
    APIServer --> FederatedServer : manages_federation
```

## 20. UML Sequence Diagram

This UML sequence diagram shows the interaction between components during a federated learning round.

```mermaid
sequenceDiagram
    participant FS as FederatedServer
    participant FC as FederatedClient
    participant PM as PrivacyManager
    participant MM as ModelManager
    participant DM as DataManager
    
    Note over FS, DM: Federated Learning Round Initialization
    FS->>FC: Broadcast Global Model Weights
    activate FC
    FC->>MM: Load Global Model
    MM-->>FC: Model Loaded
    FC->>DM: Request Local Training Data
    DM-->>FC: Return Local Data
    
    Note over FC: Local Training Phase
    FC->>FC: Local Model Training
    FC->>PM: Apply Differential Privacy
    PM-->>FC: Privatized Gradients
    
    Note over FC, FS: Secure Aggregation
    FC->>PM: Encrypt Model Updates
    PM-->>FC: Encrypted Updates
    FC->>FS: Send Encrypted Updates
    deactivate FC
    FS->>PM: Decrypt and Aggregate
    PM-->>FS: Aggregated Global Model
    
    Note over FS: Global Model Update
    FS->>MM: Update Global Model
    MM-->>FS: Model Updated
```

## 21. Entity Relationship Diagram

This ER diagram shows the relationships between different entities in the system.

```mermaid
erDiagram
    PATIENT {
        string patient_id PK
        string genetic_markers
        json clinical_data
        datetime created_at
        datetime updated_at
    }
    
    MODEL_VERSION {
        string model_id PK
        string version PK
        string model_type
        float accuracy
        datetime created_at
        json architecture
    }
    
    TRAINING_SESSION {
        string session_id PK
        string model_id FK
        string client_id
        int round_number
        float local_loss
        datetime start_time
        datetime end_time
    }
    
    CLIENT {
        string client_id PK
        string device_type
        string location
        boolean is_active
        datetime registered_at
    }
    
    MODEL_UPDATE {
        string update_id PK
        string session_id FK
        string model_id FK
        json weights
        float differential_privacy_epsilon
        datetime timestamp
    }
    
    EVALUATION_RESULT {
        string result_id PK
        string model_id FK
        string dataset_id
        float accuracy
        float precision
        float recall
        float auc_score
        json confusion_matrix
        datetime evaluated_at
    }
    
    PATIENT ||--o{ TRAINING_SESSION : participates_in
    MODEL_VERSION ||--o{ TRAINING_SESSION : used_for
    CLIENT ||--o{ TRAINING_SESSION : executes
    TRAINING_SESSION ||--o{ MODEL_UPDATE : generates
    MODEL_VERSION ||--o{ EVALUATION_RESULT : evaluated
```