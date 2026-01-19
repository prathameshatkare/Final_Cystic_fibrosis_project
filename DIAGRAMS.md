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
