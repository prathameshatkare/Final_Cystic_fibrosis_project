# CFVision-FL Mathematical Model

## Overview
The CFVision-FL system implements a 3-layer neural network for binary classification of Cystic Fibrosis diagnosis. This mathematical model combines clinical markers with genetic information to provide accurate risk assessment while maintaining interpretability.

## Neural Network Architecture

### Network Structure
```
Input Layer: R^36 → Hidden Layer 1: R^64 → Hidden Layer 2: R^32 → Output Layer: R^2
```

### Detailed Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│  (36 Clinical Features after Preprocessing)                │
├─────────────────────────────────────────────────────────────┤
│    x ∈ R^36 = [age_months, family_history_cf, ...]        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                HIDDEN LAYER 1                             │
│                (Linear + BN + ReLU)                       │
├─────────────────────────────────────────────────────────────┤
│  h₁ = ReLU(BatchNorm(W₁x + b₁))                           │
│  where:                                                    │
│  - W₁ ∈ R^(64×36) : Weight matrix                         │
│  - b₁ ∈ R^64 : Bias vector                                │
│  - BatchNorm: Normalizes inputs to stabilize training      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                HIDDEN LAYER 2                             │
│        (Linear + BN + Dropout + ReLU)                     │
├─────────────────────────────────────────────────────────────┤
│  h₂ = ReLU(Dropout(BatchNorm(W₂h₁ + b₂)))                 │
│  where:                                                    │
│  - W₂ ∈ R^(32×64) : Weight matrix                         │
│  - b₂ ∈ R^32 : Bias vector                                │
│  - Dropout(p=0.3): Regularization technique               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                OUTPUT LAYER                               │
│                (Linear Transformation)                    │
├─────────────────────────────────────────────────────────────┤
│  logits = W₃h₂ + b₃                                       │
│  where:                                                    │
│  - W₃ ∈ R^(2×32) : Weight matrix                          │
│  - b₃ ∈ R^2 : Bias vector                                 │
│  - Output: 2-dimensional vector (CF vs Non-CF)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PROBABILITY CALCULATION                      │
├─────────────────────────────────────────────────────────────┤
│  p = softmax(logits)                                       │
│  p = [p_non_cf, p_cf]                                     │
│  where p_cf is the probability of CF diagnosis            │
└─────────────────────────────────────────────────────────────┘
```

## Mathematical Formulation

### Forward Pass
1. **Input**: x ∈ R^36 (36 clinical features after preprocessing)
2. **Hidden Layer 1**:
   ```
   z₁ = W₁x + b₁
   h̃₁ = BatchNorm(z₁)
   h₁ = ReLU(h̃₁)
   ```
3. **Hidden Layer 2**:
   ```
   z₂ = W₂h₁ + b₂
   h̃₂ = BatchNorm(z₂)
   h̃₂_dropout = Dropout(h̃₂)
   h₂ = ReLU(h̃₂_dropout)
   ```
4. **Output Layer**:
   ```
   logits = W₃h₂ + b₃
   ```
5. **Softmax Activation**:
   ```
   p_i = exp(logits_i) / Σⱼ exp(logits_j)  for i,j ∈ {0,1}
   ```

## Loss Function

### During Centralized Training
```
L_total = L_CE + L_reg
```

Where:
- **Cross-Entropy Loss**:
  ```
  L_CE = -Σᵢ yᵢ log(pᵢ)
  ```
  where yᵢ is the true label (one-hot encoded) and pᵢ is the predicted probability

- **Focal Loss Option** (for handling class imbalance):
  ```
  L_FL = -α(1-p)ᵞ log(p) if y=1, else -αpᵞ log(1-p)
  ```
  where α is the balancing factor and γ is the focusing parameter

### During Federated Learning (FedProx)
```
L_FedProx = L_local + (μ/2)||w - w_t||²
```
where:
- L_local is the local loss on client data
- μ is the proximal term coefficient
- w are the current model weights
- w_t are the global model weights from the previous round

## Preprocessing Transformations

### Standardization
```
x_norm = (x - μ) / σ
```
where μ and σ are computed from the training data

### One-Hot Encoding
For categorical variables with K categories:
```
x_categorical ∈ R^K (one-hot encoded vector)
```

### Genetic Weighting
Based on CFTR2 database classification:
```
p_final = p_initial + Δp_genetic
```
where Δp_genetic depends on the mutation type:
- CF-causing: +0.4 (capped at 0.99)
- Varying consequence: +0.2 (capped at 0.95)
- Non CF-causing: -0.1 (floored at 0.01)
- Unknown significance: +0.05 (capped at 0.90)

## Performance Metrics

### Binary Classification Metrics
- **Accuracy**:
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```
- **Precision**:
  ```
  Precision = TP / (TP + FP)
  ```
- **Recall/Sensitivity**:
  ```
  Recall = TP / (TP + FN)
  ```
- **Specificity**:
  ```
  Specificity = TN / (TN + FP)
  ```
- **F1-Score**:
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve

## Model Complexity
- **Total Parameters**: Approximately 4,000 parameters
- **Layer-wise Breakdown**:
  - Input to Hidden 1: (64×36) + 64 = 2,368 parameters
  - Hidden 1 to Hidden 2: (32×64) + 32 = 2,080 parameters  
  - Hidden 2 to Output: (2×32) + 2 = 66 parameters
  - **Total**: 4,514 parameters

This compact architecture is designed for efficient execution on edge devices while maintaining high diagnostic accuracy.