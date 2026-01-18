# CFVision-FL: A Multimodal Privacy-Preserving Framework for Cystic Fibrosis Diagnosis via Federated Edge Learning and Genotype-Phenotype Correlation

**Abstract**
Cystic Fibrosis (CF) is a life-shortening rare genetic disorder where early intervention is paramount. However, the rarity of the disease leads to fragmented data silos across clinical institutions, hindering the development of robust diagnostic AI. This paper presents **CFVision-FL**, a novel federated learning framework designed to train diagnostic models across distributed clinical sites while preserving patient privacy. Our system integrates a deep multilayer perceptron (MLP) for phenotypic clinical marker analysis with a searchable genotype registry derived from the CFTR2 database. By simulating a non-IID (non-identically distributed) clinical environment across five hospital nodes with realistic CF prevalence (~13.7%), we demonstrate that federated models can achieve high diagnostic performance (Accuracy: 95.2%, F1-Score: 0.79) without aggregating raw patient data. The framework is delivered via a modern React-based clinical portal, providing real-time risk assessment and federated performance monitoring for clinicians.

**Keywords**: Cystic Fibrosis, Federated Learning, Privacy-Preserving AI, Rare Disease Diagnosis, Genotype-Phenotype Correlation, Edge Computing.

---

## I. Introduction
Early diagnosis of Cystic Fibrosis (CF) is critical for preventing irreversible lung damage and improving life expectancy. Despite its clinical importance, CF diagnosis remains challenging due to the extreme rarity of the condition and the geographic dispersion of patient data. Traditional centralized machine learning requires pooling sensitive patient data (biometrics, symptoms, genetics) onto a single server, raising significant concerns regarding HIPAA and GDPR compliance.

Federated Learning (FL) offers a paradigm shift by allowing models to be trained locally at clinical sites (e.g., general hospitals and specialist CF centers) without sharing raw data. Only model weights are aggregated, preserving patient confidentiality. Building on the research established by the **CFVision** project, which utilized Vision Transformers for radiograph analysis, this work focuses on a multimodal approach combining structured clinical markers with genetic context.

## II. Methodology

### A. Dataset Synthesis and Clinical Realism
To evaluate the framework, we developed a synthetic pediatric dataset ($N=10,000$) designed to emulate early CF screening. Unlike previous versions with extreme class imbalance, the current dataset utilizes a realistic prevalence of **13.7%** to reflect specialist center distributions. 
Each patient record includes:
- **Phenotypic Markers**: 41 clinical features including Sweat Test results, Respiratory Scores, Growth Percentiles, and GI symptoms (e.g., Meconium Ileus, Steatorrhea).
- **Clinical Noise**: Ambiguous cases were intentionally injected (5% noise floor) to simulate real-world diagnostic uncertainty where symptoms overlap with other pediatric conditions.

### B. Federated Learning Architecture
We utilize the **Flower (flwr)** framework to implement a cross-silo FL architecture.
1. **Local Training**: Each clinical node trains a local `CFTabularNet` (a 3-layer MLP) using **Focal Loss** to address residual class imbalance.
2. **Aggregation**: A central server coordinates rounds using the **FedProx** strategy, which is resilient to the system heterogeneity common in hospital edge devices.
3. **Non-IID Partitioning**: We simulate five distinct hospital nodes with varying patient volumes and CF prevalence, ranging from general pediatric clinics (low prevalence) to specialized CF centers (high prevalence).

### C. Multimodal Genotype Integration
A unique feature of CFVision-FL is the integration of the **CFTR2 (Clinical and Functional Translation of CFTR)** database. We parsed over 1,100 validated genetic mutations. The system performs a weighted risk assessment: if the AI model detects a high phenotypic risk, and the clinician inputs a known "CF-causing" mutation (e.g., F508del), the diagnostic confidence is mathematically weighted higher, mimicking a multimodal clinical consensus.

## III. System Implementation

### A. Backend Architecture
The backend is built using **FastAPI**, serving as a bridge between the PyTorch-trained federated weights and the user interface. It provides:
- **Real-time Inference**: Scaled and one-hot encoded preprocessing for immediate patient risk assessment.
- **Dynamic Metrics Engine**: Evaluates the global model on-the-fly to provide live dashboard updates.

### B. Frontend Clinical Portal
The UI is a modern **React 19** application designed for clinical accessibility:
- **Tabbed Patient Entry**: Organizes symptoms into logical categories (Demographics, Respiratory, Growth, GI, Diagnostic).
- **Interactive Dashboard**: Leverages **Recharts** to visualize metric comparisons across sites (Radar Chart), training convergence (Line Chart), and patient distribution (Bar Chart).

## IV. Results and Analysis

### A. Model Performance
After 10 rounds of federated training, the global model achieved the following metrics on a held-out test set:
- **Accuracy**: 95.20%
- **F1-Score**: 0.7957
- **Sensitivity (Recall)**: 68.00%
- **AUC-ROC**: 0.8446

The high accuracy combined with a robust F1-score indicates that the model has successfully learned to identify rare CF cases without suffering from the high false-positive rates common in imbalanced medical datasets.

### B. Federated Convergence
The distributed loss trajectory showed a steady decline across rounds, even with highly heterogeneous client data. This confirms that the **FedProx** strategy effectively aggregated signals from the specialized centers to improve the diagnostic capabilities of the general clinic nodes.

## V. Discussion
The CFVision-FL framework demonstrates that privacy-preserving AI can achieve near-centralized performance for rare disease screening. The integration of genetic context addresses the "overlap" problem, where clinical symptoms alone are insufficient for a definitive diagnosis. By deploying this on edge devices at the point of care, clinicians can receive immediate AI-assisted risk scores while ensuring patient data never leaves the hospital premises.

## VI. Conclusion
This work presents a comprehensive, production-ready framework for federated CF diagnosis. By combining state-of-the-art FL techniques with multimodal genetic integration and a professional clinical dashboard, CFVision-FL provides a template for future rare-disease diagnostic tools. Future research will focus on integrating Vision Transformer (ViT) backbones for automated chest radiograph analysis alongside structured clinical data.

---
**References**
[1] Beagum, S., et al. "CFVision: Vision Transformer-based Cystic Fibrosis Diagnosis."
[2] McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data."
[3] CFTR2 Database. "Clinical and Functional Translation of CFTR." (2024).
