# AP_CF_PAPER: Cystic Fibrosis Prediction System with Federated Learning

This project implements a comprehensive cystic fibrosis prediction system using federated learning and knowledge distillation techniques. The system enables accurate CF risk prediction while preserving patient privacy by keeping sensitive data on local devices.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Federated Learning Process](#federated-learning-process)
- [Model Architecture](#model-architecture)
- [Data Privacy](#data-privacy)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

The AP_CF_PAPER project combines advanced machine learning techniques with federated learning to create a privacy-preserving cystic fibrosis prediction system. The system consists of:
- Teacher model for high-accuracy predictions
- Student model for efficient edge deployment
- Federated learning infrastructure
- Privacy-preserving data processing
- Web and mobile interfaces

## Architecture

The system is composed of several interconnected components:

### Backend Services
- **API Layer** (`api/main.py`): FastAPI-based RESTful services for model inference and coordination
- **Federated Learning** (`federated/`): Client-server communication for distributed training
- **Models** (`models/`): Teacher and student neural network architectures
- **Training** (`training/`): Local training and evaluation modules

### Frontend Interface
- **Web Application** (`frontend/`): React-based UI for model interaction and visualization
- **Mobile Compatibility**: Optimized for mobile edge deployment

### Data Processing
- **Synthetic Dataset** (`data/synthetic_cystic_fibrosis_dataset.csv`): Privacy-preserving dataset for model training
- **Genetic Mutations** (`data/mutations.json`): CF-related genetic variant information
- **Data Generation** (`data/generate_data.py`): Synthetic data creation tools

## Features

- **Privacy-Preserving**: Uses federated learning to keep patient data on local devices
- **Knowledge Distillation**: Efficient student model for edge deployment
- **Multi-Modal**: Supports both tabular and vision-based inputs
- **Scalable**: Designed for distributed deployment across multiple institutions
- **Secure**: Implements encryption and secure aggregation protocols
- **Accessible**: Web-based interface for non-technical users

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- CUDA-compatible GPU (optional, for accelerated training)

### Backend Setup
```bash
# Clone the repository
git clone <repository-url>
cd AP_CF_PAPER

# Install Python dependencies
pip install -r requirements.txt

# Install config dependencies
pip install -r configs/requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## Usage

### Running the API Server
```bash
cd api
python main.py
```

### Running the Frontend
```bash
cd frontend
npm run dev
```

### Training Models
```bash
# Local training
cd training
python local_train.py

# Evaluate models
python eval.py
```

### Federated Learning Simulation
```bash
cd experiments
python simulate_fl_cfvision.py
```

## Federated Learning Process

The system implements a teacher-student knowledge distillation approach in a federated setting:

1. **Initialization**: Server distributes global teacher and student models to clients
2. **Local Training**: Clients train local student models on their private data
3. **Weight Updates**: Clients send encrypted model updates to the server
4. **Aggregation**: Server aggregates updates using federated averaging
5. **Distribution**: Updated global model is distributed to clients

### Privacy Mechanisms
- Differential privacy for local training
- Secure aggregation for model updates
- Homomorphic encryption for communication

## Model Architecture

### Teacher Model (`models/cfvision_teacher.py`)
- High-capacity neural network for optimal accuracy
- Advanced attention mechanisms
- Ensemble components for robust predictions

### Student Model (`models/cfvision_student.py`)
- Lightweight architecture for edge deployment
- Knowledge distillation from teacher model
- Quantization-ready layers for mobile optimization

### Tabular Model (`models/cf_tabular.py`)
- Feature engineering for structured medical data
- Embedding layers for categorical variables
- Regularization for sparse medical features

## Data Privacy

The system implements multiple privacy-preserving techniques:

- **Local Data Processing**: Patient data never leaves the local device
- **Encrypted Communications**: All transmissions are encrypted
- **Secure Aggregation**: Model updates are aggregated without revealing individual contributions
- **Differential Privacy**: Noise is added to protect individual data points

## Deployment

### Cloud Deployment
The system supports deployment on cloud platforms:

```bash
# Deploy to Render
# Configure render.yaml according to your requirements
```

### Edge Deployment
For edge devices including mobile phones and Raspberry Pi:

```bash
# Mobile deployment
python edge_inference.py

# Export to ONNX for cross-platform compatibility
python export_to_onnx.py
```

### Docker Support
Coming soon for containerized deployment.

## Contributing

We welcome contributions to improve the system:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 coding standards
- Write comprehensive docstrings
- Maintain backward compatibility
- Update documentation as needed

## Experiments

The `experiments/` directory contains:
- Baseline comparisons (`baselines.py`)
- Federated learning simulations (`simulate_fl_cfvision.py`)
- Performance evaluation tools

## Data

The system uses synthetic cystic fibrosis data to preserve privacy:
- Generated to reflect real-world CF characteristics
- Includes genetic mutations and clinical features
- Compliant with medical privacy regulations

## Troubleshooting

Common issues and solutions:

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory Issues
Reduce batch sizes or use model quantization for edge devices.

### Network Connectivity
Configure proxy settings if behind corporate firewall.

## Future Work

Planned improvements include:
- Enhanced privacy mechanisms
- Improved model compression
- Automated hyperparameter tuning
- Clinical validation studies
- Regulatory compliance features

## License

This project is licensed under [LICENSE TYPE] - see the LICENSE file for details.

## Acknowledgments

This work builds upon advances in federated learning, knowledge distillation, and medical AI research.