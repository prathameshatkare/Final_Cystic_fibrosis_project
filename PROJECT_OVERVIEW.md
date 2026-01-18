# AP_CF_PAPER Project Overview

This document provides a comprehensive overview of the AP_CF_PAPER project structure, explaining the purpose of each file and directory in the cystic fibrosis prediction system.

## Directory Structure

```
├── api/
│   └── main.py
├── configs/
│   └── requirements.txt
├── data/
│   ├── generate_data.py
│   ├── mutations.json
│   └── synthetic_cystic_fibrosis_dataset.csv
├── experiments/
│   ├── baselines.py
│   └── simulate_fl_cfvision.py
├── federated/
│   ├── client.py
│   └── server_strategy.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── .gitignore
│   ├── README.md
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── models/
│   ├── cf_tabular.py
│   ├── cfvision_student.py
│   └── cfvision_teacher.py
├── training/
│   ├── eval.py
│   └── local_train.py
├── (root directory files)
```

## File Descriptions

### API Directory
- **api/main.py**: Main entry point for the application's API, handling HTTP requests and routing them to appropriate functions for the cystic fibrosis prediction system.

### Configs Directory
- **configs/requirements.txt**: Contains a list of Python dependencies needed for the project configuration, specifying the packages and versions required to run the application.

### Data Directory
- **data/generate_data.py**: Script for generating synthetic or simulated data for the cystic fibrosis dataset, possibly for testing or expanding the training dataset.
- **data/mutations.json**: Contains information about genetic mutations related to cystic fibrosis, likely used for reference or as input data for the models.
- **data/synthetic_cystic_fibrosis_dataset.csv**: The actual dataset containing synthetic patient data for cystic fibrosis research, with features and labels for machine learning tasks.

### Experiments Directory
- **experiments/baselines.py**: Contains baseline algorithms or models used for comparison against the main CF prediction models to evaluate performance.
- **experiments/simulate_fl_cfvision.py**: Script for simulating federated learning scenarios specifically for the CF vision model, testing how the system performs in a distributed learning environment.

### Federated Directory
- **federated/client.py**: Implements the client-side logic for federated learning, where individual nodes participate in collaborative model training without sharing raw data.
- **federated/server_strategy.py**: Defines the server-side strategy for federated learning, including how to aggregate model updates from clients and coordinate the training process.

### Frontend Directory
The frontend directory contains a complete React/Vite TypeScript application:
- **frontend/src/App.tsx**: Main React component that serves as the root of the application UI
- **frontend/src/index.css**: Global CSS styles for the application
- **frontend/src/main.tsx**: Entry point for the React application
- **frontend/.gitignore**: Specifies files and directories to be ignored by Git
- **frontend/README.md**: Documentation for the frontend application
- **frontend/eslint.config.js**: ESLint configuration for code linting
- **frontend/index.html**: HTML template for the React application
- **frontend/package-lock.json**: Lock file for npm dependencies
- **frontend/package.json**: Lists project dependencies and scripts
- **frontend/tsconfig.app.json**, **frontend/tsconfig.json**, **frontend/tsconfig.node.json**: TypeScript configuration files
- **frontend/vite.config.ts**: Vite build tool configuration

### Models Directory
- **models/cf_tabular.py**: Contains tabular data processing models for cystic fibrosis prediction, likely handling structured data like patient records.
- **models/cfvision_student.py**: Implements the "student" model in a knowledge distillation framework, designed to be a lightweight version of the teacher model for efficient deployment.
- **models/cfvision_teacher.py**: Implements the "teacher" model in a knowledge distillation framework, typically a larger, more accurate model used to train the student model.

### Training Directory
- **training/eval.py**: Contains evaluation functions to assess model performance on validation/test datasets.
- **training/local_train.py**: Script for performing local model training, likely used as part of the federated learning process or for standalone training.

### Root Directory Files
- **.gitignore**: Specifies files and directories to be ignored by Git version control
- **AUTOMATION_AND_SYNC_GUIDE.md**: Documentation for automation and synchronization procedures
- **DATA_FLOW_DIAGRAMS.md**: Documentation with diagrams showing data flow through the system
- **DEPLOYMENT_GUIDE.md**: Instructions for deploying the application
- **DEPLOY_NOW.md**: Quick deployment guide
- **EDGE_DEPLOYMENT_GUIDE.md**: Specific instructions for deploying to edge devices
- **ENTITY_RELATIONSHIP_DIAGRAMS.md**: Documentation with entity relationship diagrams
- **HOW_EDGE_DEVICES_PROCESS.md**: Documentation on how edge devices process data
- **MATHEMATICAL_MODEL.md**: Mathematical formulations and models used in the project
- **MOBILE_EDGE_IMPLEMENTATION.md**: Guide for mobile edge implementation
- **MODEL_UPGRADE_GUIDE.md**: Instructions for upgrading models
- **ONLINE_DEPLOYMENT.md**: Guide for online deployment
- **SYSTEM_ARCHITECTURE.md**: Documentation describing the system architecture
- **UML_DIAGRAMS.md**: Documentation with UML diagrams
- **automate_upgrade.py**: Script for automating model upgrades
- **edge_inference.py**: Script for performing inference on edge devices
- **export_to_onnx.py**: Script to convert models to ONNX format for cross-platform compatibility
- **render.yaml**: Configuration file for deployment to Render platform
- **report.md**: Report or summary document
- **requirements.txt**: Python dependencies for the main project
- **test_deployment.py**: Script for testing deployments