# CFVision-FL Entity Relationship Diagrams

## Overview
The CFVision-FL system maintains a normalized data model to support clinical data management, model tracking, and federated learning operations. This document presents the entity relationships and data structure for the system.

## Conceptual Data Model

### Core Entities

```
┌─────────────────────────┐
│        Patient          │
├─────────────────────────┤
│ patient_id (PK)         │
│ age_months              │
│ family_history_cf       │
│ ethnicity               │
│ gender                  │
│ weight_percentile       │
│ height_percentile       │
│ weight_for_height       │
│ growth_faltering        │
│ failure_to_thrive       │
│ appetite                │
│ nutritional_risk_score  │
│ newborn_screening_result│
│ meconium_ileus          │
│ prolonged_jaundice      │
│ salty_skin              │
│ cough_type              │
│ cough_character         │
│ respiratory_infections_  │
│ frequency               │
│ wheezing_present        │
│ nasal_polyps            │
│ clubbing_fingers        │
│ respiratory_score       │
│ stool_character         │
│ stool_frequency         │
│ abdominal_distention    │
│ diarrhea_chronic        │
│ fat_malabsorption_signs │
│ sweat_test_simulated    │
│ cf_clinical_suspicion_  │
│ index                   │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

```
┌─────────────────────────┐
│    ClinicalObservation  │
├─────────────────────────┤
│ observation_id (PK)     │
│ patient_id (FK)         │
│ observation_date        │
│ age_months              │
│ family_history_cf       │
│ ethnicity               │
│ weight_percentile       │
│ height_percentile       │
│ weight_for_height       │
│ growth_faltering        │
│ failure_to_thrive       │
│ appetite                │
│ nutritional_risk_score  │
│ newborn_screening_result│
│ meconium_ileus          │
│ prolonged_jaundice      │
│ salty_skin              │
│ cough_type              │
│ cough_character         │
│ respiratory_infections_  │
│ frequency               │
│ wheezing_present        │
│ nasal_polyps            │
│ clubbing_fingers        │
│ respiratory_score       │
│ stool_character         │
│ stool_frequency         │
│ abdominal_distention    │
│ diarrhea_chronic        │
│ fat_malabsorption_signs │
│ sweat_test_simulated    │
│ cf_clinical_suspicion_  │
│ index                   │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

```
┌─────────────────────────┐
│      GeneticMarker      │
├─────────────────────────┤
│ marker_id (PK)          │
│ mutation_name           │
│ determination           │
│ clinical_significance   │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

```
┌─────────────────────────┐
│       Diagnosis         │
├─────────────────────────┤
│ diagnosis_id (PK)       │
│ patient_id (FK)         │
│ cf_diagnosis            │
│ diagnosis_probability   │
│ risk_level              │
│ diagnostic_confidence   │
│ age_at_diagnosis        │
│ genetic_marker_id (FK)  │
│ diagnosis_method        │
│ diagnosis_timestamp     │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

```
┌─────────────────────────┐
│      EdgeDevice         │
├─────────────────────────┤
│ device_id (PK)          │
│ device_type             │
│ device_name             │
│ last_sync_time          │
│ model_version           │
│ total_predictions       │
│ online_status           │
│ location                │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

```
┌─────────────────────────┐
│        Model            │
├─────────────────────────┤
│ model_id (PK)           │
│ model_name              │
│ model_path              │
│ training_timestamp      │
│ accuracy                │
│ precision               │
│ recall                  │
│ specificity             │
│ f1_score                │
│ auc_score               │
│ total_samples           │
│ training_epochs         │
│ learning_rate           │
│ loss_function           │
│ architecture            │
│ input_features          │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

```
┌─────────────────────────┐
│     TrainingSession     │
├─────────────────────────┤
│ session_id (PK)         │
│ model_id (FK)           │
│ training_start_time     │
│ training_end_time       │
│ training_method         │
│ federated_rounds        │
│ centralized_epochs      │
│ training_data_size      │
│ validation_accuracy     │
│ validation_loss         │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

```
┌─────────────────────────┐
│    FederationRound      │
├─────────────────────────┤
│ round_id (PK)           │
│ session_id (FK)         │
│ round_number            │
│ start_time              │
│ end_time                │
│ participating_clients   │
│ global_accuracy         │
│ global_loss             │
│ avg_client_accuracy     │
│ created_at              │
│ updated_at              │
└─────────────────────────┘
```

## Entity Relationships

### Primary Relationships

```
Patient (1) ──── (M) ClinicalObservation
  │                    │
  │                    │ patient_id (FK)
  │                    │
  │                    ▼
  │         ClinicalObservation contains
  │         patient demographic and
  │         symptom data
  │
  │
Patient (1) ──── (1) Diagnosis
  │                    │
  │                    │ patient_id (FK)
  │                    │
  │                    ▼
  │         Diagnosis represents
  │         the outcome for a patient
  │
  │
Patient (M) ──── (1) GeneticMarker
  │                    │
  │                    │ genetic_marker_id (FK)
  │                    │
  │                    ▼
  │         Genetic markers associated
  │         with patient diagnosis
  │
  │
Diagnosis (M) ──── (1) Model
  │                    │
  │                    │ model_id (FK)
  │                    │
  │                    ▼
  │         Diagnoses are produced
  │         by specific models
  │
  │
Model (1) ──── (M) TrainingSession
  │                    │
  │                    │ model_id (FK)
  │                    │
  │                    ▼
  │         Each model has
  │         training history
  │
  │
TrainingSession (1) ──── (M) FederationRound
  │                              │
  │                              │ session_id (FK)
  │                              │
  │                              ▼
  │                   Each training session
  │                   may involve multiple
  │                   federation rounds
  │
  │
EdgeDevice (M) ──── (M) ClinicalObservation
  │                              │
  │                              │ device_id (stored in metadata)
  │                              │
  │                              ▼
  │                   Edge devices collect
  │                   and transmit observations
  │
  │
EdgeDevice (M) ──── (1) Model
  │                    │
  │                    │ model_version (links to model_id)
  │                    │
  │                    ▼
  │         Edge devices run
  │         specific model versions
```

## Database Schema

### Tables with Foreign Key Constraints

```sql
-- Patients table
CREATE TABLE Patient (
    patient_id VARCHAR(50) PRIMARY KEY,
    age_months INTEGER,
    family_history_cf BOOLEAN,
    ethnicity VARCHAR(50),
    gender VARCHAR(10),
    -- ... other clinical attributes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clinical observations table
CREATE TABLE ClinicalObservation (
    observation_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) REFERENCES Patient(patient_id),
    observation_date DATE,
    -- ... clinical attributes matching Patient table
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Genetic markers table
CREATE TABLE GeneticMarker (
    marker_id VARCHAR(50) PRIMARY KEY,
    mutation_name VARCHAR(100) UNIQUE,
    determination VARCHAR(50),
    clinical_significance VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Diagnosis table
CREATE TABLE Diagnosis (
    diagnosis_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) REFERENCES Patient(patient_id),
    cf_diagnosis BOOLEAN,
    diagnosis_probability FLOAT,
    risk_level VARCHAR(20),
    genetic_marker_id VARCHAR(50) REFERENCES GeneticMarker(marker_id),
    diagnosis_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Edge devices table
CREATE TABLE EdgeDevice (
    device_id VARCHAR(50) PRIMARY KEY,
    device_type VARCHAR(50),
    device_name VARCHAR(100),
    last_sync_time TIMESTAMP,
    model_version VARCHAR(50),
    online_status BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models table
CREATE TABLE Model (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100),
    model_path VARCHAR(200),
    training_timestamp TIMESTAMP,
    accuracy FLOAT,
    f1_score FLOAT,
    auc_score FLOAT,
    -- ... other metrics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training sessions table
CREATE TABLE TrainingSession (
    session_id VARCHAR(50) PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES Model(model_id),
    training_start_time TIMESTAMP,
    training_method VARCHAR(50),
    -- ... other training attributes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Federation rounds table
CREATE TABLE FederationRound (
    round_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES TrainingSession(session_id),
    round_number INTEGER,
    start_time TIMESTAMP,
    global_accuracy FLOAT,
    -- ... other round attributes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Data Flow Through Entities

```
Patient Registration
    ↓
Clinical Observation Recording
    ↓
Genetic Marker Association
    ↓
Model-Based Diagnosis
    ↓
Edge Device Sync
    ↓
Federated Learning Round
    ↓
Model Improvement
```

This entity relationship model ensures data integrity while supporting the complex requirements of federated learning with clinical data, including patient privacy, model tracking, and distributed training operations.