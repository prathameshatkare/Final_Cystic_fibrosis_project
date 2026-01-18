import pandas as pd
import numpy as np
import os

def generate_realistic_cf_data(n_samples=10000, prevalence=0.1):
    np.random.seed(42)
    
    # 1. CF Diagnosis (Target)
    y = np.random.choice([0, 1], size=n_samples, p=[1-prevalence, prevalence])
    
    # 2. Features
    data = {
        "cf_diagnosis": y,
        "age_months": np.random.randint(1, 240, size=n_samples),
        "family_history_cf": np.where(y == 1, 
                                     np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]), # High correlation
                                     np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])),
        "ethnicity": np.random.choice(["Caucasian", "Hispanic", "Black", "Asian", "Other"], size=n_samples),
        "newborn_screening_result": np.where(y == 1, 
                                            np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9]), # Most screened CFers are positive
                                            np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])),
        "meconium_ileus": np.where(y == 1, 
                                  np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]), # 15-20% of CF newborns have it
                                  np.random.choice([0, 1], size=n_samples, p=[0.999, 0.001])),
        "prolonged_jaundice": np.where(y == 1, 
                                      np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]), 
                                      np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])),
        "salty_skin": np.where(y == 1, 
                              np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]), 
                              np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])),
        "cough_type": np.where(y == 1, 
                              np.random.randint(1, 4, size=n_samples), # Persistent/Productive
                              np.random.randint(0, 2, size=n_samples)), # None/Dry
        "respiratory_infections_frequency": np.where(y == 1, 
                                                   np.random.randint(3, 6, size=n_samples), 
                                                   np.random.randint(0, 3, size=n_samples)),
        "wheezing_present": np.where(y == 1, 
                                    np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]), 
                                    np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])),
        "nasal_polyps": np.where(y == 1, 
                                np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]), 
                                np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])),
        "clubbing_fingers": np.where(y == 1, 
                                   np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]), 
                                   np.random.choice([0, 1], size=n_samples, p=[0.999, 0.001])),
        "respiratory_score": np.where(y == 1, 
                                    np.random.randint(6, 11, size=n_samples), 
                                    np.random.randint(0, 5, size=n_samples)),
        "weight_percentile": np.where(y == 1, 
                                    np.random.randint(1, 40, size=n_samples), 
                                    np.random.randint(20, 100, size=n_samples)),
        "height_percentile": np.where(y == 1, 
                                    np.random.randint(1, 50, size=n_samples), 
                                    np.random.randint(30, 100, size=n_samples)),
        "weight_for_height": np.where(y == 1, 
                                     np.random.randint(1, 30, size=n_samples), 
                                     np.random.randint(20, 90, size=n_samples)),
        "growth_faltering": np.where(y == 1, 1, np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])),
        "failure_to_thrive": np.where(y == 1, 
                                     np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]), 
                                     np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])),
        "appetite": np.random.randint(0, 4, size=n_samples),
        "nutritional_risk_score": np.where(y == 1, 
                                         np.random.randint(5, 11, size=n_samples), 
                                         np.random.randint(0, 6, size=n_samples)),
        "stool_character": np.where(y == 1, 
                                   np.random.choice(["Fatty", "Loose", "Hard", "Normal"], size=n_samples, p=[0.6, 0.3, 0.05, 0.05]),
                                   np.random.choice(["Fatty", "Loose", "Hard", "Normal"], size=n_samples, p=[0.05, 0.1, 0.1, 0.75])),
        "stool_frequency": np.where(y == 1, 
                                   np.random.randint(3, 7, size=n_samples), 
                                   np.random.randint(1, 4, size=n_samples)),
        "abdominal_distention": np.where(y == 1, 1, 0),
        "diarrhea_chronic": np.where(y == 1, 1, 0),
        "fat_malabsorption_signs": np.where(y == 1, np.random.randint(2, 4, size=n_samples), 0),
        "sweat_test_simulated": np.where(y == 1, 
                                        np.random.normal(80, 15, size=n_samples), # CF usually > 60
                                        np.random.normal(20, 10, size=n_samples)), # Normal < 30
        "cf_clinical_suspicion_index": np.where(y == 1, 
                                              np.random.normal(0.8, 0.1, size=n_samples), 
                                              np.random.normal(0.2, 0.2, size=n_samples)),
        "cough_character": np.random.choice(["Wet", "Dry", "None"], size=n_samples),
        "diagnostic_confidence": 0.95
    }
    
    df = pd.DataFrame(data)
    # Clip sweat test
    df["sweat_test_simulated"] = df["sweat_test_simulated"].clip(5, 150).round(1)
    df["cf_clinical_suspicion_index"] = df["cf_clinical_suspicion_index"].clip(0, 1).round(3)
    
    # Add some "Ambiguous" cases to make accuracy realistic (not 100%)
    noise_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    df.loc[noise_indices, "cf_diagnosis"] = 1 - df.loc[noise_indices, "cf_diagnosis"]
    
    return df

if __name__ == "__main__":
    df = generate_realistic_cf_data()
    path = "data/synthetic_cystic_fibrosis_dataset.csv"
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} samples with {df['cf_diagnosis'].sum()} CF cases ({(df['cf_diagnosis'].mean()*100):.1f}% prevalence)")
