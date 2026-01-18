import json
import numpy as np
import onnxruntime as ort
import os

class CFEdgePredictor:
    def __init__(self, model_path="models/cf_tabular_edge.onnx", config_path="models/edge_config.json"):
        # Load config (scaler and columns)
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        print(f"Edge Predictor Initialized with {len(self.config['columns'])} features.")

    def preprocess(self, patient_data):
        # 1. Convert to dict if not already
        # 2. Handle categorical columns (manual get_dummies logic)
        processed_data = {}
        
        # Default all columns to 0
        for col in self.config["columns"]:
            processed_data[col] = 0.0
            
        # Fill in numeric values
        for col in self.config["numeric_cols"]:
            processed_data[col] = float(patient_data.get(col, 0))
            
        # Handle categorical (one-hot)
        for col in self.config["categorical_cols"]:
            val = patient_data.get(col)
            if val:
                dummy_col = f"{col}_{val}"
                if dummy_col in processed_data:
                    processed_data[dummy_col] = 1.0
        
        # Convert to numpy array in correct order
        ordered_values = [processed_data[col] for col in self.config["columns"]]
        x = np.array([ordered_values], dtype=np.float32)
        
        # Scale values using saved mean/scale
        mean = np.array(self.config["mean"], dtype=np.float32)
        scale = np.array(self.config["scale"], dtype=np.float32)
        x = (x - mean) / scale
        
        return x

    def predict(self, patient_data):
        x = self.preprocess(patient_data)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: x})
        logits = outputs[0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        cf_probability = float(probs[0][1])
        risk_level = "High" if cf_probability > 0.7 else "Moderate" if cf_probability > 0.3 else "Low"
        
        return {
            "cf_probability": cf_probability,
            "risk_level": risk_level
        }

if __name__ == "__main__":
    # Example usage on an edge device
    predictor = CFEdgePredictor()
    
    # Sample patient data (from a nurse's input)
    sample_patient = {
        "age_months": 12,
        "family_history_cf": 1,
        "salty_skin": 1,
        "sweat_test_simulated": 75,
        "cough_type": 2,
        "ethnicity": "Caucasian"
    }
    
    result = predictor.predict(sample_patient)
    print("\n--- Diagnostic Result (Edge Inference) ---")
    print(f"CF Probability: {result['cf_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
