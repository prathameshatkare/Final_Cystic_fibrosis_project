import torch
import os
import sys
from models.cf_tabular import CFTabularNet

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def export_to_onnx():
    # Model parameters (must match the trained model)
    from api.main import get_preprocessing_info
    _, dummy_columns, _, _ = get_preprocessing_info()
    input_dim = len(dummy_columns)
    model_path = "models/cf_tabular_central.pt"
    onnx_path = "models/cf_tabular_edge.onnx"

    print(f"Detected input dimension: {input_dim}")

    import json
    import numpy as np
    scaler, dummy_columns, categorical_cols, numeric_cols = get_preprocessing_info()
    
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "columns": dummy_columns,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }
    
    with open("models/edge_config.json", "w") as f:
        json.dump(scaler_data, f)
    print("Saved edge_config.json with scaler and column info.")

    # Initialize model
    model = CFTabularNet(input_dim=input_dim)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Error: {model_path} not found. Train the model first using 'python experiments/baselines.py'")
        return

    model.eval()

    # Create dummy input for ONNX export (batch_size=1, input_dim=41)
    dummy_input = torch.randn(1, input_dim)

    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=12, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Successfully exported model to {onnx_path}")
    print("This file can now be used on edge devices via ONNX Runtime.")

if __name__ == "__main__":
    export_to_onnx()
