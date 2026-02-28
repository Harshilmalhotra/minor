import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import pandas as pd

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.data.dataset import load_data, prepare_data
from src.models.lstm import LSTMModel
from src.experiments.centralized import train_model, evaluate

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Exp1: Centralized LSTM"
# The guide wants to see proof of feature selection. 
# We select the top features based on our feature_selection_proof.py results.
SELECTED_FEATURES = ['Global_intensity', 'Voltage', 'Sub_metering_3', 'hour_sin', 'hour_cos']
EPOCHS = 5
BATCH_SIZE = 64

def run_experiment():
    print(f"=== {EXPERIMENT_NAME} ===")
    print(f"Features: {SELECTED_FEATURES}")
    
    # Proof of Data Loading
    df = load_data(selected_features=SELECTED_FEATURES)
    print(f"Data Shape: {df.shape}")
    
    # Data Preparation
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(df)
    num_features = X_train.shape[2]
    print(f"Input features count: {num_features}")
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                      torch.tensor(y_train, dtype=torch.float32)), 
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Model Setup
    model = LSTMModel(input_size=num_features, hidden_size=64)
    
    # Training (Proof of logic)
    print(f"Training for {EPOCHS} epochs...")
    model = train_model(model, train_loader, epochs=EPOCHS)
    
    # Evaluation (Proof of metrics)
    metrics = evaluate(model, X_test, y_test, scaler_y)
    print("\n--- Final Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save Proof
    os.makedirs(os.path.join(BASE_DIR, "proofs/results"), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(BASE_DIR, f"proofs/results/metrics_exp1.csv"), index=False)
    print(f"\nProof saved to proofs/results/metrics_exp1.csv")

if __name__ == "__main__":
    run_experiment()
