import pandas as pd
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.data.dataset import load_data, prepare_data
from src.fl.server import main as run_fl
from src.experiments.centralized import train_model, evaluate, LSTMModel, TCNModel

# --- EXPERIMENT SETTINGS ---
# We will use 2 epochs and 3 rounds for "Proof-of-Concept" to keep it fast but valid.
EPOCHS = 2
ROUNDS = 3 
CLIENTS = 3
BATCH_SIZE = 128

# Define Feature Sets based on Statistical Ranking (Proof script results)
FEATURE_SETS = {
    "A (Univariate)": [],
    "B (Top 2)": ["Global_intensity", "Voltage"],
    "C (Top 3)": ["Global_intensity", "Voltage", "Sub_metering_3"],
    "D (All)": ["Global_intensity", "Voltage", "Sub_metering_3", "Global_reactive_power", "Sub_metering_1", "Sub_metering_2", "hour_sin", "hour_cos", "day_sin", "day_cos"]
}

MODELS = [
    ("Centralized LSTM", "lstm", "central"),
    ("Centralized TCN", "tcn", "central"),
    ("Federated LSTM (IID)", "lstm", "iid"),
    ("Federated TCN (IID)", "tcn", "iid"),
    ("Federated TCN (Non-IID)", "tcn", "non-iid")
]

def run_controlled_experiments():
    all_results = []
    
    print("🚀 Starting Master Controlled Experiments...")
    print(f"Goal: Evaluate 5 Models across 4 Feature Sets (20 total runs)")
    
    for set_name, features in FEATURE_SETS.items():
        print(f"\n" + "="*50)
        print(f"📊 EVALUATING FEATURE SET: {set_name}")
        print(f"Features included: {['Global_active_power'] + features}")
        print("="*50)
        
        for model_name, model_type, mode in MODELS:
            print(f"\n--- Running: {model_name} ---")
            
            try:
                if mode == "central":
                    # --- Centralized Logic ---
                    df = load_data(selected_features=features)
                    X_train, y_train, X_test, y_test, _, scaler_y = prepare_data(df)
                    num_features = X_train.shape[2]
                    
                    train_loader = DataLoader(
                        TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                      torch.tensor(y_train, dtype=torch.float32)), 
                        batch_size=BATCH_SIZE, shuffle=False
                    )
                    
                    if model_type == "lstm":
                        model = LSTMModel(input_size=num_features)
                    else:
                        model = TCNModel(input_size=num_features, num_channels=[64, 64], kernel_size=3)
                    
                    model = train_model(model, train_loader, epochs=EPOCHS)
                    metrics = evaluate(model, X_test, y_test, scaler_y)
                
                else:
                    # --- Federated Logic ---
                    history = run_fl(
                        model_type=model_type,
                        distribution=mode,
                        num_clients=CLIENTS,
                        num_rounds=ROUNDS,
                        selected_features=features
                    )
                    if history and history.metrics_distributed:
                        # Extract last round metrics
                        metrics = {k: v[-1][1] for k, v in history.metrics_distributed.items()}
                    else:
                        metrics = {"Error": 1}

                # Add metadata for table
                res = {
                    "Feature Set": set_name,
                    "Model": model_name,
                    "MAE": metrics.get("MAE", 0),
                    "RMSE": metrics.get("RMSE", 0),
                    "sMAPE": metrics.get("sMAPE", 0),
                    "Accuracy (%)": metrics.get("Accuracy (%)", 0)
                }
                all_results.append(res)
                print(f"✅ Result: MAE={res['MAE']:.4f}, Acc={res['Accuracy (%)']:.2f}%")
                
            except Exception as e:
                print(f"❌ Error in {model_name}: {e}")
                all_results.append({
                    "Feature Set": set_name,
                    "Model": model_name,
                    "MAE": "ERROR", "RMSE": "ERROR", "sMAPE": "ERROR", "Accuracy (%)": 0
                })

    # Final Table Generation
    df_results = pd.DataFrame(all_results)
    
    # Save results
    os.makedirs(os.path.join(BASE_DIR, "proofs/results"), exist_ok=True)
    df_results.to_csv(os.path.join(BASE_DIR, "proofs/results/controlled_experiment_master.csv"), index=False)
    
    print("\n" + "#"*60)
    print("🏆 FINAL COMPARISON PROOF TABLE")
    print("#"*60)
    print(df_results.to_markdown(index=False))
    print("\n✅ All results saved to proofs/results/controlled_experiment_master.csv")

if __name__ == "__main__":
    run_controlled_experiments()
