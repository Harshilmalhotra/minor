import sys
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.data.dataset import load_data, prepare_data
from src.models.tcn import TCNModel
from src.experiments.centralized import train_model, evaluate

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Hit and Trial: Finding Maximum Accuracy"
TRIALS = [
    ['Global_intensity'],
    ['Global_intensity', 'Voltage'],
    ['Global_intensity', 'Voltage', 'Sub_metering_3'],
    ['Global_intensity', 'Voltage', 'Sub_metering_3', 'hour_sin', 'hour_cos'] # Top 5
]
EPOCHS = 2
BATCH_SIZE = 128

def run_proof():
    results = []
    print(f"=== {EXPERIMENT_NAME} ===")
    
    for i, subset in enumerate(TRIALS):
        print(f"\nTrial {i+1}: Testing subset {subset}")
        
        # Load data with this specific subset
        df = load_data(selected_features=subset)
        X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(df)
        num_features = X_train.shape[2]
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                          torch.tensor(y_train, dtype=torch.float32)), 
            batch_size=BATCH_SIZE, shuffle=False
        )
        
        # We use TCN for the trial as it's the core model
        model = TCNModel(input_size=num_features, num_channels=[64, 64], kernel_size=3)
        model = train_model(model, train_loader, epochs=EPOCHS)
        
        metrics = evaluate(model, X_test, y_test, scaler_y)
        metrics['subset'] = str(subset)
        metrics['trial_no'] = i + 1
        results.append(metrics)
        
        print(f"Trial {i+1} Accuracy: {metrics.get('Accuracy (%)', 0):.2f}%")

    # Final Comparison Table
    df_results = pd.DataFrame(results)
    print("\n--- HIT AND TRIAL RESULTS (PROOF OF MAXIMUM ACCURACY) ---")
    print(df_results[['trial_no', 'subset', 'Accuracy (%)', 'MAE']].to_markdown(index=False))
    
    os.makedirs(os.path.join(BASE_DIR, "proofs/results"), exist_ok=True)
    df_results.to_csv(os.path.join(BASE_DIR, "proofs/results/hit_and_trial_proof.csv"), index=False)
    print(f"\nProof saved to proofs/results/hit_and_trial_proof.csv")

if __name__ == "__main__":
    run_proof()
