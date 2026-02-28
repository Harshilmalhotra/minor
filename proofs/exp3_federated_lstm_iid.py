import sys
import os
import pandas as pd

# Path Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.fl.server import main as run_fl

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Exp3: Federated LSTM (IID)"
SELECTED_FEATURES = ['Global_intensity', 'Voltage', 'Sub_metering_3', 'hour_sin', 'hour_cos']
ROUNDS = 5
CLIENTS = 3

def run_experiment():
    print(f"=== {EXPERIMENT_NAME} ===")
    print(f"Features: {SELECTED_FEATURES}")
    
    # Run FL Simulation
    history = run_fl(
        model_type="lstm",
        distribution="iid",
        num_clients=CLIENTS,
        num_rounds=ROUNDS,
        selected_features=SELECTED_FEATURES
    )
    
    # Extract Final Round Metrics
    if history and history.metrics_distributed:
        final_round_metrics = {k: v[-1][1] for k, v in history.metrics_distributed.items()}
        print("\n--- Final Federated Metrics ---")
        for k, v in final_round_metrics.items():
            print(f"{k}: {v:.4f}")
            
        os.makedirs(os.path.join(BASE_DIR, "proofs/results"), exist_ok=True)
        pd.DataFrame([final_round_metrics]).to_csv(os.path.join(BASE_DIR, f"proofs/results/metrics_exp3.csv"), index=False)
        print(f"\nProof saved to proofs/results/metrics_exp3.csv")
    else:
        print("Error: No metrics collected during simulation.")

if __name__ == "__main__":
    run_experiment()
