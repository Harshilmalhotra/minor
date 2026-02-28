# 🛡️ Project Verification & Code Proofs Guide

This directory contains standalone scripts designed for **line-by-line verification** of the research methodology. Each script corresponds to a specific claim in the research paper.

## 1. Feature Selection Proof (`feature_selection_proof.py`)

**Goal:** Verify why specific features were chosen.

- **Verification Point:** Lines 34-45.
- **Methods:** Pearson Correlation, Mutual Information, and Random Forest Gini Importance.
- **Output:** `proofs/results/feature_selection_proof.csv` containing the mathematical ranking.

## 2. Methodology Experiments (`master_experiment.py`)

**Goal:** Prove which features work best for which models (Hit and Trial).

- **Verification Point:** Lines 22-35 (Feature Set Definitions).
- **Logic:** Runs 20 controlled experiments comparing 4 subsets across 5 model configurations.
- **Models Verified:**
  - Centralized LSTM & TCN
  - Federated LSTM & TCN (IID)
  - Federated TCN (Non-IID)
- **Output:** `proofs/results/controlled_experiment_master.csv`

## 3. Individual Technique Proofs

If the guide wants to run a single technique in isolation:

- `exp1_centralized_lstm.py`
- `exp2_centralized_tcn.py`
- `exp3_federated_lstm_iid.py`
- `exp4_federated_tcn_iid.py`
- `exp5_federated_tcn_non_iid.py`

## 4. Mathematical Soundness

- **Dataset Preprocessing:** Verified in `src/data/dataset.py`.
- **TCN Architecture:** Verified in `src/models/tcn.py` (Causal convolutions and dilated skip connections).
- **FL Implementation:** Verified in `src/fl/server.py` using the Flower (flwr) framework.

---

**Verification Instructions for Guide:**

1. Open the script in any editor.
2. Review the `load_data` and `prepare_data` calls.
3. Observe the `train_model` or `run_fl` parameters.
4. Execute via terminal: `python3 proofs/<script_name>.py`
5. Compare the terminal output metrics with the values reported in the paper tables.
