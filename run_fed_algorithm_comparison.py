"""
Run all 10 Federated Learning Algorithms with Cascade LSTM model
on the Algorithmic data distribution and produce a detailed comparison.

This script runs FL simulation IN-PROCESS (no ray dependency) by manually
executing client training and server aggregation loops.

Algorithms: FedAvg, FedProx, FedNova, SCAFFOLD, FedOpt,
            FedAdam, FedYogi, FedAdagrad, FedDyn, Clustered FL

Output:  results/fed_algorithm_comparison.csv
         results/fed_algorithm_per_round.csv
"""

import io
import sys
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import time
import traceback
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath('.'))

from src.data.dataset import load_data, prepare_data
from src.data.split import create_algorithmic_splits
from src.models.cascade_lstm import CascadeLSTMModel
from src.experiments.metrics import evaluate_metrics

# ---------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------
MODEL_TYPE = "cascade"           # Cascade LSTM
DISTRIBUTION = "algorithmic"     # Algorithmic node data generation
NUM_CLIENTS = 5
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.001

ALGORITHMS = [
    ("FedAvg",       "fedavg"),
    ("FedProx",      "fedprox"),
    ("FedNova",      "fednova"),
    ("SCAFFOLD",     "scaffold"),
    ("FedOpt",       "fedopt"),
    ("FedAdam",      "fedadam"),
    ("FedYogi",      "fedyogi"),
    ("FedAdagrad",   "fedadagrad"),
    ("FedDyn",       "feddyn"),
    ("Clustered FL", "clustered"),
]


# ---------------------------------------------------------------
#  Helper: get / set model parameters as list of numpy arrays
# ---------------------------------------------------------------
def get_params(model):
    return [p.data.cpu().numpy().copy() for p in model.parameters()]

def set_params(model, params):
    with torch.no_grad():
        for p, w in zip(model.parameters(), params):
            p.copy_(torch.tensor(w))


# ---------------------------------------------------------------
#  Local client training
# ---------------------------------------------------------------
def train_client(model, train_data, global_params, strategy, state,
                 epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    """
    Train a local model and return updated params + number of local steps.
    Modifies training based on strategy:
      - fedprox:  adds proximal regularization
      - scaffold: applies control-variate correction to gradients
      - feddyn:   adds dynamic regularization
      - others:   standard SGD/Adam
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=False)

    global_tensors = [torch.tensor(w, dtype=torch.float32) for w in global_params]
    local_steps = 0

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)

            # -- FedProx: proximal term --
            if strategy == "fedprox":
                prox = 0.0
                for p, gp in zip(model.parameters(), global_tensors):
                    prox += torch.sum((p - gp) ** 2)
                loss = loss + (0.1 / 2.0) * prox

            # -- FedDyn: dynamic regularization --
            if strategy == "feddyn":
                reg = 0.0
                for p, gp in zip(model.parameters(), global_tensors):
                    reg += torch.sum((p - gp) ** 2)
                loss = loss + (0.01 / 2.0) * reg

            loss.backward()

            # -- SCAFFOLD: control variate correction --
            if strategy == "scaffold" and state.get("server_control") is not None:
                with torch.no_grad():
                    sc = state["server_control"]
                    lc = state.get("local_control", [np.zeros_like(s) for s in sc])
                    for i, p in enumerate(model.parameters()):
                        if i < len(sc):
                            correction = torch.tensor(sc[i] - lc[i], dtype=p.grad.dtype)
                            p.grad.add_(correction)

            optimizer.step()
            local_steps += 1

    # Update SCAFFOLD local control
    if strategy == "scaffold":
        new_params = get_params(model)
        sc = state.get("server_control") or [np.zeros_like(w) for w in global_params]
        new_lc = []
        for i in range(len(new_params)):
            delta = (global_params[i] - new_params[i]) / (local_steps * lr + 1e-10)
            c_new = delta - sc[i] if i < len(sc) else delta
            new_lc.append(c_new)
        state["local_control"] = new_lc

    return get_params(model), local_steps


# ---------------------------------------------------------------
#  Server-side aggregation
# ---------------------------------------------------------------
def weighted_average(client_results):
    total = sum(n for _, n in client_results)
    n_params = len(client_results[0][0])
    return [
        np.sum([w[i] * n for w, n in client_results], axis=0) / total
        for i in range(n_params)
    ]


def aggregate(strategy, client_results, server_state, round_idx):
    """
    Perform server-side aggregation based on strategy.
    client_results: list of (params_list, num_examples, local_steps)
    Returns new global params.
    """
    n_clients = len(client_results)
    wr = [(params, n) for params, n, _ in client_results]

    if strategy == "fedavg" or strategy == "fedprox":
        return weighted_average(wr)

    elif strategy == "fednova":
        # Normalize by local steps, then re-scale
        tau_eff = 0.0
        total_n = sum(n for _, n, _ in client_results)
        normalized = []
        for params, n, steps in client_results:
            normed = [w / max(steps, 1) for w in params]
            normalized.append((normed, n))
            tau_eff += n * steps
        tau_eff /= total_n
        avg = weighted_average(normalized)
        return [w * tau_eff for w in avg]

    elif strategy == "scaffold":
        avg = weighted_average(wr)
        sc = server_state.get("server_control")
        if sc is None:
            server_state["server_control"] = [np.zeros_like(w) for w in avg]
            return avg
        N = n_clients
        for i in range(len(sc)):
            deltas = np.mean([wr[j][0][i] - avg[i] for j in range(N)], axis=0)
            sc[i] += deltas / N
        corrected = [avg[i] - 0.1 * sc[i] for i in range(len(avg))]
        return corrected

    elif strategy == "fedopt":
        avg = weighted_average(wr)
        prev = server_state.get("prev")
        if prev is None:
            server_state["prev"] = avg
            server_state["vel"] = [np.zeros_like(w) for w in avg]
            return avg
        vel = server_state["vel"]
        pg = [avg[i] - prev[i] for i in range(len(avg))]
        for i in range(len(vel)):
            vel[i] = 0.9 * vel[i] + pg[i]
        new_w = [prev[i] + 1.0 * vel[i] for i in range(len(avg))]
        server_state["prev"] = new_w
        return new_w

    elif strategy == "fedadam":
        avg = weighted_average(wr)
        prev = server_state.get("prev")
        if prev is None:
            server_state.update({"prev": avg, "m": [np.zeros_like(w) for w in avg],
                                 "v": [np.zeros_like(w) for w in avg], "t": 0})
            return avg
        t = server_state["t"] + 1; server_state["t"] = t
        m, v = server_state["m"], server_state["v"]
        pg = [avg[i] - prev[i] for i in range(len(avg))]
        new_w = []
        for i in range(len(avg)):
            m[i] = 0.9 * m[i] + 0.1 * pg[i]
            v[i] = 0.999 * v[i] + 0.001 * (pg[i] ** 2)
            mh = m[i] / (1 - 0.9 ** t)
            vh = v[i] / (1 - 0.999 ** t)
            new_w.append(prev[i] + 0.01 * mh / (np.sqrt(vh) + 1e-8))
        server_state["prev"] = new_w
        return new_w

    elif strategy == "fedyogi":
        avg = weighted_average(wr)
        prev = server_state.get("prev")
        if prev is None:
            server_state.update({"prev": avg, "m": [np.zeros_like(w) for w in avg],
                                 "v": [np.ones_like(w) * 1e-8 for w in avg], "t": 0})
            return avg
        t = server_state["t"] + 1; server_state["t"] = t
        m, v = server_state["m"], server_state["v"]
        pg = [avg[i] - prev[i] for i in range(len(avg))]
        new_w = []
        for i in range(len(avg)):
            m[i] = 0.9 * m[i] + 0.1 * pg[i]
            gs = pg[i] ** 2
            v[i] = v[i] - (1 - 0.999) * gs * np.sign(v[i] - gs)
            v[i] = np.maximum(v[i], 1e-8)
            mh = m[i] / (1 - 0.9 ** t)
            new_w.append(prev[i] + 0.01 * mh / (np.sqrt(v[i]) + 1e-8))
        server_state["prev"] = new_w
        return new_w

    elif strategy == "fedadagrad":
        avg = weighted_average(wr)
        prev = server_state.get("prev")
        if prev is None:
            server_state.update({"prev": avg, "acc_sq": [np.zeros_like(w) for w in avg]})
            return avg
        asq = server_state["acc_sq"]
        pg = [avg[i] - prev[i] for i in range(len(avg))]
        new_w = []
        for i in range(len(avg)):
            asq[i] += pg[i] ** 2
            new_w.append(prev[i] + 0.1 * pg[i] / (np.sqrt(asq[i]) + 1e-8))
        server_state["prev"] = new_w
        return new_w

    elif strategy == "feddyn":
        avg = weighted_average(wr)
        h = server_state.get("h")
        prev_g = server_state.get("prev_global")
        if h is None:
            server_state["h"] = [np.zeros_like(w) for w in avg]
            server_state["prev_global"] = [np.copy(w) for w in avg]
            return avg
        alpha = 0.01
        for i in range(len(h)):
            h[i] -= alpha * (avg[i] - prev_g[i])
        corrected = [avg[i] - (1.0 / (alpha * n_clients + 1e-10)) * h[i] for i in range(len(avg))]
        server_state["prev_global"] = [np.copy(w) for w in corrected]
        return corrected

    elif strategy == "clustered":
        from sklearn.cluster import KMeans
        flat_vecs = []
        for params, _, _ in client_results:
            flat_vecs.append(np.concatenate([w.flatten() for w in params]))
        flat_vecs = np.array(flat_vecs)
        n_clusters = min(2, len(client_results))
        if n_clusters < 2:
            return weighted_average(wr)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = km.fit_predict(flat_vecs)
        cluster_weights, cluster_sizes = [], []
        for c_id in range(n_clusters):
            members = [i for i, l in enumerate(labels) if l == c_id]
            if not members:
                continue
            cluster_data = [(wr[i][0], wr[i][1]) for i in members]
            cluster_avg = weighted_average(cluster_data)
            cluster_total = sum(wr[i][1] for i in members)
            cluster_weights.append(cluster_avg)
            cluster_sizes.append(cluster_total)
        total = sum(cluster_sizes)
        return [
            np.sum([cw[i] * cs for cw, cs in zip(cluster_weights, cluster_sizes)], axis=0) / total
            for i in range(len(cluster_weights[0]))
        ]

    else:
        return weighted_average(wr)


# ---------------------------------------------------------------
#  Evaluate global model
# ---------------------------------------------------------------
def evaluate_global(model, X_test, y_test, scaler_y):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_t)
        loss = nn.MSELoss()(preds, y_t).item()
    preds_inv = scaler_y.inverse_transform(preds.numpy())
    y_true_inv = scaler_y.inverse_transform(y_t.numpy())
    metrics = evaluate_metrics(y_true_inv, preds_inv)
    return float(loss), {k: float(v) for k, v in metrics.items()}


# ---------------------------------------------------------------
#  Run one full FL experiment
# ---------------------------------------------------------------
def run_single_experiment(strategy_key, client_data, X_test, y_test,
                          scaler_y, num_features, num_rounds=NUM_ROUNDS):
    # Initialize global model
    global_model = CascadeLSTMModel(input_size=num_features)
    global_params = get_params(global_model)

    server_state = {}   # Strategy-specific server state
    client_states = [{} for _ in range(NUM_CLIENTS)]  # Per-client state for SCAFFOLD

    round_metrics_list = []

    for rnd in range(1, num_rounds + 1):
        client_results = []

        for cid in range(NUM_CLIENTS):
            # Create a fresh local model and load global params
            local_model = CascadeLSTMModel(input_size=num_features)
            set_params(local_model, global_params)

            # Merge server state into client state for SCAFFOLD
            if strategy_key == "scaffold":
                client_states[cid]["server_control"] = server_state.get("server_control")

            # Local training
            updated_params, local_steps = train_client(
                local_model, client_data[cid], global_params,
                strategy_key, client_states[cid]
            )
            n_examples = len(client_data[cid][0])
            client_results.append((updated_params, n_examples, local_steps))

        # Server aggregation
        global_params = aggregate(strategy_key, client_results, server_state, rnd)

        # Evaluate
        set_params(global_model, global_params)
        loss, metrics = evaluate_global(global_model, X_test, y_test, scaler_y)

        round_metrics_list.append({
            "Round": rnd,
            "Loss": loss,
            **metrics
        })

        acc_str = f"{metrics.get('Accuracy (%)', 0):.2f}%"
        mae_str = f"{metrics.get('MAE', 0):.4f}"
        print(f"    Round {rnd}/{num_rounds} | Acc: {acc_str} | MAE: {mae_str} | Loss: {loss:.6f}")

    return round_metrics_list


# ---------------------------------------------------------------
#  Main comparison
# ---------------------------------------------------------------
def run_comparison():
    print("=" * 80)
    print("  FEDERATED ALGORITHM COMPARISON -- Cascade LSTM on Algorithmic Nodes")
    print("=" * 80)
    print(f"  Model: {MODEL_TYPE.upper()} | Distribution: {DISTRIBUTION}")
    print(f"  Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS} | Local Epochs: {LOCAL_EPOCHS}")
    print("=" * 80)

    # Load and prepare data
    print("\n  Loading dataset...")
    df = load_data()
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(df)
    num_features = X_train.shape[2]
    print(f"  Data loaded: {X_train.shape[0]} train / {X_test.shape[0]} test samples, {num_features} features")

    # Create algorithmic client splits
    client_data = create_algorithmic_splits(X_train, y_train, NUM_CLIENTS)
    print(f"  Created {NUM_CLIENTS} algorithmic client splits\n")

    all_final_results = []
    all_per_round = []

    for alg_name, strategy_key in ALGORITHMS:
        print(f"\n{'-' * 60}")
        print(f"  >> Running: {alg_name} (strategy={strategy_key})")
        print(f"{'-' * 60}")

        t_start = time.time()

        try:
            round_metrics = run_single_experiment(
                strategy_key, client_data, X_test, y_test, scaler_y, num_features
            )
        except Exception as e:
            print(f"  [FAIL] {alg_name} failed: {e}")
            traceback.print_exc()
            round_metrics = []

        elapsed = time.time() - t_start

        # Final metrics = last round
        if round_metrics:
            final = round_metrics[-1].copy()
            del final["Round"]
            del final["Loss"]

            # Best metrics across rounds
            best_acc = max(r["Accuracy (%)"] for r in round_metrics)
            best_mae = min(r["MAE"] for r in round_metrics)
            best_rmse = min(r["RMSE"] for r in round_metrics)

            # Convergence round (first time Accuracy > 90%)
            conv_round = None
            for r in round_metrics:
                if r["Accuracy (%)"] >= 90.0:
                    conv_round = r["Round"]
                    break

            final_result = {
                "Algorithm": alg_name,
                **final,
                "Final Loss": round_metrics[-1]["Loss"],
                "Best Accuracy (%)": best_acc,
                "Best MAE": best_mae,
                "Best RMSE": best_rmse,
                "Convergence Round (>90%)": conv_round if conv_round else "N/A",
                "Rounds Completed": len(round_metrics),
                "Training Time (s)": round(elapsed, 2),
            }
        else:
            final_result = {
                "Algorithm": alg_name,
                "Accuracy (%)": np.nan, "MAE": np.nan, "RMSE": np.nan,
                "MAPE": np.nan, "sMAPE": np.nan,
                "Final Loss": np.nan,
                "Best Accuracy (%)": np.nan, "Best MAE": np.nan, "Best RMSE": np.nan,
                "Convergence Round (>90%)": "N/A",
                "Rounds Completed": 0,
                "Training Time (s)": round(elapsed, 2),
            }

        all_final_results.append(final_result)

        # Collect per-round data
        for r in round_metrics:
            all_per_round.append({"Algorithm": alg_name, **r})

        acc = final_result.get("Accuracy (%)", np.nan)
        mae = final_result.get("MAE", np.nan)
        print(f"  [OK] {alg_name}: Accuracy={acc:.2f}%, MAE={mae:.4f}, Time={elapsed:.1f}s")

    # ------ Save results ------
    os.makedirs("results", exist_ok=True)

    df_final = pd.DataFrame(all_final_results)
    cols_order = [
        "Algorithm", "Accuracy (%)", "MAE", "RMSE", "MAPE", "sMAPE",
        "Final Loss", "Best Accuracy (%)", "Best MAE", "Best RMSE",
        "Convergence Round (>90%)", "Rounds Completed", "Training Time (s)"
    ]
    cols_order = [c for c in cols_order if c in df_final.columns]
    df_final = df_final[cols_order]
    df_final.to_csv("results/fed_algorithm_comparison.csv", index=False)

    if all_per_round:
        df_rounds = pd.DataFrame(all_per_round)
        df_rounds.to_csv("results/fed_algorithm_per_round.csv", index=False)

    # Print table
    print("\n")
    print("=" * 100)
    print("  FINAL COMPARISON RESULTS")
    print("=" * 100)
    try:
        print(df_final.to_markdown(index=False, floatfmt=".4f"))
    except Exception:
        print(df_final.to_string(index=False))

    print(f"\n[DONE] Results saved to results/fed_algorithm_comparison.csv")
    if all_per_round:
        print(f"[DONE] Per-round data saved to results/fed_algorithm_per_round.csv")

    return df_final


if __name__ == "__main__":
    run_comparison()
