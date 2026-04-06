"""
Generate comparative visualization plots from federated algorithm comparison results.

Reads:  results/fed_algorithm_comparison.csv
Saves:  results/plots/fed_algorithm_comparison.png
        results/plots/fed_algorithm_radar.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_bar_comparison(csv_path="results/fed_algorithm_comparison.csv",
                        output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # ── 1. Grouped Bar Chart: Key Metrics ──
    metrics = ["Accuracy (%)", "MAE", "RMSE", "MAPE", "sMAPE"]
    available = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(len(available), 1, figsize=(14, 4 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for ax, metric in zip(axes, available):
        bars = ax.bar(df["Algorithm"], df[metric], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f"Comparison: {metric}", fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, df[metric]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.xticks(rotation=35, ha='right', fontsize=10)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "fed_algorithm_comparison.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅  Bar chart saved to {bar_path}")

    # ── 2. Combined Accuracy + MAE Side-by-Side ──
    if "Accuracy (%)" in df.columns and "MAE" in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy
        bars1 = ax1.barh(df["Algorithm"], df["Accuracy (%)"], color=plt.cm.Greens(np.linspace(0.3, 0.9, len(df))),
                         edgecolor="black", linewidth=0.5)
        ax1.set_xlabel("Accuracy (%)", fontsize=12, fontweight='bold')
        ax1.set_title("Tracking Accuracy by Algorithm", fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars1, df["Accuracy (%)"]):
            if not np.isnan(val):
                ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                         f'{val:.2f}%', va='center', fontsize=9, fontweight='bold')

        # MAE
        bars2 = ax2.barh(df["Algorithm"], df["MAE"], color=plt.cm.Reds(np.linspace(0.3, 0.9, len(df))),
                         edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("MAE", fontsize=12, fontweight='bold')
        ax2.set_title("Mean Absolute Error by Algorithm", fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars2, df["MAE"]):
            if not np.isnan(val):
                ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        side_path = os.path.join(output_dir, "fed_algorithm_acc_mae.png")
        plt.savefig(side_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅  Accuracy/MAE comparison saved to {side_path}")

    # ── 3. Training Time Bar ──
    if "Training Time (s)" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(df["Algorithm"], df["Training Time (s)"],
                      color=plt.cm.Blues(np.linspace(0.3, 0.9, len(df))),
                      edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("Training Time per Algorithm", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, df["Training Time (s)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.xticks(rotation=35, ha='right', fontsize=10)
        plt.tight_layout()
        time_path = os.path.join(output_dir, "fed_algorithm_training_time.png")
        plt.savefig(time_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅  Training time chart saved to {time_path}")


def plot_per_round(csv_path="results/fed_algorithm_per_round.csv",
                   output_dir="results/plots"):
    """Plot per-round convergence curves for each algorithm."""
    if not os.path.exists(csv_path):
        print("  ⚠️  No per-round data found, skipping convergence plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    for metric in ["Accuracy (%)", "MAE", "RMSE"]:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        for alg in df["Algorithm"].unique():
            sub = df[df["Algorithm"] == alg].sort_values("Round")
            ax.plot(sub["Round"], sub[metric], marker='o', label=alg, linewidth=2, markersize=5)

        ax.set_xlabel("Round", fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f"Convergence: {metric} per Round", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"convergence_{metric.replace(' ', '_').replace('(%)', 'pct').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(output_dir, fname)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅  Convergence plot saved to {save_path}")


if __name__ == "__main__":
    plot_bar_comparison()
    plot_per_round()
