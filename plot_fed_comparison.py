import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication-style aesthetics
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 300
})

# Create results plots directory
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load data
df_summary = pd.read_csv("results/fed_algorithm_comparison.csv")
df_rounds = pd.read_csv("results/fed_algorithm_per_round.csv")

# 1. Convergence Metrics (Accuracy & Loss vs Round)
def plot_convergence(df_rounds):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Define distinct colors for 10 algorithms
    colors = sns.color_palette("husl", len(df_rounds['Algorithm'].unique()))
    
    sns.lineplot(data=df_rounds, x='Round', y='Accuracy (%)', hue='Algorithm', marker='o', ax=ax1, palette=colors)
    ax1.set_title("Model Accuracy Convergence over Rounds", fontweight='bold')
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Algorithms")
    
    sns.lineplot(data=df_rounds, x='Round', y='Loss', hue='Algorithm', marker='s', ax=ax2, palette=colors, legend=False)
    ax2.set_title("Training Loss Reduction over Rounds", fontweight='bold')
    ax2.set_ylabel("MSE Loss")
    ax2.set_xlabel("Communication Round")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/convergence_metrics.png", bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved {PLOT_DIR}/convergence_metrics.png")

# 2. Final Error Comparison (MAE & RMSE)
def plot_error_bars(df_summary):
    # Filter out algorithms with NaN metrics
    df_plot = df_summary.dropna(subset=['MAE', 'RMSE'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Reset index for better plotting
    df_melted = df_plot.melt(id_vars='Algorithm', value_vars=['MAE', 'RMSE'], var_name='Metric', value_name='Value')
    
    sns.barplot(data=df_melted, x='Algorithm', y='Value', hue='Metric', ax=ax, palette='viridis')
    ax.set_title("Comparison of Error Metrics (Lower is Better)", fontweight='bold')
    ax.set_ylabel("Error Value")
    ax.set_xlabel("Federated Learning Algorithm")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/error_comparison.png")
    plt.close()
    print(f"  [OK] Saved {PLOT_DIR}/error_comparison.png")

# 3. Accuracy Evolution Heatmap
def plot_heatmap(df_rounds):
    pivot_df = df_rounds.pivot(index='Algorithm', columns='Round', values='Accuracy (%)')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
    plt.title("Algorithm Accuracy Evolution Matrix", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/accuracy_heatmap.png")
    plt.close()
    print(f"  [OK] Saved {PLOT_DIR}/accuracy_heatmap.png")

# 4. Radar Chart (Holistic Comparison)
def plot_radar(df_summary):
    # Select algorithms and normalize metrics for radar (0-1 range)
    metrics = ['Accuracy (%)', 'MAE', 'RMSE', 'sMAPE']
    df_radar = df_summary.dropna(subset=metrics).copy()
    
    # We want higher to be better for all markers on radar, so invert error metrics
    # Normalization: (val - min) / (max - min)
    for m in metrics:
        if m != 'Accuracy (%)':
            # Invert: Higher value = lower error
            df_radar[m] = 1 / (df_radar[m] + 1e-6)
        
        mi, ma = df_radar[m].min(), df_radar[m].max()
        if ma > mi:
            df_radar[m] = (df_radar[m] - mi) / (ma - mi)
        else:
            df_radar[m] = 1.0

    labels = np.array(metrics)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = sns.color_palette("Set2", len(df_radar))
    
    for i, (idx, row) in enumerate(df_radar.iterrows()):
        values = row[metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row['Algorithm'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Holistic Performance Comparison (Normalized)", fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/algorithm_radar.png", bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved {PLOT_DIR}/algorithm_radar.png")

if __name__ == "__main__":
    print("Generating Graphical Representations for Research Paper...")
    plot_convergence(df_rounds)
    plot_error_bars(df_summary)
    plot_heatmap(df_rounds)
    plot_radar(df_summary)
    print("\n[DONE] All plots generated in 'results/plots/'")
