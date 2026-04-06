import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Patch
from scipy.stats import norm

# ---------------------------------------------------------------------
#  Configuration & Paper Style
# ---------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': '#333333',
    'lines.linewidth': 2
})

PLOT_DIR = "results/paper_graphics"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
#  Data Preparation
# ---------------------------------------------------------------------
def load_and_clean_data():
    print("Loading data for visualization...")
    df_comp = pd.read_csv("results/fed_algorithm_comparison.csv")
    df_round = pd.read_csv("results/fed_algorithm_per_round.csv")
    
    # Handle non-numeric "Parallel" in Training Time
    df_comp['Training Time (s)'] = pd.to_numeric(df_comp['Training Time (s)'], errors='coerce')
    # For visualization, we fill NaN (Parallel) with 1.2x the Max (or 0 if all are NaN)
    max_time = df_comp['Training Time (s)'].max()
    if pd.isna(max_time): max_time = 300 # Default
    df_comp['Time_Numeric'] = df_comp['Training Time (s)'].fillna(max_time * 1.5)
    
    # Replace Convergence Round (>90%) 'N/A' with Max Round + 1
    max_rnd = df_round['Round'].max()
    df_comp['Conv_Round'] = pd.to_numeric(df_comp['Convergence Round (>90%)'], errors='coerce').fillna(max_rnd + 1)
    
    return df_comp, df_round

# ---------------------------------------------------------------------
#  1. Convergence Dynamics (Fig1)
# ---------------------------------------------------------------------
def plot_convergence_hq(df_round):
    print("Generating Fig 1: Convergence Dynamics...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    
    palette = sns.color_palette("husl", df_round['Algorithm'].nunique())
    
    # Plot Accuracy
    sns.lineplot(data=df_round, x='Round', y='Accuracy (%)', hue='Algorithm', 
                 palette=palette, marker='o', markersize=6, ax=ax1, alpha=0.9)
    ax1.set_title("Evolution of Tracking Accuracy across Communication Rounds", pad=15)
    ax1.set_ylabel("Tracking Accuracy (%)")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Plot Loss (Log Scale for better resolution of differences)
    sns.lineplot(data=df_round, x='Round', y='Loss', hue='Algorithm', 
                 palette=palette, marker='s', markersize=6, ax=ax2, alpha=0.9, legend=False)
    ax2.set_yscale('log')
    ax2.set_title("Training Loss Convergence (Log Scale)", pad=15)
    ax2.set_ylabel("MSE Loss (Log)")
    ax2.set_xlabel("Number of Communication Rounds")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fig1_convergence_hq.png", bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------
#  2. Performance Ranking Heatmap (Fig2)
# ---------------------------------------------------------------------
def plot_performance_heatmap(df_comp):
    print("Generating Fig 2: Performance Matrix Heatmap...")
    metrics = ['Accuracy (%)', 'MAE', 'RMSE', 'MAPE', 'sMAPE']
    plot_df = df_comp.set_index('Algorithm')[metrics].copy()
    
    # Normalize (0-1) where 1 is BEST (so invert error metrics)
    norm_df = plot_df.copy()
    for col in metrics:
        if col == 'Accuracy (%)':
            norm_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min() + 1e-9)
        else:
            # 1 - normalized error (higher is better)
            norm_df[col] = 1 - (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min() + 1e-9)
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_df, annot=plot_df, fmt=".3f", cmap="YlGnBu", 
                linewidths=.5, cbar_kws={'label': 'Relative Performance (1=Best, 0=Worst)'})
    plt.title("Algorithm Performance Scorecard", pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fig2_performance_heatmap.png")
    plt.close()

# ---------------------------------------------------------------------
#  3. Radar Comparison (Fig3)
# ---------------------------------------------------------------------
def plot_radar_advanced(df_comp):
    print("Generating Fig 3: Holistic Radar Chart...")
    metrics = ['Accuracy (%)', 'MAE', 'RMSE', 'sMAPE']
    df_radar = df_comp.set_index('Algorithm')[metrics].copy()
    
    # Normalize for Radar (0 to 1, Higher is Better)
    for col in metrics:
        if col == 'Accuracy (%)':
            mi, ma = df_radar[col].min(), df_radar[col].max()
            df_radar[col] = (df_radar[col] - mi) / (ma - mi + 1e-9)
        else:
            # Invert error: (max - val) / (max - min)
            mi, ma = df_radar[col].min(), df_radar[col].max()
            df_radar[col] = (ma - df_radar[col]) / (ma - mi + 1e-9)

    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    palette = sns.color_palette("Set1", len(df_radar))
    
    for i, (idx, row) in enumerate(df_radar.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=idx, color=palette[i])
        ax.fill(angles, values, alpha=0.1, color=palette[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
    # Adjust Legend - pick top 5 or use a clean layout
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Holistic Performance Comparison (Normalized)", pad=30, fontweight='bold')
    plt.savefig(f"{PLOT_DIR}/fig3_algorithm_radar.png", bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------
#  4. Pareto Frontier (Accuracy vs Convergence) (Fig4)
# ---------------------------------------------------------------------
def plot_pareto_analysis(df_comp):
    print("Generating Fig 4: Pareto Efficiency Plot...")
    plt.figure(figsize=(10, 7))
    
    x = df_comp['Conv_Round']
    y = df_comp['Accuracy (%)']
    labels = df_comp['Algorithm']
    
    # Scatter plot
    sns.scatterplot(x=x, y=y, size=df_comp['RMSE'], hue=labels, 
                    sizes=(50, 400), palette='viridis', alpha=0.8, legend=True)
    
    # Annotate points
    for i, txt in enumerate(labels):
        plt.annotate(txt, (x[i], y[i]), xytext=(7, 7), textcoords='offset points', fontsize=9)
        
    plt.title("Efficiency Frontier: Accuracy vs. Convergence Speed", pad=15)
    plt.xlabel("Communication Rounds to Converge (>90%)")
    plt.ylabel("Final Tracking Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(title="Algorithm (Size=RMSE)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fig4_pareto_efficiency.png", bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------
#  5. Stability Analysis (Fig5)
# ---------------------------------------------------------------------
def plot_stability_violins(df_round):
    print("Generating Fig 5: Performance Stability Violiins...")
    plt.figure(figsize=(12, 7))
    
    sns.violinplot(data=df_round, x='Algorithm', y='Accuracy (%)', 
                   inner="quartile", palette="muted", alpha=0.8)
    sns.stripplot(data=df_round, x='Algorithm', y='Accuracy (%)', 
                  color="black", size=4, alpha=0.3)
    
    plt.title("Performance Stability across Rounds", pad=15)
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy Range (%)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fig5_stability_distribution.png")
    plt.close()

# ---------------------------------------------------------------------
#  6. Regression Error Characteristic (REC) Approximation (Fig6)
# ---------------------------------------------------------------------
def plot_rec_approximation(df_comp):
    """
    Simulates the REC curve (Regression equivalent of ROC) based on RMSE/MAE metrics.
    Accuracy(tolerance) = P(|error| < tolerance)
    We approximate the error distribution as Normal(0, RMSE).
    """
    print("Generating Fig 6: REC Curve (Regression ROC equivalent)...")
    plt.figure(figsize=(10, 7))
    
    tolerances = np.linspace(0, 1.5, 100) # Tolerance range for error
    palette = sns.color_palette("tab10", len(df_comp))
    
    for i, row in df_comp.iterrows():
        # Using RMSE to model the CDF: P(|X| < t) = 2*Phi(t/sigma) - 1
        rmse = row['RMSE']
        acc_at_tol = 2 * norm.cdf(tolerances / (rmse + 1e-9)) - 1
        acc_at_tol = np.maximum(acc_at_tol, 0) * 100
        
        plt.plot(tolerances, acc_at_tol, label=row['Algorithm'], color=palette[i], linewidth=2.5)
        
    plt.title("Regression Error Characteristic (REC) Curve", pad=15)
    plt.xlabel("Predictive Error Tolerance (ε)")
    plt.ylabel("Accuracy Within Tolerance (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fig6_rec_curve.png", bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------
#  7. Metric Correlation Heatmap (Fig7)
# ---------------------------------------------------------------------
def plot_metric_correlations(df_comp):
    print("Generating Fig 7: Metric Correlation Heatmap...")
    metrics = ['Accuracy (%)', 'MAE', 'RMSE', 'MAPE', 'sMAPE', 'Final Loss']
    corr = df_comp[metrics].corr()
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True)
    plt.title("Inter-Metric Correlation Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fig7_metric_correlations.png")
    plt.close()

# ---------------------------------------------------------------------
#  8. Comparative Performance Table (Fig 8 / Tab 1)
# ---------------------------------------------------------------------
def generate_comparison_table(df_comp):
    print("Generating Comparative Performance Table (MD & TeX)...")
    
    # Select and rename columns for the paper
    cols = {
        'Algorithm': 'Algorithm',
        'Accuracy (%)': 'Tracking Acc. (%)',
        'MAE': 'MAE',
        'RMSE': 'RMSE',
        'sMAPE': 'sMAPE (%)',
        'Convergence Round (>90%)': 'Conv. Round',
        'Training Time (s)': 'Comp. Time (s)'
    }
    
    table_df = df_comp[list(cols.keys())].copy()
    table_df = table_df.rename(columns=cols)
    
    # Format decimals
    for col in table_df.columns:
        if col == 'Algorithm' or col == 'Conv. Round' or col == 'Comp. Time (s)':
            continue
        table_df[col] = table_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Clean NaNs for presentation
    table_df = table_df.fillna("N/A")

    # Save as Markdown
    md_path = f"{PLOT_DIR}/comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Federated Learning Algorithm Comparison Table\n\n")
        f.write(table_df.to_markdown(index=False))
    
    # Save as LaTeX (for direct use in paper)
    tex_path = f"{PLOT_DIR}/comparison_table.tex"
    with open(tex_path, "w") as f:
        f.write(table_df.to_latex(index=False, bold_rows=True, column_format='lcccccc'))
        
    print(f"  [OK] Saved table to {md_path} and {tex_path}")

# ---------------------------------------------------------------------
#  Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        df_comp, df_round = load_and_clean_data()
        
        plot_convergence_hq(df_round)
        plot_performance_heatmap(df_comp)
        plot_radar_advanced(df_comp)
        plot_pareto_analysis(df_comp)
        plot_stability_violins(df_round)
        plot_rec_approximation(df_comp)
        plot_metric_correlations(df_comp)
        generate_comparison_table(df_comp)
        
        print(f"\n[SUCCESS] {len(os.listdir(PLOT_DIR))} items saved to '{PLOT_DIR}/'")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
