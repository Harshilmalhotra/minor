import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_plots():
    results_file = 'results/final_demo_results.csv'
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Run experiments first.")
        return

    df = pd.read_csv(results_file)
    output_dir = 'results/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    metrics_to_plot = ['Accuracy (%)', 'MAE', 'RMSE', 'MAPE', 'sMAPE']
    
    # 1. Bar charts for each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Algorithm', y=metric, data=df, palette='viridis')
        plt.title(f'Comparison of {metric} across Algorithms', pad=20)
        plt.xticks(rotation=45, ha='right')
        
        # Add labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), textcoords='offset points')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_").replace("(%)", "").strip()}_comparison.png'), dpi=300)
        plt.close()

    # 2. Combined Matrix (Heatmap view of normalized metrics)
    plt.figure(figsize=(12, 8))
    
    # Normalize data for heatmap (min-max scaling per column)
    df_metrics = df.set_index('Algorithm')[metrics_to_plot]
    
    # For Accuracy, higher is better. For others (Errors), lower is better.
    # We invert error metrics so higher values always mean "better performance" on the heatmap
    normalized_df = df_metrics.copy()
    
    # Accuracy: (val - min) / (max - min)
    normalized_df['Accuracy (%)'] = (df_metrics['Accuracy (%)'] - df_metrics['Accuracy (%)'].min()) / (df_metrics['Accuracy (%)'].max() - df_metrics['Accuracy (%)'].min() + 1e-9)
    
    # Errors: (max - val) / (max - min)
    error_metrics = ['MAE', 'RMSE', 'MAPE', 'sMAPE']
    for err in error_metrics:
        normalized_df[err] = (df_metrics[err].max() - df_metrics[err]) / (df_metrics[err].max() - df_metrics[err].min() + 1e-9)
        
    sns.heatmap(normalized_df, annot=df_metrics, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Relative Performance Score (Higher is Better)'})
    plt.title('Performance Matrix across all Metrics\n(Cell text shows raw values, color shows relative performance)', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_matrix.png'), dpi=300)
    plt.close()

    print(f"Successfully generated {len(metrics_to_plot) + 1} plots in {output_dir}/")

if __name__ == "__main__":
    create_plots()
