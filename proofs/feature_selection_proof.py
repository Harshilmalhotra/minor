import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import os

# SETTINGS
DATA_PATH = "/Users/harshil/Desktop/minor/household_power_consumption.txt"
OUTPUT_DIR = "/Users/harshil/Desktop/minor/proofs/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess():
    print("Step 1: Loading Dataset...")
    df = pd.read_csv(DATA_PATH, sep=';', low_memory=False, nrows=100000) # 100k rows for proof
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df = df.interpolate(method='time').bfill()
    df_hourly = df.resample('1h').mean().dropna()
    
    # Cyclical Features
    df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['day_sin'] = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
    df_hourly['day_cos'] = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)
    
    return df_hourly

def calculate_feature_selection():
    df = load_and_preprocess()
    target = 'Global_active_power'
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]

    print("\nTechnique 1: Pearson Correlation")
    # Formula: r = cov(X,Y) / (std(X) * std(Y))
    corr = df.corr()[target].abs().drop(target)

    print("Technique 2: Mutual Information (MI)")
    # Formula: I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=features)

    print("Technique 3: Random Forest (MDI)")
    # Mean Decrease in Impurity
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=features)

    # Calculate Contribution Value (Normalization)
    # Formula: Normalized_Score = (Score - Min) / (Max - Min)
    scaler = MinMaxScaler()
    
    metrics_df = pd.DataFrame({
        'Correlation': corr,
        'Mutual_Info': mi_series,
        'RF_Importance': rf_importance
    })

    normalized_metrics = pd.DataFrame(
        scaler.fit_transform(metrics_df),
        columns=[f'{c}_Norm' for c in metrics_df.columns],
        index=metrics_df.index
    )

    # Final Contribution Value = Average of Normalized Scores
    normalized_metrics['Contribution_Value'] = normalized_metrics.mean(axis=1)
    
    final_proof = pd.concat([metrics_df, normalized_metrics], axis=1)
    final_proof = final_proof.sort_values(by='Contribution_Value', ascending=False)
    
    print("\n--- FEATURE SELECTION PROOF TABLE ---")
    print(final_proof)
    
    final_proof.to_csv(os.path.join(OUTPUT_DIR, "feature_selection_proof.csv"))
    print(f"\nResults saved to {OUTPUT_DIR}/feature_selection_proof.csv")
    
    # Hit and Trial Simulation logic (simplified for proof)
    print("\n--- HIT AND TRIAL SIMULATION ---")
    subsets = [
        ['Global_intensity'],
        ['Global_intensity', 'Voltage'],
        ['Global_intensity', 'Voltage', 'Sub_metering_3'],
        ['Global_intensity', 'Voltage', 'Sub_metering_3', 'hour_sin', 'hour_cos']
    ]
    
    for i, subset in enumerate(subsets):
        print(f"Trial {i+1}: Features {subset} -> Simulating Training...")
        # In actual scripts, we would train here. This proof script just outlines the logic.

if __name__ == "__main__":
    calculate_feature_selection()
