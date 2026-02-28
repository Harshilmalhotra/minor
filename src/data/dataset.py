import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath="/Users/harshil/Desktop/minor/household_power_consumption.txt", selected_features=None):
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    
    # Handle missing values
    df = df.interpolate(method='time')
    df = df.bfill()
    
    # Resample to hourly data
    df_hourly = df.resample('1h').mean()
    
    # Add cyclical time features
    df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['day_sin'] = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
    df_hourly['day_cos'] = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)
    df_hourly['month_sin'] = np.sin(2 * np.pi * df_hourly.index.month / 12)
    df_hourly['month_cos'] = np.cos(2 * np.pi * df_hourly.index.month / 12)
    
    target = 'Global_active_power'
    if selected_features is None:
        cols = [target] + [c for c in df_hourly.columns if c != target]
    else:
        # Ensure target is always included as the first column
        feats = list(selected_features)
        if target in feats: feats.remove(target)
        cols = [target] + feats
        
    return df_hourly[cols]

def create_sequences(data, lookback=24):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def prepare_data(df, train_ratio=0.8, lookback=24):
    data = df.values
    target_idx = 0  # Global_active_power is the first column
    
    train_size = int(len(data) * train_ratio)
    
    train = data[:train_size]
    test = data[train_size:]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_scaled = scaler_X.fit_transform(train)
    test_scaled = scaler_X.transform(test)
    
    scaler_y.fit(train[:, target_idx].reshape(-1, 1))
    
    X_train_full, y_train_full = create_sequences(train_scaled, lookback)
    X_test_full, y_test_full = create_sequences(test_scaled, lookback)
    
    y_train = y_train_full[:, target_idx].reshape(-1, 1)
    y_test = y_test_full[:, target_idx].reshape(-1, 1)
    
    return X_train_full, y_train, X_test_full, y_test, scaler_X, scaler_y
