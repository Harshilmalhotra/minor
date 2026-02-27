import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred) * 100  # Converted to percentage
    
    # Calculate Normalized RMSE (NRMSE) Tracking Accuracy
    # Highly robust for load forecasting with large variances
    y_max, y_min = np.max(y_true), np.min(y_true)
    nrmse = rmse / (y_max - y_min) if y_max > y_min else 0
    tracking_accuracy = (1 - nrmse) * 100
    
    return {
        "Accuracy (%)": tracking_accuracy,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape(y_true, y_pred)
    }
