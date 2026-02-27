import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import load_data, prepare_data
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.experiments.metrics import evaluate_metrics

def train_model(model, train_loader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
    return model

def evaluate(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
    preds_inv = scaler_y.inverse_transform(preds)
    y_true_inv = scaler_y.inverse_transform(y_test)
    
    return evaluate_metrics(y_true_inv, preds_inv)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    print("Loading data...")
    df = load_data()
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(df)
    
    num_features = X_train.shape[2]
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                      torch.tensor(y_train, dtype=torch.float32)), 
        batch_size=64, shuffle=False
    )
    
    print(f"\n--- Training Centralized LSTM (Input Features: {num_features}) ---")
    lstm = LSTMModel(input_size=num_features)
    lstm = train_model(lstm, train_loader, epochs=args.epochs)
    lstm_metrics = evaluate(lstm, X_test, y_test, scaler_y)
    print("LSTM Metrics:", lstm_metrics)
    
    print(f"\n--- Training Centralized TCN (Input Features: {num_features}) ---")
    tcn = TCNModel(input_size=num_features, num_channels=[128, 128, 128], kernel_size=3)
    tcn = train_model(tcn, train_loader, epochs=args.epochs)
    tcn_metrics = evaluate(tcn, X_test, y_test, scaler_y)
    print("TCN Metrics:", tcn_metrics)
