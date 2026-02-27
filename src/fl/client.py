import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TimeSeriesClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data=None, scaler=None, epochs=2, batch_size=64):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.scaler = scaler
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        loader = DataLoader(
            TensorDataset(*self.train_data),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        for _ in range(self.epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = self.criterion(output, yb)
                loss.backward()
                self.optimizer.step()
                
        return self.get_parameters(config), len(loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        if self.test_data is None or self.scaler is None:
            return float('inf'), 0, {}
            
        xb, yb = self.test_data
        with torch.no_grad():
            preds = self.model(xb)
            loss = self.criterion(preds, yb)
            
        preds_inv = self.scaler.inverse_transform(preds.numpy())
        y_true_inv = self.scaler.inverse_transform(yb.numpy())
        
        import numpy as np
        from src.experiments.metrics import evaluate_metrics
        
        metrics = evaluate_metrics(y_true_inv, preds_inv)
        
        # Cast numpy float to native float for flwr RPC compatibility
        metrics = {k: float(v) for k, v in metrics.items()}
        
        return float(loss), len(xb), metrics
