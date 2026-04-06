"""
Strategy-aware Federated Client.
Extends the base TimeSeriesClient with support for:
  - SCAFFOLD control variate corrections during local training
  - FedDyn dynamic regularization term in local loss
  - FedNova local step tracking
  - Standard training for all other algorithms
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class FedClient(fl.client.NumPyClient):
    """
    A flexible federated client that adapts its local training loop
    based on the selected federated strategy.
    """

    def __init__(self, model, train_data, test_data=None, scaler=None,
                 epochs=2, batch_size=64, strategy_type="fedavg",
                 scaffold_lr_correction=0.01, feddyn_alpha=0.01):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.scaler = scaler
        self.epochs = epochs
        self.batch_size = batch_size
        self.strategy_type = strategy_type
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # SCAFFOLD state
        self.scaffold_lr_correction = scaffold_lr_correction
        self.local_control = None
        self.server_control = None

        # FedDyn state
        self.feddyn_alpha = feddyn_alpha
        self.prev_global_params = None

        # FedNova step counter
        self.local_steps = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        # Store global params for FedDyn and SCAFFOLD
        global_params = [torch.tensor(p.copy()) for p in parameters]

        loader = DataLoader(
            TensorDataset(*self.train_data),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.local_steps = 0

        for _ in range(self.epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = self.criterion(output, yb)

                # ── FedDyn: add dynamic regularization ──
                if self.strategy_type == "feddyn":
                    reg_loss = 0.0
                    for param, gp in zip(self.model.parameters(), global_params):
                        reg_loss += torch.sum((param - gp) ** 2)
                    loss = loss + (self.feddyn_alpha / 2.0) * reg_loss

                # ── SCAFFOLD: apply control variate correction ──
                if self.strategy_type == "scaffold" and self.server_control is not None:
                    loss.backward()
                    with torch.no_grad():
                        for i, param in enumerate(self.model.parameters()):
                            if self.local_control is not None and i < len(self.local_control):
                                correction = self.server_control[i] - self.local_control[i]
                                param.grad.add_(torch.tensor(correction, dtype=param.grad.dtype))
                    self.optimizer.step()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.local_steps += 1

        # Update SCAFFOLD local control variate
        if self.strategy_type == "scaffold":
            new_local_control = []
            new_params = self.get_parameters(config)
            for i in range(len(new_params)):
                delta = (parameters[i] - new_params[i]) / (self.local_steps * 0.001 + 1e-10)
                if self.server_control is not None and i < len(self.server_control):
                    new_c = delta - self.server_control[i]
                else:
                    new_c = delta
                new_local_control.append(new_c)
            self.local_control = new_local_control

        # Return metrics with local_steps for FedNova
        return self.get_parameters(config), len(loader.dataset), {"local_steps": self.local_steps}

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

        from src.experiments.metrics import evaluate_metrics
        metrics = evaluate_metrics(y_true_inv, preds_inv)

        # Cast numpy float to native float for flwr RPC compatibility
        metrics = {k: float(v) for k, v in metrics.items()}

        return float(loss), len(xb), metrics
