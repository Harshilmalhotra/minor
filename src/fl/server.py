import flwr as fl
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import load_data, prepare_data
from src.data.split import create_iid_splits, create_non_iid_splits, create_algorithmic_splits
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.models.cascade_lstm import CascadeLSTMModel
from src.models.bilstm import BiLSTMModel
from src.fl.client import TimeSeriesClient
from src.fl.layering import LayeredFedAvg

def main(model_type="lstm", distribution="iid", num_clients=5, num_rounds=20, selected_features=None, strategy_type="fedavg"):
    print(f"Starting FL Simulation for {model_type.upper()} ({distribution.upper()}) with {strategy_type.upper()}")
    df = load_data(selected_features=selected_features)
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(df)
    
    num_features = X_train.shape[2]
    
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    test_data = (X_test_torch, y_test_torch)
    
    if distribution == "iid":
        client_data = create_iid_splits(X_train, y_train, num_clients)
    elif distribution == "non-iid":
        client_data = create_non_iid_splits(X_train, y_train, num_clients)
    elif distribution == "algorithmic":
        client_data = create_algorithmic_splits(X_train, y_train, num_clients)
        
    def client_fn(cid: str) -> fl.client.Client:
        if model_type == "lstm":
            model = LSTMModel(input_size=num_features)
        elif model_type == "cascade":
            model = CascadeLSTMModel(input_size=num_features)
        elif model_type == "bilstm":
            model = BiLSTMModel(input_size=num_features)
        else:
            model = TCNModel(input_size=num_features, num_channels=[128, 128, 128], kernel_size=3)
            
        tc = TimeSeriesClient(
            model=model, 
            train_data=client_data[int(cid)],
            test_data=test_data,
            scaler=scaler_y
        )
        if hasattr(tc, 'to_client'):
            return tc.to_client()
        return tc
        
    def evaluate_metrics_aggregation_fn(metrics):
        if not metrics:
            return {}
        total_examples = sum([num_examples for num_examples, _ in metrics])
        aggregated = {}
        for key in metrics[0][1].keys():
            aggregated[key] = sum([num_examples * m[key] for num_examples, m in metrics]) / total_examples
        return aggregated

    if strategy_type == "fedprox":
        strategy = fl.server.strategy.FedProx(
            proximal_mu=0.1,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
    elif strategy_type == "layered":
        strategy = LayeredFedAvg(
            layer_bias={0: 1.0, 1: 0.95}, # Example bias for layering
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "tcn"], default="lstm")
    parser.add_argument("--dist", type=str, choices=["iid", "non-iid"], default="iid")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    
    main(model_type=args.model, distribution=args.dist, num_clients=args.clients, num_rounds=args.rounds)
