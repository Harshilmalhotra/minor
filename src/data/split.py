import numpy as np
import torch

def create_iid_splits(X_train, y_train, num_clients):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    
    client_data = []
    for idx in split_indices:
        client_data.append((
            torch.tensor(X_train[idx], dtype=torch.float32),
            torch.tensor(y_train[idx], dtype=torch.float32)
        ))
    return client_data

def create_non_iid_splits(X_train, y_train, num_clients):
    """
    Chronological partitioning to simulate Non-IID data.
    Each client gets a contiguous chunk of time, representing 
    different seasonal/temporal behaviors.
    """
    split_size = len(X_train) // num_clients
    
    client_data = []
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_clients - 1 else len(X_train)
        
        client_data.append((
            torch.tensor(X_train[start_idx:end_idx], dtype=torch.float32),
            torch.tensor(y_train[start_idx:end_idx], dtype=torch.float32)
        ))
    return client_data
