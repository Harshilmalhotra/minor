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

def create_algorithmic_splits(X_train, y_train, num_clients):
    """
    Simulates nodes generating data via distinct algorithms.
    Each client modifies the base data using a distinct mathematical function.
    """
    client_data = []
    
    for i in range(num_clients):
        X_client = X_train.copy()
        y_client = y_train.copy()
        
        # Apply deterministic but unique algorithms per node
        if i % 4 == 0:
            # Algorithm 1: Add sinusoidal noise (simulating periodic fluctuations)
            time = np.arange(len(X_client))
            noise = 0.5 * np.sin(2 * np.pi * time / 24) # Daily cycle noise
            X_client += noise[:, np.newaxis, np.newaxis]
        elif i % 4 == 1:
            # Algorithm 2: Random Walk Drift
            drift = np.cumsum(np.random.normal(0, 0.05, size=X_client.shape), axis=0)
            X_client += drift
        elif i % 4 == 2:
            # Algorithm 3: High frequency Gaussian noise
            noise = np.random.normal(0, 0.2, size=X_client.shape)
            X_client += noise
        else:
            # Algorithm 4: Scaling factor (e.g., larger household generating more usage)
            scale = 1.0 + (i * 0.1)
            X_client *= scale
            y_client *= scale
            
        client_data.append((
            torch.tensor(X_client, dtype=torch.float32),
            torch.tensor(y_client, dtype=torch.float32)
        ))
    return client_data
