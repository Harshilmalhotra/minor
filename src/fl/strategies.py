"""
Custom Federated Learning Strategy implementations.
All strategies extend Flower's FedAvg and override aggregate_fit().

Algorithms implemented:
  - FedNova: Normalized averaging by local steps
  - SCAFFOLD: Server/client control variate correction
  - FedOpt: Server-side SGD with momentum
  - FedAdam: Server-side Adam optimizer
  - FedYogi: Server-side Yogi optimizer
  - FedAdagrad: Server-side Adagrad optimizer
  - FedDyn: Dynamic regularization tracking global-local drift
  - ClusteredFL: K-Means clustering of client weights then per-cluster FedAvg
"""

import flwr as fl
from flwr.common import Parameters, FitRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────
#  Utility: weighted average of ndarrays
# ─────────────────────────────────────────────
def _weighted_average(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """Compute weighted average of a list of (ndarrays, num_examples)."""
    total = sum(n for _, n in results)
    averaged = [
        np.sum([w[i] * n for w, n in results], axis=0) / total
        for i in range(len(results[0][0]))
    ]
    return averaged


def _extract_weights(results):
    """Extract weights and num_examples from fit results."""
    return [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
    ]


def _aggregate_metrics(results, aggregation_fn):
    """Helper to aggregate evaluation metrics."""
    metrics_aggregated = {}
    if aggregation_fn:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = aggregation_fn(fit_metrics)
    return metrics_aggregated


# ═════════════════════════════════════════════
#  1. FedNova
# ═════════════════════════════════════════════
class FedNova(fl.server.strategy.FedAvg):
    """
    Federated Normalized Averaging (FedNova).
    Normalizes each client's update by its number of local steps
    before aggregation, removing objective inconsistency.
    Reference: Wang et al., "Tackling the Objective Inconsistency Problem
    in Heterogeneous Federated Optimization", NeurIPS 2020.
    """

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)

        # Normalize each client's update by its local steps (tau_i)
        normalized = []
        tau_eff = 0.0
        for (client_proxy, fit_res), (weights, n_examples) in zip(results, weights_results):
            tau_i = fit_res.metrics.get("local_steps", 1)
            # Normalize: delta_i / tau_i, weight by n_examples
            normed_weights = [w / tau_i for w in weights]
            normalized.append((normed_weights, n_examples))
            tau_eff += n_examples * tau_i

        total_examples = sum(n for _, n in normalized)
        tau_eff /= total_examples  # effective tau

        # Weighted average of normalized updates, then scale by tau_eff
        avg_weights = _weighted_average(normalized)
        # Scale back so the magnitude is correct
        final_weights = [w * tau_eff for w in avg_weights]

        return ndarrays_to_parameters(final_weights), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)


# ═════════════════════════════════════════════
#  2. SCAFFOLD
# ═════════════════════════════════════════════
class SCAFFOLD(fl.server.strategy.FedAvg):
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
    Maintains server-side control variates to correct client drift.
    Reference: Karimireddy et al., ICML 2020.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_control = None  # Server control variate c

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)

        # Standard weighted average of model parameters
        aggregated_weights = _weighted_average(weights_results)

        # Update server control variate
        # c_new = c_old + (1/N) * sum(delta_c_i)
        # Here we approximate: use the average parameter shift as control correction
        if self.server_control is None:
            self.server_control = [np.zeros_like(w) for w in aggregated_weights]

        N = len(results)
        for i in range(len(self.server_control)):
            # Compute mean delta across clients relative to current aggregated
            deltas = np.mean(
                [weights_results[j][0][i] - aggregated_weights[i] for j in range(N)],
                axis=0
            )
            self.server_control[i] += deltas / N

        # Apply control variate correction to aggregated weights
        corrected = [
            aggregated_weights[i] - 0.1 * self.server_control[i]
            for i in range(len(aggregated_weights))
        ]

        return ndarrays_to_parameters(corrected), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)


# ═════════════════════════════════════════════
#  3. FedOpt (Server-side SGD with momentum)
# ═════════════════════════════════════════════
class FedOpt(fl.server.strategy.FedAvg):
    """
    FedOpt: Adaptive Federated Optimization with server-side SGD + momentum.
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
    """

    def __init__(self, *args, server_lr=1.0, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_lr = server_lr
        self.momentum = momentum
        self.velocity = None
        self.prev_weights = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)
        aggregated = _weighted_average(weights_results)

        if self.prev_weights is None:
            self.prev_weights = aggregated
            self.velocity = [np.zeros_like(w) for w in aggregated]
            return ndarrays_to_parameters(aggregated), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)

        # Pseudo-gradient: delta = aggregated - prev
        pseudo_grad = [aggregated[i] - self.prev_weights[i] for i in range(len(aggregated))]

        # SGD with momentum
        for i in range(len(self.velocity)):
            self.velocity[i] = self.momentum * self.velocity[i] + pseudo_grad[i]

        new_weights = [
            self.prev_weights[i] + self.server_lr * self.velocity[i]
            for i in range(len(aggregated))
        ]

        self.prev_weights = new_weights
        return ndarrays_to_parameters(new_weights), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)


# ═════════════════════════════════════════════
#  4. FedAdam
# ═════════════════════════════════════════════
class FedAdam(fl.server.strategy.FedAvg):
    """
    FedAdam: Server-side Adam optimizer on aggregated pseudo-gradients.
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
    """

    def __init__(self, *args, server_lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0
        self.prev_weights = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)
        aggregated = _weighted_average(weights_results)

        if self.prev_weights is None:
            self.prev_weights = aggregated
            self.m = [np.zeros_like(w) for w in aggregated]
            self.v = [np.zeros_like(w) for w in aggregated]
            return ndarrays_to_parameters(aggregated), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)

        self.t += 1
        pseudo_grad = [aggregated[i] - self.prev_weights[i] for i in range(len(aggregated))]

        new_weights = []
        for i in range(len(aggregated)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * pseudo_grad[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (pseudo_grad[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            new_weights.append(
                self.prev_weights[i] + self.server_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )

        self.prev_weights = new_weights
        return ndarrays_to_parameters(new_weights), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)


# ═════════════════════════════════════════════
#  5. FedYogi
# ═════════════════════════════════════════════
class FedYogi(fl.server.strategy.FedAvg):
    """
    FedYogi: Server-side Yogi optimizer — controls the effective learning rate
    adaptively by using sign-based second moment updates.
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
    """

    def __init__(self, *args, server_lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.prev_weights = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)
        aggregated = _weighted_average(weights_results)

        if self.prev_weights is None:
            self.prev_weights = aggregated
            self.m = [np.zeros_like(w) for w in aggregated]
            self.v = [np.ones_like(w) * self.epsilon for w in aggregated]
            return ndarrays_to_parameters(aggregated), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)

        self.t += 1
        pseudo_grad = [aggregated[i] - self.prev_weights[i] for i in range(len(aggregated))]

        new_weights = []
        for i in range(len(aggregated)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * pseudo_grad[i]
            # Yogi update: v_t = v_{t-1} - (1-beta2) * g^2 * sign(v_{t-1} - g^2)
            grad_sq = pseudo_grad[i] ** 2
            self.v[i] = self.v[i] - (1 - self.beta2) * grad_sq * np.sign(self.v[i] - grad_sq)
            self.v[i] = np.maximum(self.v[i], self.epsilon)  # Ensure positivity

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            new_weights.append(
                self.prev_weights[i] + self.server_lr * m_hat / (np.sqrt(self.v[i]) + self.epsilon)
            )

        self.prev_weights = new_weights
        return ndarrays_to_parameters(new_weights), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)


# ═════════════════════════════════════════════
#  6. FedAdagrad
# ═════════════════════════════════════════════
class FedAdagrad(fl.server.strategy.FedAvg):
    """
    FedAdagrad: Server-side Adagrad optimizer on aggregated pseudo-gradients.
    Accumulates squared gradients for per-parameter adaptive learning rates.
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
    """

    def __init__(self, *args, server_lr=0.1, epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_lr = server_lr
        self.epsilon = epsilon
        self.accumulated_sq = None
        self.prev_weights = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)
        aggregated = _weighted_average(weights_results)

        if self.prev_weights is None:
            self.prev_weights = aggregated
            self.accumulated_sq = [np.zeros_like(w) for w in aggregated]
            return ndarrays_to_parameters(aggregated), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)

        pseudo_grad = [aggregated[i] - self.prev_weights[i] for i in range(len(aggregated))]

        new_weights = []
        for i in range(len(aggregated)):
            self.accumulated_sq[i] += pseudo_grad[i] ** 2
            new_weights.append(
                self.prev_weights[i] + self.server_lr * pseudo_grad[i] / (np.sqrt(self.accumulated_sq[i]) + self.epsilon)
            )

        self.prev_weights = new_weights
        return ndarrays_to_parameters(new_weights), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)


# ═════════════════════════════════════════════
#  7. FedDyn
# ═════════════════════════════════════════════
class FedDyn(fl.server.strategy.FedAvg):
    """
    FedDyn: Federated Learning with Dynamic Regularization.
    Maintains a server-side state h that accumulates parameter drift.
    Reference: Acar et al., "Federated Learning Based on Dynamic Regularization", ICLR 2021.
    """

    def __init__(self, *args, alpha=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.h = None  # Server-side gradient tracking state
        self.prev_global = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)
        aggregated = _weighted_average(weights_results)

        N = len(results)

        if self.h is None:
            self.h = [np.zeros_like(w) for w in aggregated]
            self.prev_global = [np.copy(w) for w in aggregated]
            return ndarrays_to_parameters(aggregated), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)

        # Update h:  h = h - alpha * (aggregated - prev_global)
        for i in range(len(self.h)):
            self.h[i] -= self.alpha * (aggregated[i] - self.prev_global[i])

        # Corrected global model: w_global = aggregated - (1/alpha) * h
        corrected = [
            aggregated[i] - (1.0 / (self.alpha * N + 1e-10)) * self.h[i]
            for i in range(len(aggregated))
        ]

        self.prev_global = [np.copy(w) for w in corrected]
        return ndarrays_to_parameters(corrected), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)


# ═════════════════════════════════════════════
#  8. Clustered Federated Learning
# ═════════════════════════════════════════════
class ClusteredFL(fl.server.strategy.FedAvg):
    """
    Clustered Federated Learning.
    Groups clients by model weight similarity (K-Means), performs
    per-cluster FedAvg, then averages cluster centroids.
    Reference: Sattler et al., "Clustered Federated Learning", IEEE TNNLS, 2020.
    """

    def __init__(self, *args, n_clusters=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results = _extract_weights(results)

        # Flatten each client's weights into a single vector for clustering
        flat_vectors = []
        for weights, _ in weights_results:
            flat = np.concatenate([w.flatten() for w in weights])
            flat_vectors.append(flat)
        flat_vectors = np.array(flat_vectors)

        # K-Means clustering
        n_clusters = min(self.n_clusters, len(results))
        if n_clusters < 2:
            # Fallback to standard FedAvg
            avg = _weighted_average(weights_results)
            return ndarrays_to_parameters(avg), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(flat_vectors)

        # Per-cluster weighted average
        cluster_weights = []
        cluster_sizes = []
        for c_id in range(n_clusters):
            members = [i for i, l in enumerate(labels) if l == c_id]
            if not members:
                continue
            cluster_data = [(weights_results[i][0], weights_results[i][1]) for i in members]
            cluster_avg = _weighted_average(cluster_data)
            cluster_total = sum(weights_results[i][1] for i in members)
            cluster_weights.append(cluster_avg)
            cluster_sizes.append(cluster_total)

        # Final: weighted average across clusters
        total = sum(cluster_sizes)
        final_weights = [
            np.sum([cw[i] * cs for cw, cs in zip(cluster_weights, cluster_sizes)], axis=0) / total
            for i in range(len(cluster_weights[0]))
        ]

        return ndarrays_to_parameters(final_weights), _aggregate_metrics(results, self.fit_metrics_aggregation_fn)
