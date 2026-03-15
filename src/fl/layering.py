import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class LayeredFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, layer_bias: Dict[int, float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_bias = layer_bias or {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results:
            return None, {}
        
        # Simulating Layering: Group clients into Layers (e.g., by CID or index)
        # Mathematically: W_global = sum(beta_l * Agg(Layer_l))
        
        # For this prototype, we'll split clients into 2 layers
        layer_groups = {0: [], 1: []}
        for i, (client, fit_res) in enumerate(results):
            layer_id = i % 2 
            layer_groups[layer_id].append((client, fit_res))
            
        # Aggregate each layer
        layer_weights = []
        layer_examples = []
        
        for layer_id, group in layer_groups.items():
            if not group: continue
            
            # Simple average within layer
            weights_results = [
                (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in group
            ]
            
            # Weighted average for the layer
            total_examples = sum([num_examples for _, num_examples in weights_results])
            aggregated_weights = [
                np.sum([w * num_examples for w, num_examples in weights_results], axis=0) / total_examples
                for w in zip(*[w for w, _ in weights_results])
            ]
            
            # Apply layer bias (beta_l)
            bias = self.layer_bias.get(layer_id, 1.0)
            biased_weights = [w * bias for w in aggregated_weights]
            
            layer_weights.append(biased_weights)
            layer_examples.append(total_examples)
            
        # Final aggregation across layers
        total_layers_examples = sum(layer_examples)
        final_weights = [
            np.sum([w * n for w, n in zip(layer_weights, layer_examples)], axis=0) / total_layers_examples
            for w in zip(*layer_weights)
        ]
        
        parameters_aggregated = fl.common.ndarrays_to_parameters(final_weights)
        
        # Metrics aggregation
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
        return parameters_aggregated, metrics_aggregated
