from typing import List
import math
import torch
from par.par import PAR


class Balance(PAR):
    """
    BALANCE algorithm for Byzantine-robust decentralized federated learning.
    distance-based
    """

    def __init__(self, rank, neighbors, gamma=0.3, kappa=1.0, alpha=0.3, **args) -> None:
        super().__init__(rank, neighbors, **args)
        self.gamma = gamma  # Threshold multiplier
        self.kappa = kappa  # Exponential decay rate
        self.alpha = alpha
        self.src = neighbors

    def par(
        self,
        params,
        params_list: List[torch.Tensor],
        params_dict,
        classes_params_dict,
        layers_n,
        model,
        test_loader,
        grad,
        grad_list: List[torch.Tensor],
        b,
        device_id,
        dataset,
        target,
        epoch
    ):
        if len(params_list) == 0:
            return params

        # Define similarity threshold based on epoch
        threshold = self.gamma * math.exp(-self.kappa * (epoch / 150))

        # Filter models based on similarity
        filtered_params = []
        for i, received_params in enumerate(params_list):
            difference = torch.norm(params - received_params)
            if difference <= threshold * torch.norm(params):
                filtered_params.append(received_params)
                
        # Aggregate accepted models
        aggregated_params = self.alpha * params
        if len(filtered_params) > 0:
            aggregated_params += (1-self.alpha) * torch.mean(torch.stack(filtered_params, dim=0), dim=0)
        else:
            aggregated_params += (1-self.alpha) * params

        return aggregated_params
