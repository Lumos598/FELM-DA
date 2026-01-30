from typing import List
import torch
from par.par import PAR


class DFLTrust(PAR):
    """
    FLTrust is a robust aggregation method that:
    1. Uses cosine similarity to measure trust between client updates
    2. Normalizes client gradients to prevent magnitude-based attacks
    3. Applies trust scores to weight different client contributions
    distance-based
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)
        self.lr = args.get("meta_lr", 1e-3)

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
        # Step 1: Compute cosine similarity between server's and clients' gradients
        cos_sim = []
        for i in range(len(params_list)):
            cos_sim.append(torch.dot(grad_list[i].flatten(), grad.flatten()) / (torch.norm(grad_list[i]) * torch.norm(grad)))
        
        # Step 2: Calculate trust scores
        ts = []
        for i in range(len(params_list)):
            if cos_sim[i] > 0:
                ts.append(cos_sim[i])
            else:
                ts.append(0)
            # Normalize client gradients to have the same magnitude as server's gradient
            grad_list[i] = (torch.norm(grad)/torch.norm(grad_list[i]))*grad_list[i]
        
        # Step 3: Aggregate gradients using trust scores as weights
        # Ensure sum of trust scores is not too small to prevent division by zero
        sum_ts = max(sum(ts),1e-6)
        grad_ = torch.zeros_like(grad)
        for i in range(len(params_list)):
            grad_ += (ts[i]/sum_ts) * grad_list[i]

        # Step 4: Update parameters using the weighted aggregate gradient
        grad = grad_
        params += self.lr*grad.view(-1)
        return params
