from typing import List
import torch
from par.par import PAR

class SCClip(PAR):
    """
    Clipped Gossip algorithm for federated learning.
    This algorithm averages the received parameters and clips them to avoid large updates.
    distance-based
    """

    def __init__(self, rank, neighbors, clip_threshold=1.0, **args) -> None:
        """
        Initializes the Clipped Gossip algorithm.
        
        Args:
            clip_threshold (float): The threshold for gradient clipping (L2 norm).
        """
        super().__init__(rank, neighbors, **args)
        self.clip_threshold = clip_threshold

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
        
        distances = [(params - params_list[i]).norm() for i in range(len(params_list))]
        if len(distances) >= 2:
            tau = sorted(distances)[-2]
        else:
            tau = distances[-1]
        diff_params = []
        for i in range(len(params_list)):
            diff_params.append(params_list[i] - params)
        clipped_params_list = []
        for diff in diff_params:
            # Compute L2 norm
            diff_norm = torch.norm(diff)
            # clip
            scale = min(1, tau/diff_norm)
            if torch.isnan(diff_norm):
                break
            clipped_params_list.append(params + scale * diff)
        # Average all clipped parameters
        res = 1/(len(clipped_params_list)+1) * params
        for p in clipped_params_list:
            res += 1/(len(clipped_params_list)+1) * p
        return res
