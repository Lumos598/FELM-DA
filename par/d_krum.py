from typing import List
import torch
from par.par import PAR


class DKrum(PAR):
    """
    1. Finds the n-b-2 closest params for each neighbor
    2. Calc score of each neighbor
    3. Use the smallest score neighbor as final res
    distance-based
    Requirement: n >= 2b + 3
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)

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
        n = len(params_list)
        # assert n >= 2 * b + 3, "Krum requirement: n >= 2b + 3."
        num_selection = max(n - b - 2, 1)
        scores = []
        for i, p1 in enumerate(params_list):
            dists = []
            for j, p2 in enumerate(params_list):
                if i != j:
                    d = (p1 - p2).pow(2).sum().sqrt()
                    dists.append(d)
            sorted_index = torch.argsort(torch.tensor(dists), descending=False)
            dists = torch.tensor(dists)
            scores.append(dists[sorted_index[0 : num_selection]].sum())
        res = torch.argsort(torch.tensor(scores), descending=False)
        all = torch.hstack([params_list[res[0]].unsqueeze(1), params.unsqueeze(1)])
        return torch.mean(all, 1)
