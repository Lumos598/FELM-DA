from par.par import PAR
import torch
from typing import List


class DMedian(PAR):
    """
    Coordinate-wise median of n received params.
    Requirement: n >= 2b + 1
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
        """
        Coordinate-wise median of n received params.
        """
        n = len(params_list)
        # assert n >= 2 * b + 1, "The number of neighbors should >= 2b + 1."
        all = torch.stack(params_list, 1)
        m = torch.median(all, 1).values
        res = torch.stack([m, params], 1)
        return torch.mean(res, 1)
