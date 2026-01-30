from typing import List
import torch
from par.par import PAR


class Average(PAR):
    """
    Average all received params.
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
        if len(params_list) == 0:
            return params
        params_list.append(params)
        all = torch.stack(params_list, dim=1)
        m = torch.mean(all, dim=1)
        return m
