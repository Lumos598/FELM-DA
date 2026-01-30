from typing import List
import torch
import random
from attack.attack import Attack

random.seed(0)

class HiddenAttack(Attack):
    """
    Attack certain dimention.
    Paper: The Hidden Vulnerability of Distributed Learning in Byzantium
    """

    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        all_params = torch.stack(params_list, dim=1)
        params_n = params_list[0].shape[0]
        E = torch.tensor([1 if random.uniform(0, 1) < 0.5 else 0 for _ in range(params_n)])
        mean_params = all_params.mean(dim=1)
        # rand_dim = -1
        # mean_params[rand_dim] += 5
        res = mean_params + E * 5
        return res

