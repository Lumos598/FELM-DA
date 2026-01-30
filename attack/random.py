from attack.attack import Attack
import numpy as np
import torch


class RandomAttack(Attack):
    """
    Random Attack.
    noise: A random value that satisfies the (0, 1) uniform distribution
    """

    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list):
        params_n = params_list[0].shape[0]
        random_vector = torch.rand(params_n)
        all = torch.stack(params_list, axis=1)
        m = torch.mean(all, 1)
        res = m + random_vector
        return res
        
    def name(self):
        name = "random"
        return name