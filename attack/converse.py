from typing import List
import torch
from attack.attack import Attack


class ConverseAttack(Attack):
    """
    Return (-10) * params.
    """

    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        all = torch.stack(params_list, axis=1)
        m = torch.mean(all, 1)
        return -10 * m
    
    def name(self):
        name = "converse"
        return name
