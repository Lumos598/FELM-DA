from typing import List
import torch


class Attack:
    """
    Base class for all attack methods.
    """
    def attack(self, params_list: List[torch.Tensor]):
        raise NotImplementedError("Attack methods should implement this method!")
