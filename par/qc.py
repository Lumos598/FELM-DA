import math
from typing import List
import torch
from par.par import PAR


class QC(PAR):
    """
    Q-Consensus.
    distance-based
    Requirements: n >= b + 1
    """

    def __init__(self, rank, neighbors, init_value=1.0, **args) -> None:
        super().__init__(rank, neighbors)
        self.q = {}
        self.weights = {}
        self.rewards = {}
        self.src = neighbors
        self.neighbors_n = 1 + len(neighbors)
        for s in neighbors:
            self.q[s] = init_value
            self.weights[s] = 1 / self.neighbors_n

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
        assert n >= b + 1, "The number of params should >= b + 1."
        params_n = params.shape[0]
        batch_size = min(params_n, 5120)
        epochs_n = math.ceil(params_n / batch_size)
        step_size = 0.1

        for s in self.src:
            self.rewards[s] = 0

        for i in range(epochs_n):
            step_size -= 0.001
            start, end = batch_size * i, min(params_n, batch_size * (i + 1))
            self_p = params[start:end]
            # update r
            for j, s in enumerate(self.src):
                r_ij = math.exp(-1 * 1000 * abs((params_list[j][start:end] - self_p).mean()))
                self.rewards[s] += r_ij
        sum_r = sum(self.rewards.values())
        # update q
        for j, s in enumerate(self.src):
            r_ij = self.rewards[s] / max(1e-6, sum_r)
            self.q[s] += max(step_size, 0.0001) * (r_ij - self.q[s])
        sum_q = sum(self.q.values())
        for j, s in enumerate(self.src):
            self.weights[s] = (1 - 1 / self.neighbors_n) * (self.q[s] / sum_q)
            if(self.weights[s] < 0.1):
                self.weights[s] = 0
        w_sum = sum(self.weights.values()) + 1 / self.neighbors_n
        for i,s in enumerate(self.src):
            self.weights[s] = self.weights[s] / w_sum
        res = ((1 / self.neighbors_n) / w_sum) * params
        for i,s in enumerate(self.src):
            res += self.weights[s] / w_sum * params_list[i]
        return res
