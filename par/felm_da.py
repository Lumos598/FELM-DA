import math
from typing import List
import copy
import torch
from par.par import PAR


class FELM_DA(PAR):
    def __init__(self, rank, neighbors, init_value=1.0, **args) -> None:
        """
        Initialize the FELM-DA aggregator.
        
        Args:
            rank: The rank/ID of current node
            neighbors: List of neighbor nodes
            init_value: Initial Q-value for each neighbor (default: 1.0)
        """
        super().__init__(rank, neighbors)
        self.q = {}
        self.weights = {}
        self.rewards = {}
        self.src = neighbors
        # Total number of nodes including self
        self.neighbors_n = 1 + len(neighbors)
        self.his_self_p = None
        self.self_p_0 = None
        # Initialize Q-values and weights for each neighbor
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
        params_n = params.shape[0]
        step_size = 0.1
        
        start, end = 0, 0
        for l_idx in range(layers_n):
            weights_ = [params_dict[f'weight_{l_idx}']]
            bias_ = [params_dict[f'bias_{l_idx}']]
            length = (weights_[0].data.shape[0]+bias_[0].data.shape[0])
            end = start + length
            start = end
        batch_size = min(end, 512)
        steps_n = math.ceil(end / batch_size)
        
        # Initialize rewards for each neighbor
        for s in self.src:
            self.rewards[s] = 0
            
        # Update Q-values based on parameter similarity
        for step in range(steps_n):
            step_size -= 0.001
            s_, e_ = batch_size * step, min(end, batch_size * (step + 1))
            params_c_s = params[s_:e_]
            
            # Get neighbor parameters for current batch
            params_c_n = [[0 for _ in range(end)] for _ in range(len(self.src))]
            for i in range(len(self.src)):
                params_c_n[i] = params_list[i][s_:e_]
            # Calculate rewards based on parameter similarity
            for i, s in enumerate(self.src):
                r_ij = math.exp(-1000 * abs((params_c_n[i] - params_c_s).mean()))
                self.rewards[s] = r_ij
            sum_r = sum(self.rewards.values())
            for i, s in enumerate(self.src):
                r_ij = self.rewards[s] / sum_r
                self.q[s] += max(step_size, 0.01) * (r_ij - self.q[s])
        sum_q = sum(self.q.values())
        for i,s in enumerate(self.src):
            self.weights[s] = (1 - 1 / self.neighbors_n) * (self.q[s]/sum_q)
        w_ij = copy.copy(self.weights)
        for i,s in enumerate(self.src):
            if w_ij[s] < (sum(self.weights.values())/(2*len(self.src))):
                w_ij[s] = 0
        sum_w = sum(w_ij.values()) + 1 / self.neighbors_n
        for i,s in enumerate(self.src):
            w_ij[s] = w_ij[s] / sum_w
        
        # delayed aggregation
        if epoch!=0 and epoch % 15 == 0:
            res = ((1 / self.neighbors_n)/sum_w) * params
            for i,s in enumerate(self.src):
                res += w_ij[s] * params_list[i]
            return res
        
        res_s = params[0:end]
        if self.self_p_0 is None:
            self.self_p_0 = res_s
        res_e = ((1 / self.neighbors_n)/sum_w) * params[end:params_n]
        for i,s in enumerate(self.src):
            res_e += w_ij[s] * params_list[i][end:params_n]
        return torch.cat((res_s, res_e), dim=0)
