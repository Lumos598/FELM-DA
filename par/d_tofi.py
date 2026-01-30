from typing import List
import torch
from par.par import PAR
from utils import set_meta_model_flat_params
import torch.nn.functional as F
import math
from models import CIFAR10, MNIST, CIFAR100, FashionMNIST, FEMNIST


class DToFi(PAR):
    """
    Decentralized ToFi algorithm for Byzantine-robust federated learning.
    Each node evaluates neighbors' models using a local reference dataset and applies a two-filter process.
    distance+performance based
    """

    meta_models = {"MNIST": MNIST, "CIFAR10": CIFAR10, "CIFAR100": CIFAR100, "FashionMNIST": FashionMNIST, "FEMNIST": FEMNIST}
    def __init__(self, rank, neighbors, **args):
        super().__init__(rank, neighbors, **args)
        self.tau = args.get("tau", 0.8)
        self.epsilon = args.get("epsilon", 2)
        self.lr = args.get("meta_lr", 1e-3)
        self.h_params = None

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
        Perform the two-filter process to aggregate neighbor parameters.
        Args:
            params: Local model parameters.
            params_list (List[torch.Tensor]): Received parameters from neighbors.
            params_dict, classes_params_dict, layers_n, model, test_loader, grad,
            grad_list, b, device_id, data, target, epoch: Other input arguments.
        
        Returns:
            torch.Tensor: Aggregated model parameters.
        """
        if len(params_list) == 0:
            return params
        
        # Evaluate neighbors' parameters using the local reference dataset
        if dataset == "FEMNIST":
            temp_model = self.meta_models[dataset]().get_model()
        else:
            temp_model = self.meta_models[dataset]()
        test_data, test_labels = next(iter(test_loader))
        test_data, test_labels = test_data.to(device_id), test_labels.to(device_id)
        criterion = F.cross_entropy
        losses = []
        for params_ in params_list:
            set_meta_model_flat_params(temp_model, params_)
            temp_model.to(device_id)
            temp_model.eval()
            outputs = temp_model(test_data)
            loss = criterion(outputs, test_labels)
            losses.append(loss)
        # compute standard deviation
        mu = sum(losses) / len(losses)
        variance = math.sqrt(sum((x-mu) ** 2 for x in losses)/len(losses))
        # compute normarlized losses
        losses = [(x-mu)/max(1e-6, variance) for x in losses]

        # filter the params with the normarlized losses
        filter1_params = []
        filter1_losses = []
        for i, l in enumerate(losses):
            if math.exp(-l) > self.tau:
                filter1_params.append(params_list[i])
                filter1_losses.append(losses[i])

        # filter the params with the similarity
        filter2_params = []
        filter2_losses = []
        if self.h_params is not None:
            for i, p in enumerate(filter1_params):
                cos_sim = torch.dot((params-p)/self.lr, (self.h_params-params)/self.lr) / (torch.norm((params-p)/self.lr) * torch.norm((self.h_params-params)/self.lr))
                arccos_sim = torch.acos(cos_sim)
                norm_diff = torch.norm((params-p)/self.lr-(self.h_params-params)/self.lr)
                similarity = arccos_sim + norm_diff
                if similarity < self.epsilon:
                    filter2_params.append(filter1_params[i])
                    filter2_losses.append(math.exp(-filter1_losses[i]))
        # compute weights
        model.to(device_id)
        model.eval()
        outputs = model(test_data)
        self_loss = math.exp(criterion(outputs, test_labels))
        weights = [l/(sum(filter2_losses)+self_loss) for l in filter2_losses]
        res = (self_loss/(sum(filter2_losses)+self_loss)) * params
        for i, p in enumerate(filter2_params):
            res += weights[i] * p
        return res
