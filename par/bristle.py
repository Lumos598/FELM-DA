from typing import List, Dict
import torch
import torch.nn.functional as F
from par.par import PAR
import numpy as np
import math
from collections import Counter
import copy
from utils import set_meta_model_flat_params


label_len = {
    "MNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "FashionMNIST": 10,
    "FEMNIST": 62
}

class Bristle(PAR):
    """
    Bristle middleware for decentralized federated learning in Byzantine, non-i.i.d. environments.
    distance+performance based
    """    
    def __init__(self, rank, neighbors, **args) -> None:
        """
        Initialize the Bristle algorithm with the given parameters.
        """
        super().__init__(rank, neighbors, **args)
        self.src = neighbors
        self.alpha = args.get("alpha", 0.4)  # Exploration-exploitation ratio
        self.beta = args.get("beta", 30)    # Max. prioritized models
        self.kappa = args.get("kappa", 10) # Test samples per class
        self.phi = args.get("phi", 3) # familiar class selection size
    
    def distance_based_prioritizer(self, received_models: List[torch.Tensor], my_model: torch.Tensor) -> List[torch.Tensor]:
        """
        Prioritize models based on the Euclidean distance.
        """
        distances = [torch.dist(my_model, model, p=2) for model in received_models] # compute Euclidean distance
        sorted_indices = torch.argsort(torch.tensor(distances)) # sort
        sizes = [len(sorted_indices) // 3] * 2 + [len(sorted_indices) - 2 * (len(sorted_indices) // 3)]
        low, medium, high = torch.split(sorted_indices, sizes) #  split into three groups
        # Sampling based on alpha
        num_samples = min(len(received_models), self.beta)

        selected_indices = []
        if len(low) > 0:
            selected_indices.extend(low[:int((1 - self.alpha) ** 2 * num_samples)])
        if len(medium) > 0:
            selected_indices.extend(medium[:int((2 * self.alpha * (1 - self.alpha)) * num_samples)])
        if len(high) > 0:
            selected_indices.extend(high[:int((self.alpha ** 2) * num_samples)])
        return [received_models[idx] for idx in selected_indices]

    def performance_based_integrator(self, prioritized_params: List[torch.Tensor], 
                                    test_data: torch.Tensor, test_labels: torch.Tensor, 
                                    self_params: torch.Tensor, familiar_classes: List, device_id) -> torch.Tensor:
        """
        Integrate prioritized models based on performance evaluation for familiar and foreign classes.
        """
        # Step 1: Initialize variables
        F1 = {}  # Dictionary to store F1 scores for models and classes
        certainty = {}  # Certainty values for each model
        disc = {}  # Discrepancy values for each model and class
        faWg = {}  # Familiar class weights
        foWg = {}  # Foreign class weights
        # Evaluate F1 scores for current and prioritized models
        all_params = [self_params] + prioritized_params
        if len(prioritized_params) == 0:
            return self_params
        
        for i, params_ in enumerate(all_params):
            F1[i] = {}
            temp_model = copy.deepcopy(self.model)
            set_meta_model_flat_params(temp_model, params_)
            temp_model.to(device_id).eval()
            for c in familiar_classes:
                F1[i][c] = self.evaluate_f1(temp_model, c, test_data, test_labels).cpu()
        
        sorted_f1 = sorted(F1[0].values(), reverse=True)
        top_f1 = sorted_f1[:self.phi]
        if len(top_f1) != 0:
            certainty[0] = max(sum(top_f1) / len(top_f1) - torch.std(torch.tensor(top_f1)), 0)
        if torch.isnan(torch.tensor(certainty[0])) or certainty[0] == 0 or len(top_f1) == 0:
            certainty[0] = 1e-3
        self_weight = max(certainty[0], 1/len(prioritized_params))

        # Step 2: Compute certainty for prioritized models
        for i, _ in enumerate(prioritized_params):
            sorted_f1 = sorted(F1[i+1].values(), reverse=True)
            top_f1 = sorted_f1[:self.phi]  # Select top-k F1 scores
            certainty[i+1] = max(sum(top_f1) / len(top_f1) - torch.std(torch.tensor(top_f1)), 1/len(prioritized_params), 0)
            if torch.isnan(torch.tensor(certainty[i+1])) or certainty[i+1] == 0:
                certainty[i+1] = 1e-3
        # Step 3: Compute disc and weights for familiar and foreign classes
        # prioritized models include self model
        for i, _ in enumerate(prioritized_params):
            disc[i] = {}
            faWg[i] = {}
            for c in familiar_classes:
                if F1[i+1][c] >= F1[0][c]:
                    disc[i][c] = (abs(F1[i+1][c] - F1[0][c])*10) ** 3
                else:
                    disc[i][c] = float('-inf')  # Ignore models with worse performance
                faWg[i][c] = max(1e-3, (10/(1+math.exp(-disc[i][c]/100)))-4)*certainty[i+1]
            foWg[i] =  max(1e-3, (10/(1+math.exp(-sum(disc[i].values())/100)))-4)*certainty[i+1]

        # Step 4: Integrate familiar and foreign class contributions
        classes_params_list = []
        start, end = 0, 0
        for i, params_ in enumerate(prioritized_params):
            start, end = 0, 0
            temp_params = []
            for j in reversed(range(label_len[self.dataset])):
                end = start + self.classes_params_dict[j].shape[0]
                if start == 0:
                    temp_params.append(params_[-end:])
                else:
                    temp_params.append(params_[-end:-start])
                start = end
            classes_params_list.append(temp_params)
        sum_faWg = {}
        for _, inner_dict in faWg.items():
            for inner_key, value in inner_dict.items():
                if inner_key not in sum_faWg:
                    sum_faWg[inner_key] = 0
                sum_faWg[inner_key] += value
        sum_faWg = {key: value+self_weight for key, value in sum_faWg.items()}
        sum_foWg = sum(foWg.values()) + self_weight
        for label in range(label_len[self.dataset]):
            res_e = torch.zeros_like(self.classes_params_dict[label])
            for i, params_ in enumerate(classes_params_list):
                if label in familiar_classes:
                    res_e += (faWg[i][label] / sum_faWg[label]) * params_[label]
                else:
                    res_e += (foWg[i] / sum_foWg) * params_[label]
            if label in familiar_classes:
                res_e += (self_weight/sum_faWg[label]) * self.classes_params_dict[label]
            else:
                res_e += (self_weight/sum_foWg) * self.classes_params_dict[label]
            self.classes_params_dict[label] = res_e
        flattend_params = torch.cat(list(self.classes_params_dict.values()), dim=0)
        params_n = self_params.shape[0]
        return torch.cat((self_params[0: (params_n - end)], flattend_params), dim=0)
    
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
        Perform Bristle aggregation of models.
        """
        self.dataset = dataset
        self.classes_params_dict = classes_params_dict
        self.model = model
        # Distance-based prioritizer
        prioritized_models = self.distance_based_prioritizer(params_list, params)
        label_counts = target.bincount()
        label_num = (label_counts > 0).sum().item()
        familiar_classes = [label for label, count in enumerate(label_counts) if count>target.shape[0]/label_num]  # Define familiar classes based on data
        # Performance-based integrator
        test_data, test_labels = next(iter(test_loader))
        test_data, test_labels = test_data.to(device_id), test_labels.to(device_id)
        integrated_model = self.performance_based_integrator(prioritized_models, test_data, test_labels, params, familiar_classes, device_id)
        
        return integrated_model
    
    @staticmethod
    def compute_f1(predictions, labels):
        """
        Compute F1-score for model predictions.
        """
        # one-hot
        predictions = F.one_hot(predictions, num_classes=62).float()
        labels = F.one_hot(labels, num_classes=62).float()
        tp = torch.sum(labels * predictions, dim=0)
        fp = torch.sum(predictions * (1 - labels), dim=0)
        fn = torch.sum((1 - predictions) * labels, dim=0)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        return 2 * precision * recall / (precision + recall + 1e-10)
    
    def evaluate_f1(self, model, class_label, test_data, test_labels):
        """
        Evaluate F1 score for a specific model and class.
        """
        # Filter test data for the specific class
        class_data = test_data[test_labels == class_label]
        class_labels = test_labels[test_labels == class_label]
        # Get predictions
        predictions = model(class_data).argmax(dim=1)
        f1_scores = self.compute_f1(predictions, class_labels)
        return f1_scores[class_label]

    def integrate_familiar_classes(self, models, weights):
        """
        Integrate familiar class contributions.
        """
        integrated_model = torch.zeros_like(models[0])
        for model, weight in weights.items():
            integrated_model += weight * model
        return integrated_model

    def integrate_foreign_classes(self, models, weights):
        """
        Integrate foreign class contributions.
        """
        integrated_model = torch.zeros_like(models[0])
        for model, weight in weights.items():
            integrated_model += weight * model
        return integrated_model
    