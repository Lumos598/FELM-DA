from typing import List
import torch


class PAR:
    """
    Parameters Aggeregation Rule.

    All par need to implement this.
    """

    def __init__(self, rank, neighbors, **args) -> None:
        self.rank = rank
        self.neighbors = neighbors

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
        """Aggeregate params.

        Args:
            params: Local model parameters.
            params_list: List of received model parameters from neighbors.
            params_dict: Dictionary of parameters
            classes_params_dict: Class-specific parameters
            layers_n: Number of layers in the model
            model: The model being trained
            test_loader: DataLoader for testing
            grad: Local gradient
            grad_list: List of gradients from neighbors
            b: Byzantine tolerance parameter
            device_id: ID of the device being used
            dataset: Dataset being used
            data: Input data for training
            target: Target labels for training
            epoch: Current training epoch.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("PAR should implement this method!")
