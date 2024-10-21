"""Flower Server."""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import numpy as np

from fedprox.models import test


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-10 test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        # We could compile the model here but we are not going to do it because
        # running test() is so lightweight that the overhead of compiling the model
        # negate any potential speedup. Please note this is specific to the model and
        # dataset used in this baseline. In general, compiling the model is worth it

        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
