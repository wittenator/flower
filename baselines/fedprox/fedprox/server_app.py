"""fedprox: A Flower Baseline."""

from collections import OrderedDict
import json
from typing import Callable, Dict, List, Optional, Tuple

from fedprox.dataset import load_data
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from flwr.common.typing import NDArrays, Scalar
from fedprox.utils import instantiate_model_from_string, seed_all
from fedprox.model import get_weights, test
from flwr_datasets import FederatedDataset
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import Compose, Normalize, ToTensor

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Do weighted average of accuracy metric."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def gen_evaluate_fn(
    dataset: str,
    device: torch.device,
    model: torch.nn.Module,
    context: Context,
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
        # Create global FDS
    FDS = FederatedDataset(
            dataset=dataset,
            partitioners={},
        )
    testdata = FDS.load_split("test")
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    testdata = testdata.with_transform(apply_transforms)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=128, shuffle=False)

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-10 test set for evaluation."""
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        loss, accuracy = test(model, testloader, device=device)
        print(f"Round {server_round} accuracy: {accuracy}")
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    seed_all(42)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    net = instantiate_model_from_string(context.run_config["model.class"], num_classes=context.run_config["dataset.num_classes"])
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedProx(
        fraction_fit=float(fraction_fit),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=gen_evaluate_fn(
            dataset=context.run_config["dataset.name"],
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            model=net,
            context=context,
        ),
        proximal_mu=context.run_config["fedprox.proximal_mu"],
    )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
