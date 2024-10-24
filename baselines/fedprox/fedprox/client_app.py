"""fedprox: A Flower Baseline."""

import json
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedprox.dataset import load_data
from fedprox.model import get_weights, set_weights, test, train
from fedprox.utils import instantiate_model_from_string


class FlowerClient(NumPyClient):
    """A class defining the client."""

    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Train model using this client's data."""
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            proximal_mu=config["proximal_mu"],
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Evaluate model using this client's data."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    net = instantiate_model_from_string(context.run_config["model.class"], num_classes=context.run_config["dataset.num_classes"])
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader = load_data(
        partition_id,
        num_partitions,
        partitioner_class=context.run_config["dataset.partitioner"],
        partitioner_kwargs=json.loads(context.run_config["dataset.partitioner_kwargs"]),
        dataset=context.run_config["dataset.name"],
    )
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
