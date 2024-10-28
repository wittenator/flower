"""scaffold: A Flower Baseline."""

from collections import OrderedDict
import json
from logging import log
from flwr.common.parameter import parameters_to_ndarrays
import torch
import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from scaffold.dataset import load_data
from scaffold.model import get_weights, set_weights, test, train
from scaffold.utils import class_from_string, instantiate_model_from_string, marshal_numpy, unmarshal_numpy


class FlowerClient(NumPyClient):
    """A class defining the client."""

    def __init__(self, net, trainloader, valloader, local_epochs, optimizer_class, optimizer_kwargs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.local_control = [np.zeros_like(p.cpu().detach().numpy()) for k, p in net.state_dict().items()]

    def fit(self, parameters, config):
        """Train model using this client's data."""
        set_weights(self.net, parameters)
        global_control = unmarshal_numpy(config["global_control"])
        train_loss = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            global_control=global_control,
            local_control=self.local_control,
        )
        # update local control
        with torch.no_grad():
            y_delta = []
            c_plus = []
            c_delta = []

            for x, y_i in zip(parameters, self.net.state_dict().values()):
                y_delta.append((y_i.cpu() - x).numpy())

            coef = 1 / (self.local_epochs * self.optimizer_kwargs["lr"])
            for c, c_i, y_del in zip(global_control, self.local_control, y_delta):
                c_plus.append(c_i - c - coef * y_del)

            for c_p, c_l in zip(c_plus, self.local_control):
                c_delta.append(c_p - c_l)

            self.local_control = c_plus
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "c_delta": marshal_numpy(c_delta) , "y_delta": marshal_numpy(y_delta)},
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
    optimizer_class = class_from_string(context.run_config["optimizer.class"])
    optimizer_kwargs = {
        "lr": float(context.run_config["optimizer.lr"]),
        "weight_decay": float(context.run_config["optimizer.weight_decay"]),
        "momentum": float(context.run_config["optimizer.momentum"]),
    }
    trainloader, valloader = load_data(
        partition_id,
        num_partitions,
        partitioner_class=context.run_config["dataset.partitioner"],
        partitioner_kwargs=json.loads(context.run_config["dataset.partitioner_kwargs"]),
        dataset=context.run_config["dataset.name"],
    )
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
