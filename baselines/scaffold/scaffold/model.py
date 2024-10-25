"""scaffold: A Flower Baseline."""

from collections import OrderedDict

import torch
import numpy as np


def train(net, trainloader, epochs, device, global_control, local_control, optimizer_class=torch.optim.SGD , optimizer_kwargs = {"lr": 0.1, "momentum": 0.9}):
    """Train the model on the training set.

    Args:
        net (torch.nn.Module): The neural network model.
        trainloader (torch.utils.data.DataLoader): The data loader for the training set.
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device to use for training (e.g. "cuda" for GPU or "cpu" for CPU).
        control_global (list): The global control parameters.
        control_local (list): The local control parameters.
        optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use for training. Defaults to torch.optim.SGD.
        optimizer_kwargs (dict, optional): Additional keyword arguments to pass to the optimizer class. Defaults to {"lr": 0.1, "momentum": 0.9}.

    Returns:
        float: The average training loss.
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            if images.shape[0] == 1:
                # Skip batches with a single image
                continue
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            for param, c, c_i in zip(
                net.parameters(), global_control, local_control
            ):
                if param.requires_grad:
                    param.grad.data += torch.tensor(c - c_i).to(device)
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
