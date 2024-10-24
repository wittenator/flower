"""fedprox: A Flower Baseline."""

import importlib

from flwr.common.context import Context

def class_from_string(class_string: str) -> type:
    module = importlib.import_module('.'.join(class_string.split('.')[:-1]))
    class_ = getattr(module, class_string.split('.')[-1])
    return class_

def instantiate_model_from_string(model_string: str, num_classes) -> object:
    model_class = class_from_string(model_string)
    net = model_class(num_classes=num_classes)
    return net

def seed_all(seed: int) -> None:
    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
