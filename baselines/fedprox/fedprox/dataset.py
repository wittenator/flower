"""fedprox: A Flower Baseline."""

from fedprox.utils import class_from_string
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

FDS = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, partitioner_class: str = "IidPartitioner", partitioner_kwargs: dict = {}, dataset: str = "uoft-cs/cifar10"):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        partitioner_class = class_from_string(partitioner_class)
        partitioner = partitioner_class(num_partitions=num_partitions, **partitioner_kwargs)
        FDS = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner},
        )
    partition = FDS.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=64)
    return trainloader, testloader
