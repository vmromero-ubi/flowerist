from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from client import FlowerClient
from custom_strategies.fedcustom import FedCustom
from fl_utils import get_parameters, set_parameters
from models.net import Net
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
from trainer import test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

NUM_CLIENTS = 100

def load_datasets(num_clients:int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)   
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # split each training partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10 # corresponds to 10 percent of the training data
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))   
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32, shuffle=True))
    testloader = DataLoader(testset, batch_size=32, shuffle=False) # this corresponds to the full dataset assumed to be in the server
    return trainloaders, valloaders, testloader

def dirichlet_split(dataset, batch_size=32, dirichlet_alpha=0.5, num_workers=100) -> List[Subset]:
    """
    Splits a dataset into multiple subsets using the Dirichlet distribution.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        batch_size (int, optional): The desired batch size for each subset. Defaults to 32.
        dirichlet_alpha (float, optional): The concentration parameter for the Dirichlet distribution. Defaults to 0.5.
        num_workers (int, optional): The number of workers to split the dataset among. Defaults to 100.

    Returns:
        list of torch.utils.data.Subset: A list of subsets of the original dataset.

    Raises:
        AssertionError: If the number of instances for each label is not equal to the expected count.

    Note:
        This function uses the Dirichlet distribution to split the dataset into subsets. The Dirichlet distribution
        is a continuous probability distribution defined over the simplex, which makes it suitable for generating
        proportions for splitting the dataset among multiple workers.

        The function first calculates the count of instances for each label in the dataset. It then iteratively
        splits the dataset by randomly assigning instances to each worker based on the proportions generated from
        the Dirichlet distribution. The proportions are adjusted to ensure that each worker has a similar number
        of instances.

        Finally, the function adjusts the subsets to obey the desired batch size requirements by redistributing
        instances between adjacent subsets.

    """
    labels_tensors = list(dataset.targets)
    labels = [tensor.item() for tensor in labels_tensors] if isinstance(labels_tensors[0], torch.Tensor) else labels_tensors

    counts = {label: labels.count(label) for label in set(labels)}

    min_size=0
    idx_batch=[[] for _ in range(num_workers)]
    training_data_lengths = None
    while min_size<2:
        for key in counts.keys():
            key_indices=np.where(np.array(labels)==key)[0]
            assert len(key_indices)==counts[key], "Expected equal value"
            np.random.shuffle(key_indices)
            key_proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_workers))
            key_proportions = np.array(
                [
                    p*(len(idx_j)<len(labels)/num_workers) for p, idx_j in zip(key_proportions, idx_batch)
                ]
            )
            key_proportions=key_proportions/key_proportions.sum()
            key_proportions=(np.cumsum(key_proportions)*len(key_indices)).astype(int)[:-1]

            idx_batch=[
                idx_j+idx.tolist() for idx_j, idx in zip(idx_batch, np.split(key_indices, key_proportions))
            ]
            training_data_lengths = [len(idx_j) for idx_j in idx_batch]
            min_size = min(training_data_lengths)

    # adjust to obey batch_size requirements
    while any(len(idx_j)%batch_size==1 for idx_j in idx_batch):
        for i, idx_j in enumerate(idx_batch):
            if len(idx_j) % batch_size == 1:
                if i < len(idx_batch) - 1:
                    idx_batch[i+1].append(idx_j.pop())
                elif i > 0:
                    idx_batch[i-1].append(idx_j.pop())
    training_data_lengths = [len(idx_j) for idx_j in idx_batch]
    return [Subset(dataset, idx_j) for idx_j in idx_batch]

trainloader, valloader, testloader = load_datasets(NUM_CLIENTS)

def client_fn(cid) -> FlowerClient:
    print(f"Client {cid}: Loading data")
    net = Net().to(DEVICE)
    client_trainloader = trainloader[int(cid)]
    client_valloader = valloader[int(cid)]
    return FlowerClient(cid, net, client_trainloader, client_valloader, DEVICE)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net().to(DEVICE)
    server_testloader = testloader
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, server_testloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
    client_resources=client_resources,
)