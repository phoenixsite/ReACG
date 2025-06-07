import os
from typing import Tuple, Union
from pathlib import Path

import torch
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from robustbench import load_cifar10, load_model
from robustbench.data import (CustomImageFolder, get_preprocessing,
                              load_cifar100, get_timm_model_preprocessing)
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel


def load_imagenet(
    n_examples: int, data_dir: str, transforms_test
) -> Tuple[torch.Tensor]:
    """Load ImageNet data.

    Parameters
    ----------
    n_examples : int
        number of images to load.
    data_dir : str
        path to the imagenet.
    transforms_test : torchvision.transforms
        preprocessing for the imagenet pictures.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tensor of imagenet images and the ground truth labels.
    """
    assert n_examples <= 5000
    dataset = CustomImageFolder(
        f"{data_dir}/val",
        transform=transforms_test,
    )

    x_test = list()
    y_test = list()

    for index in range(n_examples):
        x, y, _ = dataset.__getitem__(index)
        x_test.append(x.unsqueeze(0))
        y_test.append(y)

    x_test = torch.vstack(x_test)
    y_test = torch.tensor(y_test)

    return x_test, y_test


def load_model_and_dataset(
    model_name: str,
    dataset: str,
    n_examples: int,
    threat_model: str = "Linf",
    relative_root: str = "../",
) -> Tuple:
    if dataset not in {"cifar10", "cifar100", "imagenet"}:
        raise NotImplementedError(f"Dataloader for {dataset} is not implemented.")

    model = load_model(
        model_name,
        model_dir=os.path.join(relative_root, "models"),
        dataset=dataset,
        threat_model=threat_model,
    )

    data_dir = os.path.join(relative_root, "data")

    if dataset == "cifar10":
        x_test, y_test = load_cifar10(
            n_examples=n_examples, data_dir=data_dir
        )
    elif dataset == "cifar100":
        x_test, y_test = load_cifar100(
            n_examples=n_examples, data_dir=data_dir
        )
    elif dataset == "imagenet":

        # prepr = get_preprocessing(
        #     BenchmarkDataset(dataset), ThreatModel(threat_model), model_name, None
        # )

        data_config = resolve_model_data_config(model, use_test_size=True)
        prepr = create_transform(**data_config, is_training=False)

        x_test, y_test = load_imagenet(
            n_examples=n_examples,
            data_dir=os.path.join(data_dir, "imagenet"),
            transforms_test=prepr,
        )
    return model, x_test, y_test
