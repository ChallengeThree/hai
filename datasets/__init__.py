from torch.utils.data import Dataset

from datasets.CustomDataset import CustomImageDataset
from datasets.FastImageDataset import FastImageDataset

__all__ = ["CustomImageDataset", "FastImageDataset"]

DATASET_REGISTRY: 'dict[str, Dataset]' = {
    "CustomImageDataset": CustomImageDataset,
    "FastImageDataset": FastImageDataset,
}

def get_dataset(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name]
