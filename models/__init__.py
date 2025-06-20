import torch

from models.Resnet18 import Resnet18Model
from models.Resnet50 import Resnet50Model
from models.Resnet50HeadArc import Resnet50HeadArc

__all__ = ['Resnet18Model', 'Resnet50HeadArc']

MODEL_REGISTRY: 'dict[str, torch.nn.Module]' = {
    "Resnet18Model": Resnet18Model,
    "Resnet50Model": Resnet50Model,
    "Resnet50HeadArc": Resnet50HeadArc,
}

def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]