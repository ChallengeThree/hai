from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch import optim


SCHEDULER_REGISTRY: dict[str, type] = {
    "step": StepLR,
    "multistep": MultiStepLR,
    "cosine": CosineAnnealingLR,
    "plateau": ReduceLROnPlateau,
    "cosine_warm_restart": CosineAnnealingWarmRestarts,
}

def get_scheduler(name: str, optimizer: optim.Optimizer, params: dict) -> optim.lr_scheduler._LRScheduler:
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {name}")
    return SCHEDULER_REGISTRY[name](optimizer, **params)