__all__ = ['get_train_transforms', 'get_val_transforms', 'TRANSFORM_REGISTRY', 'load_transforms_from_yaml']
import yaml
import torchvision.transforms as transforms

from autoAugment.autoaugment import ImageNetPolicy, SVHNPolicy, CIFAR10Policy



def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    

TRANSFORM_REGISTRY = {
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "AutoAugment": transforms.AutoAugment,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
    "Resize": transforms.Resize,
    "CenterCrop": transforms.CenterCrop,
    # autoaugment
    "CIFAR10Policy": CIFAR10Policy,
    "SVHNPolicy": SVHNPolicy,
    "ImageNetPolicy": ImageNetPolicy,
}


def build_transforms(transform_list):
    ops = []
    for item in transform_list:
        name = item["name"]
        params = item.get("params", {})
        transform_cls = TRANSFORM_REGISTRY[name]
        ops.append(transform_cls(**params) if params else transform_cls())
    return transforms.Compose(ops)


def load_transforms_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    train_tf = build_transforms(cfg["transforms"]["train"])
    val_tf = build_transforms(cfg["transforms"]["val"])
    return train_tf, val_tf
