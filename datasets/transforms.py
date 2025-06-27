__all__ = ['get_train_transforms', 'get_val_transforms', 'TRANSFORM_REGISTRY', 'load_transforms_from_yaml']
import yaml
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

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
    
class CustomAutoAugment:
    def __init__(self, policy):
        # str이면 enum 또는 커스텀 클래스로 해석
        self.policy = resolve_enum_or_custom_policy(policy) if isinstance(policy, str) else policy

    def __call__(self, img):
        # torchvision 방식인 경우
        if isinstance(self.policy, AutoAugmentPolicy):
            return transforms.AutoAugment(policy=self.policy)(img)
        # 사용자 커스텀 클래스라면
        return self.policy(img)
    
    

TRANSFORM_REGISTRY = {
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "AutoAugment": CustomAutoAugment,
    # "AutoAugment": transforms.AutoAugment,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
    "Resize": transforms.Resize,
    "CenterCrop": transforms.CenterCrop,
    # autoaugment
    "CIFAR10Policy": CIFAR10Policy,
    "SVHNPolicy": SVHNPolicy,
    "ImageNetPolicy": ImageNetPolicy,
}


def resolve_enum_or_custom_policy(policy_name: str):
    key = policy_name.lower()
    
    # ① PyTorch AutoAugmentPolicy Enum 값 처리
    try:
        return getattr(AutoAugmentPolicy, key.upper())
    except AttributeError:
        pass

    # ② 사용자 정의 정책 클래스 매핑
    if key == "custom_imagenet":
        return ImageNetPolicy()
    elif key == "custom_cifar10":
        return CIFAR10Policy()
    elif key == "custom_svhn":
        return SVHNPolicy()

    # ③ 실패할 경우 에러
    raise ValueError(f"[resolve_enum_or_custom_policy] Unknown policy: {policy_name}")


def resolve_enum_params(name, params):
    if name == "AutoAugment" and isinstance(params.get("policy"), str):
        params["policy"] = resolve_enum_or_custom_policy(params["policy"])
    return params



def build_transforms(transform_list):
    ops = []
    for item in transform_list:
        name = item["name"]
        params = resolve_enum_params(name, item.get("params", {}))
        transform_cls = TRANSFORM_REGISTRY[name]
        ops.append(transform_cls(**params) if params else transform_cls())
    return transforms.Compose(ops)


def load_transforms_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    train_tf = build_transforms(cfg["transforms"]["train"])
    val_tf = build_transforms(cfg["transforms"]["val"])
    return train_tf, val_tf
