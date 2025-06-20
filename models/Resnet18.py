__all_ = ['Resnet18Model']
import torchvision.models as models
from torch import nn


class Resnet18Model(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18Model, self).__init__()
        self.backbone = models.resnet18(pretrained=True)  # ResNet18 모델 불러오기
        self.feature_dim = self.backbone.fc.in_features 
        self.backbone.fc = nn.Identity()  # feature extractor로만 사용
        self.head = nn.Linear(self.feature_dim, num_classes)  # 분류기

    def forward(self, x):
        x = self.backbone(x)       
        x = self.head(x) 
        return x