__all_ = ['Resnet50Model']
import torchvision.models as models
from torch import nn


class Resnet50Model(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50Model, self).__init__()
        self.backbone = models.resnet50(pretrained=True)  # ResNet50 모델 불러오기
        self.feature_dim = self.backbone.fc.in_features 
        self.backbone.fc = nn.Identity()  # feature extractor로만 사용
        self.head = nn.Linear(self.feature_dim, num_classes)  # 분류기

    def forward(self, x, labels=None):
        x = self.backbone(x)       
        x = self.head(x) 
        return x