__all__ = ['Resnet50HeadArc']
import torch.nn as nn
import timm

from models.ArcMarginProduct import ArcMarginProduct


class Resnet50HeadArc(nn.Module):
    def __init__(self, num_classes=396):
        super(Resnet50HeadArc, self).__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)  # remove fc
        self.neck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.head = ArcMarginProduct(512, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = self.neck(features)
        if labels is not None:
            logits = self.head(features, labels)
        else:
            logits = nn.functional.linear(nn.functional.normalize(features), nn.functional.normalize(self.head.weight))
        return logits
