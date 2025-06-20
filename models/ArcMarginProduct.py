__all__ = ['ArcMarginProduct']
import torch
import torch.nn as nn
import torch.nn.functional as F


# ArcMarginProduct 구현
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # scale factor
        self.m = torch.tensor(m, dtype=torch.float32)  # margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)
        self.th = torch.cos(torch.pi - self.m)
        self.mm = torch.sin(torch.pi - self.m) * self.m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # cos(θ)
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # only apply margin to correct class
        output *= self.s
        return output
