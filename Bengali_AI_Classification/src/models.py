import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet34(weights=None)

        # Remove the fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)
    
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2
