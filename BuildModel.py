import torch
import torch.nn as nn
from torchvision.models import resnet101


class Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = resnet101(pretrained=True)
        self.num_features = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad(False)
        self.model.fc = nn.Linear(self.num_features, num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def get_model(self):
        return self.model
