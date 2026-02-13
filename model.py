
import torch.nn as nn
import torchvision.models as models
from config import *

def resnet18_cifar():
    model = models.resnet18(weights=None)


    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(512, num_classes)
    return model
