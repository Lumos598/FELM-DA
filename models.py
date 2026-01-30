import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import numpy as np


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 64, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(192, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        x = self.fc(x)
        return x


class FashionMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 64, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(192, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc(x)
        return x


class FEMNIST():
    def __init__(self, num_classes=62):
        super(FEMNIST, self).__init__()
        self.model=resnet18(weights=ResNet18_Weights.DEFAULT)
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, 
                                    original_conv1.out_channels, 
                                    kernel_size=original_conv1.kernel_size,
                                    stride=original_conv1.stride, 
                                    padding=original_conv1.padding, 
                                    bias=original_conv1.bias)
        with torch.no_grad():
            self.model.conv1.weight = nn.Parameter(original_conv1.weight.sum(dim=1, keepdim=True))
        
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 512),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes))
    
    def get_model(self):    
        return self.model


class CIFAR100():
    def __init__(self):
        super(CIFAR100, self).__init__()
        self.model=resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.6),nn.Linear(num_ftrs, 100))
        
    def get_model(self):    
        return self.model
    
def data_augmentation(x):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter()
    ])
    return transform(x)
