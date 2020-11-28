import torch
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Linear, Module
from torch.nn.functional import softmax
from torch.nn.functional import relu
from torch.nn import BatchNorm1d

# Define yoga model with the backbone from dense pose
class YogaPoseEstimatorModel(Module):
    def __init__(self, backbone, num_classes, pixel_mean, pixel_std):
        super().__init__()
        self.backbone = backbone
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc1 = Linear(256, 64)
        self.bn1 = BatchNorm1d(64)
        self.fc2 = Linear(64, num_classes)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)['p6']
        x = self.avg_pool(x)
        x = x.view((-1, 256))
        x = self.fc1(x)
        x = self.bn1(x)
        x = relu(x)
        x = self.fc2(x)
        x = softmax(x, dim=1)
        return x

    def preprocess(self, tensor):
        # Preprocessing from the source code for the original model
        tensor = (tensor - self.pixel_mean) / self.pixel_std
        return tensor