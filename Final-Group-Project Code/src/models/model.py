# 2

# def create_model(num_classes=9):
import torch
import torch.nn as nn
from torchvision import models

class create_model(nn.Module):
    def __init__(self, num_classes=9):
        super(create_model, self).__init__()

        # Load pre-trained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the original fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove it and replace in forward()

        # New classification head
        self.fc1 = nn.Linear(in_features, 256)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)  # feature extraction
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.output_activation(x)
        return x
