from typing import Optional, override

import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18  # type: ignore


class ResNet3DModel(nn.Module):
    def __init__(self, num_classes: int, weights: Optional[R3D_18_Weights] = R3D_18_Weights.DEFAULT):
        """
        Initialize the ResNet3DModel with a 3D ResNet architecture.

        Args:
            num_classes (int): The number of output classes for classification.
            weights (Optional[R3D_18_Weights]): Pretrained weights for the model.
        """
        super(ResNet3DModel, self).__init__()
        self.resnet3d = r3d_18(weights=weights)

        # Replace the final fully connected layer to match the number of action classes
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

        # Freeze all layers except the final fully connected layer
        for param in self.resnet3d.parameters():
            param.requires_grad = False
        for param in self.resnet3d.fc.parameters():
            param.requires_grad = True

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes).
        """
        return self.resnet3d(x)

    def unfreeze_layers(self, layers_to_unfreeze: Optional[int] = None) -> None:
        """
        Unfreeze the specified number of layers for fine-tuning.

        Args:
            layers_to_unfreeze (Optional[int]): The number of layers to unfreeze from the top.
        """
        if layers_to_unfreeze is not None:
            layers = list(self.resnet3d.children())
            for layer in layers[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
