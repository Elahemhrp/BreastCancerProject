import torch
import torch.nn as nn
from torchvision import models
from .config import Config


class BreastCancerModel(nn.Module):
    """
    Breast Cancer Classification Model with configurable backbone.
    
    Supported backbones:
    - resnet18: Smaller, faster, good for quick experiments
    - resnet34: Deeper, better accuracy, recommended
    - efficientnet_b0: Efficient architecture, good accuracy/speed tradeoff
    """
    
    def __init__(self, backbone_name=None, num_classes=None):
        super(BreastCancerModel, self).__init__()
        
        self.backbone_name = backbone_name or Config.BACKBONE
        self.num_classes = num_classes or Config.NUM_CLASSES
        
        if self.backbone_name == "resnet18":
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, self.num_classes)
            
        elif self.backbone_name == "resnet34":
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, self.num_classes)
            
        elif self.backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
            
        else:
            raise ValueError(f"Backbone {self.backbone_name} not supported. "
                           f"Choose from: resnet18, resnet34, efficientnet_b0")
            
    def forward(self, x):
        return self.backbone(x)
    
    def get_num_params(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze backbone layers for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always keep the final layer trainable
        if self.backbone_name in ["resnet18", "resnet34"]:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.backbone_name == "efficientnet_b0":
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
