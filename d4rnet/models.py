"""Student architecture from Section 3.1 of the D4RNet paper."""

import torch
from torch import nn
from torch.autograd import Function
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    mobilenet_v2,
    mobilenet_v3_large,
)


class _ReverseGradient(Function):
    @staticmethod
    def forward(ctx, features, strength):
        ctx.strength = strength
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class GradientReversal(nn.Module):
    """Identity in the forward pass; multiply backward gradients by -strength."""

    def __init__(self, strength=0.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, features):
        return _ReverseGradient.apply(features, self.strength)


class D4RNet(nn.Module):
    """MobileNet student with a training-only auxiliary domain classifier.

    Parameter names match the original DDD_DISTILL_GRL checkpoints. The GRL
    follows the projector, as in the original implementation and Algorithm 1.
    Setting num_domains=0 removes the entire auxiliary branch for deployment.
    """

    def __init__(self, backbone="mobilenet_v2", num_classes=22, num_domains=0, pretrained=False):
        super().__init__()
        if num_classes < 1 or num_domains < 0:
            raise ValueError("num_classes must be positive and num_domains nonnegative")
        if backbone == "mobilenet_v2":
            network = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
            feature_dim, auxiliary_dim = 1280, 24 + 32 + 96
        elif backbone == "mobilenet_v3_large":
            network = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            )
            feature_dim, auxiliary_dim = 960, 24 + 40 + 160
        else:
            raise ValueError(f"Unsupported student backbone: {backbone}")
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.features = network.features
        self.extract_layers = (3, 6, 13)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.main_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.grl_enabled = False
        if num_domains:
            self.aux_projection = nn.Sequential(
                nn.Linear(auxiliary_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
            self.grl = GradientReversal()
            self.aux_classifier = nn.Linear(128, num_domains)
        else:
            self.aux_projection = self.grl = self.aux_classifier = None

    def forward(self, images, return_aux=False):
        if return_aux and self.aux_classifier is None:
            raise ValueError("The auxiliary branch requires num_domains > 0")
        auxiliary = []
        for index, layer in enumerate(self.features):
            images = layer(images)
            if return_aux and index in self.extract_layers:
                auxiliary.append(self.avgpool(images).flatten(1))
        logits = self.main_classifier(self.avgpool(images).flatten(1))
        if not return_aux:
            return logits
        domain_features = self.aux_projection(torch.cat(auxiliary, dim=1))
        if self.grl_enabled:
            domain_features = self.grl(domain_features)
        return logits, self.aux_classifier(domain_features)

    def set_grl(self, enabled, strength):
        self.grl_enabled = bool(enabled)
        if self.grl is not None:
            self.grl.strength = float(strength)


def create_teacher(pretrained=False, num_classes=22):
    """Load timm lazily so student inference does not require timm."""
    import timm

    return timm.create_model(
        "swin_base_patch4_window7_224", pretrained=pretrained, num_classes=num_classes
    )
