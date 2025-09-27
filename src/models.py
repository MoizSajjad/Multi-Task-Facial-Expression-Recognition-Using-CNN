import torch
import torch.nn as nn
import torchvision.models as tv

class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 8):
        super().__init__()
        self.cls = nn.Linear(in_dim, num_classes)
        self.va  = nn.Linear(in_dim, 2)  # valence, arousal

    def forward(self, feats):
        return self.cls(feats), self.va(feats)

class MultiTaskNet(nn.Module):
    """
    backbone ∈ {"resnet18", "efficientnet_b0"}
    """
    def __init__(self, backbone: str = "resnet18", num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "resnet18":
            try:
                weights = tv.ResNet18_Weights.DEFAULT if pretrained else None
                m = tv.resnet18(weights=weights)
            except Exception:
                m = tv.resnet18(weights=None)
            in_dim = m.fc.in_features
            # keep everything except the final FC
            self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> [B, 512, 1, 1]
            self.pool = nn.Identity()
            in_dim = in_dim

        elif self.backbone_name == "efficientnet_b0":
            try:
                weights = tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                m = tv.efficientnet_b0(weights=weights)
            except Exception:
                m = tv.efficientnet_b0(weights=None)
            # EfficientNet already has avgpool before classifier
            in_dim = m.classifier[1].in_features  # 1280
            m.classifier = nn.Identity()
            self.backbone = m  # returns [B, 1280] after flatten
            self.pool = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.head = MultiTaskHead(in_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        if f.ndim == 4:  # e.g., resnet outputs [B, C, 1, 1]
            f = f.flatten(1)
        logits, va = self.head(f)
        return logits, va
