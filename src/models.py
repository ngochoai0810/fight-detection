"""M1: CNN phan loai frame. M3: CNN + LSTM phan loai chuoi frame."""
import torch
from torch import nn
from torchvision import models as tv


def build_backbone(arch, pretrained=True):
    """Tra ve mang trich dac trung va so chieu dac trung."""
    if arch == "resnet18":
        net = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif arch == "mobilenet_v2":
        net = tv.mobilenet_v2(weights=tv.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        dim = net.last_channel
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"Kien truc khong ho tro: {arch}")
    return net, dim


class FrameCNN(nn.Module):
    def __init__(self, arch="resnet18", pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone, dim = build_backbone(arch, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def build_cnn(arch="resnet18", pretrained=True):
    return FrameCNN(arch, pretrained)


class CNNLSTM(nn.Module):
    def __init__(self, arch="resnet18", hidden_size=256, pretrained=True, freeze_backbone=False, dropout=0.3):
        super().__init__()
        self.backbone, dim = build_backbone(arch, pretrained)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.lstm = nn.LSTM(dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # giu nguyen thong ke BatchNorm khi dong bang
        return self

    def forward(self, clips):
        batch, steps = clips.shape[:2]
        with torch.set_grad_enabled(self.training and not self.freeze_backbone):
            features = self.backbone(clips.flatten(0, 1))
        _, (hidden, _) = self.lstm(features.view(batch, steps, -1))
        return self.head(hidden[-1]).squeeze(-1)
