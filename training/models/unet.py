import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is None:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            return self.conv(x)

        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet18UNet(nn.Module):
    def __init__(
        self,
        max_depth: float = 20.0,
        pretrained_encoder: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_encoder else None
        encoder = resnet18(weights=weights)

        self.max_depth = max_depth
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = UpBlock(64, 0, 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        input_size = image.shape[-2:]

        stem = self.stem(image)
        x = self.pool(stem)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.up4(layer4, layer3)
        x = self.up3(x, layer2)
        x = self.up2(x, layer1)
        x = self.up1(x, stem)
        x = self.up0(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return torch.sigmoid(self.output(x)) * self.max_depth
