import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def compute_patch_entropy(gray_patch, num_bins=16):
    """
    gray_patch: (B, 1, H, W)
    """
    b, _, h, w = gray_patch.shape
    flat = gray_patch.view(b, -1)
    # normalize to [0,1]
    flat = (flat - flat.min(dim=1, keepdim=True)[0]) / (
        flat.max(dim=1, keepdim=True)[0] - flat.min(dim=1, keepdim=True)[0] + 1e-6
    )
    hist = []
    for i in range(b):
        h_i = torch.histc(flat[i], bins=num_bins, min=0.0, max=1.0)
        p_i = h_i / (h_i.sum() + 1e-6)
        hist.append(p_i)
    hist = torch.stack(hist, dim=0)  # (B, num_bins)
    entropy = -(hist * (hist + 1e-10).log()).sum(dim=1)  # (B,)
    return entropy


class DensityAwareDynamicEncoder(nn.Module):
    """
    A simplified DDE: for each image, we compute a global grayscale entropy
    and map it to one of K levels, then choose encoder depth accordingly.
    """

    def __init__(self, in_channels=3, base_channels=64, num_levels=3, seg_num_classes=2):
        super().__init__()
        self.num_levels = num_levels

        # shared encoder blocks
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # decoder shared for all levels
        self.up1 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up2 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up3 = Up(base_channels * 2 + base_channels, base_channels)

        self.outc = OutConv(base_channels, seg_num_classes)

        # learnable thresholds (initialized to reasonable values, but can be overridden)
        self.register_buffer("entropy_thresholds", torch.linspace(0.5, 2.0, steps=num_levels - 1))

    def set_entropy_thresholds(self, thresholds: torch.Tensor):
        with torch.no_grad():
            self.entropy_thresholds.copy_(thresholds)

    def forward(self, x):
        # x: (B,3,H,W)
        # compute grayscale and entropy per image
        with torch.no_grad():
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            entropy = compute_patch_entropy(gray)  # (B,)
            # map entropy to levels 0..K-1
            levels = torch.zeros_like(entropy, dtype=torch.long)
            for t in self.entropy_thresholds:
                levels += (entropy > t).long()

        # shared encoder
        x1 = self.inc(x)        # (B, C)
        x2 = self.down1(x1)     # (B, 2C)
        x3 = self.down2(x2)     # (B, 4C)
        x4 = self.down3(x3)     # (B, 8C)

        # For simplicity, use same depth but attach level embedding as a global feature
        level_embed = F.one_hot(levels, num_classes=self.num_levels).float()  # (B, K)

        # decoder
        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)
        logits = self.outc(u3)  # (B, seg_num_classes, H, W)

        # global feature for dual-domain / anchor usage
        global_feat = F.adaptive_avg_pool2d(u3, output_size=1).view(x.size(0), -1)  # (B, C)
        global_feat = torch.cat([global_feat, level_embed.to(global_feat.device)], dim=1)

        return logits, u3, global_feat


def build_dde_unet(seg_num_classes: int = 2, base_channels: int = 64, num_levels: int = 3):
    return DensityAwareDynamicEncoder(
        in_channels=3,
        base_channels=base_channels,
        num_levels=num_levels,
        seg_num_classes=seg_num_classes,
    )

