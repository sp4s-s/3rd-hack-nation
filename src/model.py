import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CompactGeoEmbed(nn.Module):
    def __init__(self, embed_c=32, proj_dim=96, pretrained=False):
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=None if not pretrained else models.MobileNet_V2_Weights.IMAGENET1K_V1
        ).features
        self.backbone = backbone
        self.reduce = nn.Conv2d(1280, embed_c, 1)
        self.elev_conv = nn.Conv2d(1, embed_c, 3, padding=1)
        self.conv_head = nn.Sequential(
            nn.Conv2d(embed_c * 2, embed_c, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(embed_c, embed_c, 3, padding=1),
            nn.ReLU(True),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_c, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Linear(proj_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, elev=None):
        x = self.backbone(img)
        x = self.reduce(x)
        if elev is None:
            elev = torch.zeros(x.size(0), 1, img.size(2), img.size(3), device=x.device)
        elev = F.interpolate(elev, size=x.shape[2:], mode="bilinear", align_corners=False)
        e = self.elev_conv(elev)
        x = self.conv_head(torch.cat([x, e], 1))
        p = self.proj(x)
        p = F.normalize(p, dim=1)
        risk = self.risk_head(p).squeeze(-1)
        return x, p, risk
