import torch
import torch.nn as nn
from torchvision.models import resnet18


class PrivilegedAgent(nn.Module):
    def __init__(self, in_channels=4, map_size=320, resolution=0.2):
        super().__init__()
        self.map_size   = map_size
        self.resolution = resolution

        # ResNet-18 백본 (랜덤 초기화)
        backbone = resnet18(pretrained=False)
        backbone.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Up-conv 3개 (512→256→128→64)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True)
        )

        # velocity 주입
        self.vel_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True)
        )

        # K=1 waypoint heatmap
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def soft_argmax(self, heatmap):
        """heatmap (B,1,H,W) → 픽셀 좌표 (B,2)"""
        B, _, H, W = heatmap.shape
        flat    = heatmap.reshape(B, -1)
        flat    = torch.softmax(flat, dim=-1)
        flat    = flat.reshape(B, H, W)

        xs = torch.linspace(0, W-1, W, device=heatmap.device)
        ys = torch.linspace(0, H-1, H, device=heatmap.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')

        px = (flat * gx).sum(dim=(-2, -1))
        py = (flat * gy).sum(dim=(-2, -1))
        return torch.stack([px, py], dim=-1)    # (B, 2) 픽셀

    def pixel_to_meter(self, pixel_coords):
        """픽셀 좌표 → base_link 미터 좌표"""
        S, R = self.map_size, self.resolution
        px   = pixel_coords[:, 0]
        py   = pixel_coords[:, 1]
        ly   = (px - S / 2) * R    # 좌우
        lx   = (S  - py)    * R    # 전후
        return torch.stack([lx, ly], dim=-1)    # (B, 2) 미터

    def forward(self, M, v):
        feat = self.backbone(M)         # (B, 512, H', W')
        feat = self.up1(feat)
        feat = self.up2(feat)
        feat = self.up3(feat)           # (B, 64, H'', W'')

        vel  = self.vel_fc(v)           # (B, 64)
        vel  = vel.unsqueeze(-1).unsqueeze(-1)
        feat = feat + vel

        heatmap  = self.head(feat)              # (B, 1, H'', W'')
        px_coord = self.soft_argmax(heatmap)    # (B, 2) 픽셀
        waypoint = self.pixel_to_meter(px_coord)# (B, 2) 미터

        return waypoint