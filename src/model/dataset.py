import pickle
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class PADataset(Dataset):
    def __init__(self, data_path, augment=False):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.augment = augment
        print(f"[Dataset] {len(self.data)} 프레임 로드 | augment={augment}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # BEV: (H,W,C) → (C,H,W), float32, 0~1
        M = torch.tensor(item['bev'], dtype=torch.float32)
        M = M.permute(2, 0, 1) / 255.0        # (4, 320, 320)

        # 속도
        v = torch.tensor([item['speed']], dtype=torch.float32)

        # 정답 waypoint
        w = torch.tensor(item['waypoint'], dtype=torch.float32)  # (2,)

        # 증강
        if self.augment:
            M, w = self._augment(M, w)

        return M, v, w

    def _augment(self, M, w):
        angle = float(np.random.uniform(-5, 5))    # ±5도
        shift = float(np.random.uniform(-5, 5))    # ±5 픽셀

        C, H, W = M.shape
        center  = (W / 2.0, H / 2.0)
        mat     = cv2.getRotationMatrix2D(center, angle, 1.0)
        mat[0, 2] += shift

        M_aug = torch.zeros_like(M)
        for c in range(C):
            ch     = M[c].numpy().astype(np.float32)
            ch_aug = cv2.warpAffine(ch, mat, (W, H), flags=cv2.INTER_LINEAR)
            M_aug[c] = torch.tensor(ch_aug)

        # waypoint에도 동일 회전 + 이동
        rad   = np.radians(-angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        wx, wy = w[0].item(), w[1].item()
        wx_new = wx * cos_a - wy * sin_a
        wy_new = wx * sin_a + wy * cos_a + shift * 0.2
        w_aug  = torch.tensor([wx_new, wy_new], dtype=torch.float32)

        return M_aug, w_aug