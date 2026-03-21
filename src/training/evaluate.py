#!/usr/bin/env python3
"""
학습된 PA 모델 평가
python src/training/evaluate.py --ckpt checkpoints/pa_best.pth
"""
import os
import sys
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.model.dataset  import PADataset
from src.model.pa_model import PrivilegedAgent
from torch.utils.data   import DataLoader


def evaluate(ckpt_path, config_path='configs/config.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 로드
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = PrivilegedAgent(in_channels=cfg['model']['in_channels']).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"모델 로드: epoch={ckpt['epoch']}, val_dist={ckpt['val_dist']:.3f}m")

    # 데이터셋
    ds     = PADataset(cfg['paths']['dataset'], augment=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    errors = []
    x_errors, y_errors = [], []

    with torch.no_grad():
        for M, v, w_gt in loader:
            M, v, w_gt = M.to(DEVICE), v.to(DEVICE), w_gt.to(DEVICE)
            w_pred     = model(M, v)
            err        = (w_pred - w_gt).cpu().numpy()
            dist       = np.linalg.norm(err, axis=1)
            errors.extend(dist.tolist())
            x_errors.extend(np.abs(err[:, 0]).tolist())
            y_errors.extend(np.abs(err[:, 1]).tolist())

    print(f"\n전체 {len(errors)} 샘플 평가 결과")
    print(f"  평균 거리 오차:  {np.mean(errors):.3f} m")
    print(f"  중앙값 오차:     {np.median(errors):.3f} m")
    print(f"  최대 오차:       {np.max(errors):.3f} m")
    print(f"  x 오차 (전후):   {np.mean(x_errors):.3f} m")
    print(f"  y 오차 (좌우):   {np.mean(y_errors):.3f} m")
    print(f"  0.5m 이내 비율: {np.mean(np.array(errors) < 0.5)*100:.1f}%")
    print(f"  1.0m 이내 비율: {np.mean(np.array(errors) < 1.0)*100:.1f}%")

    # 오차 분포 시각화
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('거리 오차 (m)')
    plt.ylabel('빈도')
    plt.title('PA 모델 waypoint 예측 오차 분포')
    plt.axvline(np.mean(errors), color='r', linestyle='--',
                label=f'평균: {np.mean(errors):.3f}m')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['paths']['log_dir'], 'error_dist.png'))
    print(f"\n오차 분포 저장: {cfg['paths']['log_dir']}/error_dist.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   default='checkpoints/pa_best.pth')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    evaluate(args.ckpt, args.config)