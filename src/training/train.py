#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.model.dataset  import PADataset
from src.model.pa_model import PrivilegedAgent


def train(config_path='configs/config.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tr_cfg   = cfg['training']
    m_cfg    = cfg['model']
    p_cfg    = cfg['paths']
    DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(p_cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(p_cfg['log_dir'], exist_ok=True)

    # ── 데이터셋 ─────────────────────────────────────
    full_ds    = PADataset(p_cfg['dataset'], augment=tr_cfg['augment'])
    n          = len(full_ds)
    train_n    = int(n * tr_cfg['train_ratio'])
    val_n      = n - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n])

    train_loader = DataLoader(
        train_ds, batch_size=tr_cfg['batch_size'],
        shuffle=True,  num_workers=tr_cfg['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr_cfg['batch_size'],
        shuffle=False, num_workers=tr_cfg['num_workers'], pin_memory=True
    )
    print(f"Train: {train_n}  Val: {val_n}  Device: {DEVICE}")

    # ── 모델 ─────────────────────────────────────────
    model     = PrivilegedAgent(in_channels=m_cfg['in_channels']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg['lr'])
    scheduler = StepLR(optimizer,
                       step_size=tr_cfg['lr_step'],
                       gamma=tr_cfg['lr_gamma'])

    # ── 로그 파일 ─────────────────────────────────────
    log_file = open(os.path.join(p_cfg['log_dir'], 'train_log.csv'), 'w')
    log_file.write("epoch,train_l1,val_l1,val_dist_m\n")

    best_val = float('inf')
    EPOCHS   = tr_cfg['epochs']

    # ── 학습 루프 ─────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_loss = 0.0
        for M, v, w_gt in train_loader:
            M, v, w_gt = M.to(DEVICE), v.to(DEVICE), w_gt.to(DEVICE)
            w_pred     = model(M, v)
            loss       = F.l1_loss(w_pred, w_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dist = 0.0
        with torch.no_grad():
            for M, v, w_gt in val_loader:
                M, v, w_gt = M.to(DEVICE), v.to(DEVICE), w_gt.to(DEVICE)
                w_pred     = model(M, v)
                val_loss  += F.l1_loss(w_pred, w_gt).item()
                val_dist  += torch.norm(w_pred - w_gt, dim=-1).mean().item()

        val_loss /= len(val_loader)
        val_dist /= len(val_loader)
        scheduler.step()

        # 로그
        log_file.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dist:.4f}\n")
        log_file.flush()
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train L1: {train_loss:.4f} | "
              f"Val L1: {val_loss:.4f} | "
              f"Val Dist: {val_dist:.3f}m")

        # Best 저장
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_loss'   : val_loss,
                'val_dist'   : val_dist,
                'config'     : cfg,
            }, os.path.join(p_cfg['checkpoint_dir'], 'pa_best.pth'))
            print(f"  → Best 저장 (dist: {val_dist:.3f}m)")

        # 주기 저장
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(p_cfg['checkpoint_dir'], f'pa_epoch{epoch}.pth')
            )

    log_file.close()
    print(f"\n학습 완료. Best val_loss: {best_val:.4f}")


if __name__ == '__main__':
    train(config_path='configs/config.yaml')