#!/usr/bin/env python3
"""
수집된 데이터셋 빠른 검증
python check_data.py
"""
import pickle
import numpy as np
import sys

def check(path='data/raw/dataset.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"총 프레임: {len(data)}")
    if not data:
        print("데이터 없음")
        return

    sample = data[0]
    print(f"\n[샘플 확인]")
    print(f"  bev shape:  {sample['bev'].shape}   (기대: (320,320,4))")
    print(f"  waypoint:   {sample['waypoint']}    (기대: (x,y) float)")
    print(f"  speed:      {sample['speed']:.2f} m/s")

    wps    = np.array([d['waypoint'] for d in data])
    speeds = np.array([d['speed']    for d in data])

    print(f"\n[waypoint 통계]")
    print(f"  x (전방): mean={wps[:,0].mean():.2f}  "
          f"std={wps[:,0].std():.2f}  "
          f"min={wps[:,0].min():.2f}  max={wps[:,0].max():.2f}")
    print(f"  y (좌우): mean={wps[:,1].mean():.2f}  "
          f"std={wps[:,1].std():.2f}  "
          f"min={wps[:,1].min():.2f}  max={wps[:,1].max():.2f}")

    print(f"\n[속도 통계]")
    print(f"  mean={speeds.mean():.2f}  "
          f"std={speeds.std():.2f}  "
          f"min={speeds.min():.2f}  max={speeds.max():.2f} m/s")

    print(f"\n[BEV 채널별 활성화 비율]")
    for i, name in enumerate(['도로', '차선', 'NPC', 'Path']):
        ratio = np.mean([d['bev'][:,:,i].sum() > 0 for d in data])
        print(f"  Ch{i+1} ({name}): {ratio*100:.1f}% 프레임에서 활성화")

    print("\n데이터 검증 완료!")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/dataset.pkl'
    check(path)