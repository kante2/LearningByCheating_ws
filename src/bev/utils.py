import numpy as np
import cv2


def world_to_pixel(lx, ly, map_size=320, resolution=0.2):
    """
    base_link 좌표 → BEV 픽셀 좌표
    lx: 전방 (+x), ly: 좌측 (+y)
    Ego = 하단 중앙 (map_size/2, map_size)
    """
    px = int(map_size / 2 + ly / resolution)
    py = int(map_size     - lx / resolution)
    return px, py


def pixel_to_world(px, py, map_size=320, resolution=0.2):
    """BEV 픽셀 좌표 → base_link 좌표 (미터)"""
    ly = (px - map_size / 2) * resolution
    lx = (map_size - py)     * resolution
    return lx, ly


def is_in_map(px, py, map_size=320):
    return 0 <= px < map_size and 0 <= py < map_size