import numpy as np
import cv2
from .utils import world_to_pixel, is_in_map


class BEVRenderer:
    def __init__(self, map_size=320, resolution=0.2):
        self.map_size   = map_size
        self.resolution = resolution

        self.road_layer = None
        self.lane_layer = None

    # ── 정적 레이어 초기화 (한 번만 호출) ────────────────
    def init_static_layers(self, road_polygons, lane_lines):
        """
        road_polygons: [[(lx,ly), ...], ...]  base_link 기준 폴리곤
        lane_lines:    [[(lx,ly), ...], ...]  차선 라인
        """
        S = self.map_size
        R = self.resolution

        self.road_layer = np.zeros((S, S), dtype=np.uint8)
        self.lane_layer = np.zeros((S, S), dtype=np.uint8)

        for poly in road_polygons:
            pts = np.array([
                [int(S/2 + ly/R), int(S - lx/R)]
                for lx, ly in poly
            ], dtype=np.int32)
            cv2.fillPoly(self.road_layer, [pts], 1)

        for line in lane_lines:
            pts = [
                (int(S/2 + ly/R), int(S - lx/R))
                for lx, ly in line
            ]
            for i in range(len(pts) - 1):
                cv2.line(self.lane_layer, pts[i], pts[i+1], 1, thickness=2)

    # ── NPC 채널 렌더링 ───────────────────────────────
    def render_npcs(self, npc_list):
        S, R = self.map_size, self.resolution
        ch = np.zeros((S, S), dtype=np.uint8)
        for npc in npc_list:
            lx, ly = npc['lx'], npc['ly']
            px, py = world_to_pixel(lx, ly, S, R)
            if is_in_map(px, py, S):
                w = max(1, int(npc.get('width',  2.0) / R / 2))
                h = max(1, int(npc.get('length', 4.5) / R / 2))
                cv2.rectangle(ch, (px-w, py-h), (px+w, py+h), 1, -1)
        return ch

    # ── Path 채널 렌더링 ──────────────────────────────
    def render_path(self, path_points):
        S, R = self.map_size, self.resolution
        ch = np.zeros((S, S), dtype=np.uint8)
        if not path_points or len(path_points) < 2:
            return ch

        prev = None
        for lx, ly in path_points:
            px, py = world_to_pixel(lx, ly, S, R)
            if is_in_map(px, py, S):
                if prev is not None:
                    cv2.line(ch, prev, (px, py), 1, thickness=3)
                prev = (px, py)
        return ch

    # ── 전체 맵 생성 ──────────────────────────────────
    def render(self, npc_list, path_points):
        """
        매 프레임 호출
        Returns: M (map_size, map_size, 4) uint8
        """
        assert self.road_layer is not None, \
            "init_static_layers() 먼저 호출하세요"

        M = np.zeros((self.map_size, self.map_size, 4), dtype=np.uint8)
        M[:, :, 0] = self.road_layer
        M[:, :, 1] = self.lane_layer
        M[:, :, 2] = self.render_npcs(npc_list)
        M[:, :, 3] = self.render_path(path_points)
        return M

    # ── 시각화용 ─────────────────────────────────────
    def to_color(self, M, waypoint=None):
        vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        vis[M[:, :, 0] == 1] = [42,  74,  53]
        vis[M[:, :, 1] == 1] = [200, 200, 112]
        vis[M[:, :, 2] == 1] = [51,  102, 187]
        vis[M[:, :, 3] == 1] = [232, 112,  64]

        # Ego
        cx = self.map_size // 2
        cy = int(self.map_size * 0.88)
        cv2.rectangle(vis, (cx-8, cy-12), (cx+8, cy+12), (204, 68, 68), -1)

        # VALOR waypoint
        if waypoint is not None:
            lx, ly = waypoint
            px, py = world_to_pixel(lx, ly, self.map_size, self.resolution)
            if is_in_map(px, py, self.map_size):
                cv2.circle(vis, (px, py), 6, (255, 255, 0), -1)

        return vis