#!/usr/bin/env python3
"""
PA 데이터 수집 ROS 노드
실행: rosrun <package> data_collector.py
"""
import rospy
import pickle
import os
import sys
import cv2

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.bev.renderer import BEVRenderer

import yaml


class PADataCollector:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        bev_cfg = self.cfg['bev']
        self.renderer = BEVRenderer(
            map_size   = bev_cfg['map_size'],
            resolution = bev_cfg['resolution']
        )

        # 수신 데이터
        self.waypoint  = None   # (x, y) — PA 정답
        self.path_pts  = []     # [(x,y), ...] — Ch4용
        self.speed     = 0.0
        self.npc_list  = []     # NPC 리스트

        # 데이터셋
        self.dataset   = []
        self.save_path = self.cfg['paths']['dataset']
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # ROS 구독
        ros_cfg = self.cfg['ros']
        rospy.Subscriber(ros_cfg['waypoint_topic'], PointStamped, self.cb_waypoint)
        rospy.Subscriber(ros_cfg['path_topic'],     Path,         self.cb_path)
        rospy.Subscriber(ros_cfg['speed_topic'],    Float32,      self.cb_speed)

        rospy.on_shutdown(self.save)
        rospy.loginfo("[DataCollector] 초기화 완료")

    # ── 콜백 ─────────────────────────────────────────
    def cb_waypoint(self, msg):
        self.waypoint = (msg.point.x, msg.point.y)

    def cb_path(self, msg):
        self.path_pts = [
            (p.pose.position.x, p.pose.position.y)
            for p in msg.poses
        ]

    def cb_speed(self, msg):
        self.speed = msg.data

    # ── 정적 레이어 초기화 ────────────────────────────
    def init_static_layers(self, road_polygons, lane_lines):
        self.renderer.init_static_layers(road_polygons, lane_lines)
        rospy.loginfo("[DataCollector] 정적 레이어 초기화 완료")

    # ── 프레임 수집 ───────────────────────────────────
    def collect_frame(self, visualize=False):
        if self.waypoint is None:
            rospy.logwarn_throttle(2.0, "[DataCollector] waypoint 미수신")
            return False
        if self.renderer.road_layer is None:
            rospy.logwarn_throttle(2.0, "[DataCollector] 정적 레이어 미초기화")
            return False

        # BEV 맵 생성
        M = self.renderer.render(
            npc_list    = self.npc_list,
            path_points = self.path_pts
        )

        # 저장
        self.dataset.append({
            'bev'     : M,                  # (320, 320, 4) uint8
            'waypoint': self.waypoint,      # (x, y) float, base_link 미터
            'speed'   : self.speed          # float, m/s
        })

        # 시각화
        if visualize:
            vis = self.renderer.to_color(M, waypoint=self.waypoint)
            cv2.imshow('BEV Map', vis)
            cv2.waitKey(1)

        if len(self.dataset) % 100 == 0:
            rospy.loginfo(f"[DataCollector] {len(self.dataset)} 프레임 수집")

        return True

    # ── 저장 ─────────────────────────────────────────
    def save(self):
        if not self.dataset:
            rospy.logwarn("[DataCollector] 저장할 데이터 없음")
            return
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.dataset, f)
        rospy.loginfo(
            f"[DataCollector] 저장 완료: "
            f"{len(self.dataset)} 프레임 → {self.save_path}"
        )

    def __len__(self):
        return len(self.dataset)


# ── 메인 ─────────────────────────────────────────────
if __name__ == '__main__':
    rospy.init_node('pa_data_collector')

    collector = PADataCollector(config_path='configs/config.yaml')

    # TODO: 실제 road_polygons, lane_lines 로드 후 호출
    # collector.init_static_layers(road_polygons, lane_lines)

    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        collector.collect_frame(visualize=True)
        rate.sleep()