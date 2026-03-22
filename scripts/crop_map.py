#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList


class LocalBEVNav:
    def __init__(self):
        rospy.init_node("local_bev_nav", anonymous=False)

        self.map_path = "/home/autonav/jang_ws/src/yolo_detector/data/R_KR_PG_KATRI"

        # 64m x 64m local map
        self.map_size_m = 64.0
        self.half_m = self.map_size_m / 2.0

        # grid resolution
        self.resolution = 0.2   # m/cell
        self.width = int(self.map_size_m / self.resolution)
        self.height = int(self.map_size_m / self.resolution)

        # local frame
        self.frame_id = "base_link"

        # ego vehicle: 2023 Hyundai IONIQ 5
        self.ego_size = (4.635, 1.890, 1.605)
        self.ego_rear_to_center = 0.60

        # fallback object sizes
        self.default_npc_size = (4.4, 1.8, 1.5)
        self.default_ped_size = (0.6, 0.6, 1.7)
        self.default_obs_size = (1.0, 1.0, 1.0)

        self.ego_msg = None
        self.obj_msg = None

        self.grid_pub = rospy.Publisher("/local_bev_grid", OccupancyGrid, queue_size=1)
        self.obj_pub = rospy.Publisher("/local_bev_objects", MarkerArray, queue_size=1)
        self.ego_pub = rospy.Publisher("/local_bev_ego", Marker, queue_size=1)

        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_callback)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.obj_callback)

        self.lane_marking_set = self._load_json(os.path.join(self.map_path, "lane_marking_set.json"))
        self.singlecrosswalk_set = self._load_json(os.path.join(self.map_path, "singlecrosswalk_set.json"))
        self.surface_marking_set = self._load_json(os.path.join(self.map_path, "surface_marking_set.json"))

        rospy.loginfo("local_bev_nav started")

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def ego_callback(self, msg):
        self.ego_msg = msg

    def obj_callback(self, msg):
        self.obj_msg = msg

    def map_to_local(self, x, y, z=0.0):
        ego = self.ego_msg
        ex = ego.position.x
        ey = ego.position.y
        ez = ego.position.z
        yaw = math.radians(ego.heading)

        dx = x - ex
        dy = y - ey

        c = math.cos(yaw)
        s = math.sin(yaw)

        # local x: forward, local y: left/right
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        lz = z - ez

        return lx, ly, lz

    def heading_to_local_yaw(self, heading_deg):
        ego_yaw = math.radians(self.ego_msg.heading)
        obj_yaw = math.radians(heading_deg)
        return obj_yaw - ego_yaw

    def in_crop(self, x, y):
        return (-self.half_m <= x <= self.half_m and
                -self.half_m <= y <= self.half_m)

    def metric_to_grid(self, x, y):
        """
        base_link-centered metric coord -> grid index
        grid origin is bottom-left (-half_m, -half_m)
        """
        gx = int((x + self.half_m) / self.resolution)
        gy = int((y + self.half_m) / self.resolution)
        return gx, gy

    def valid_grid(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def set_cell(self, grid, gx, gy, value):
        if self.valid_grid(gx, gy):
            idx = gy * self.width + gx
            if value > grid[idx]:
                grid[idx] = value

    def draw_circle(self, grid, x, y, radius_m, value):
        gx, gy = self.metric_to_grid(x, y)
        rr = max(1, int(radius_m / self.resolution))
        for iy in range(gy - rr, gy + rr + 1):
            for ix in range(gx - rr, gx + rr + 1):
                if not self.valid_grid(ix, iy):
                    continue
                if (ix - gx) ** 2 + (iy - gy) ** 2 <= rr * rr:
                    self.set_cell(grid, ix, iy, value)

    def draw_line_metric(self, grid, x1, y1, x2, y2, thickness_m, value):
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(2, int(dist / (self.resolution * 0.5)))
        for i in range(steps + 1):
            t = float(i) / float(steps)
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            self.draw_circle(grid, x, y, thickness_m * 0.5, value)

    def fill_polygon_metric(self, grid, pts, value):
        """
        simple scan fill in metric space
        pts: [(x,y), ...]
        """
        if len(pts) < 3:
            return

        ys = [p[1] for p in pts]
        y_min = max(-self.half_m, min(ys))
        y_max = min(self.half_m, max(ys))

        gy_min = max(0, int((y_min + self.half_m) / self.resolution))
        gy_max = min(self.height - 1, int((y_max + self.half_m) / self.resolution))

        n = len(pts)
        for gy in range(gy_min, gy_max + 1):
            y = gy * self.resolution - self.half_m + self.resolution * 0.5
            intersections = []

            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]

                if abs(y2 - y1) < 1e-6:
                    continue

                cond = (y1 <= y < y2) or (y2 <= y < y1)
                if cond:
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(x)

            intersections.sort()
            for i in range(0, len(intersections) - 1, 2):
                x_start = intersections[i]
                x_end = intersections[i + 1]

                gx_start = max(0, int((x_start + self.half_m) / self.resolution))
                gx_end = min(self.width - 1, int((x_end + self.half_m) / self.resolution))

                for gx in range(gx_start, gx_end + 1):
                    self.set_cell(grid, gx, gy, value)

    def build_grid(self):
        grid = [0] * (self.width * self.height)

        # 1) lane markings
        for item in self.lane_marking_set:
            pts = item.get("points", [])
            if len(pts) < 2:
                continue

            local_pts = []
            inside = False
            for p in pts:
                x, y, z = self.map_to_local(p[0], p[1], p[2] if len(p) > 2 else 0.0)
                local_pts.append((x, y))
                if self.in_crop(x, y):
                    inside = True

            if not inside:
                continue

            lane_width = float(item.get("lane_width", 0.15))
            thickness = max(0.12, lane_width * 0.8)

            for i in range(len(local_pts) - 1):
                x1, y1 = local_pts[i]
                x2, y2 = local_pts[i + 1]
                self.draw_line_metric(grid, x1, y1, x2, y2, thickness, 100)

        # 2) crosswalks
        for item in self.singlecrosswalk_set:
            pts = item.get("points", [])
            if len(pts) < 3:
                continue

            local_poly = []
            inside = False
            for p in pts:
                x, y, z = self.map_to_local(p[0], p[1], p[2] if len(p) > 2 else 0.0)
                local_poly.append((x, y))
                if self.in_crop(x, y):
                    inside = True

            if not inside:
                continue

            self.fill_polygon_metric(grid, local_poly, 70)

        # 3) surface markings
        for item in self.surface_marking_set:
            pts = item.get("points", [])
            if len(pts) < 3:
                continue

            local_poly = []
            inside = False
            for p in pts:
                x, y, z = self.map_to_local(p[0], p[1], p[2] if len(p) > 2 else 0.0)
                local_poly.append((x, y))
                if self.in_crop(x, y):
                    inside = True

            if not inside:
                continue

            self.fill_polygon_metric(grid, local_poly, 45)

        return grid

    def get_object_size(self, obj, default_size):
        try:
            # user-confirmed mapping in your environment
            length = float(obj.size.x)
            width = float(obj.size.y)
            height = float(obj.size.z)

            if length <= 0.01 or width <= 0.01 or height <= 0.01:
                return default_size
            return (length, width, height)
        except Exception:
            return default_size

    def get_rear_to_center_offset(self, size):
        length = size[0]
        if length >= 10.0:
            return 3.0
        elif length >= 6.0:
            return 2.2
        elif length >= 5.0:
            return 1.7
        elif length >= 4.3:
            return 1.5
        else:
            return 1.2

    def quat_from_yaw(self, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return 0.0, 0.0, sy, cy

    def make_cube_marker(self, mid, ns, x, y, z, yaw, size, rgba):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + size[2] * 0.5

        qx, qy, qz, qw = self.quat_from_yaw(yaw)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        m.scale.x = size[0]
        m.scale.y = size[1]
        m.scale.z = size[2]

        m.color.r = rgba[0]
        m.color.g = rgba[1]
        m.color.b = rgba[2]
        m.color.a = rgba[3]

        m.lifetime = rospy.Duration(0.15)
        return m

    def make_text_marker(self, mid, ns, x, y, z, text, scale=0.6):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + 2.0
        m.pose.orientation.w = 1.0
        m.scale.z = scale
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.text = text
        m.lifetime = rospy.Duration(0.15)
        return m

    def build_object_markers(self):
        ma = MarkerArray()
        mid = 0

        # ego fixed at local origin
        ego_cx = self.ego_rear_to_center
        ego_cy = 0.0
        ego_cz = 0.0
        ma.markers.append(
            self.make_cube_marker(mid, "ego", ego_cx, ego_cy, ego_cz, 0.0, self.ego_size, (1.0, 0.0, 0.0, 0.9))
        )
        mid += 1
        ma.markers.append(
            self.make_text_marker(mid, "ego_label", ego_cx, ego_cy, ego_cz, "Ego", 0.7)
        )
        mid += 1

        if self.obj_msg is None:
            return ma

        # npc
        for npc in self.obj_msg.npc_list:
            x, y, z = self.map_to_local(npc.position.x, npc.position.y, npc.position.z)
            if not self.in_crop(x, y):
                continue

            size = self.get_object_size(npc, self.default_npc_size)
            yaw = self.heading_to_local_yaw(npc.heading)
            offset = self.get_rear_to_center_offset(size)

            cx = x + offset * math.cos(yaw)
            cy = y + offset * math.sin(yaw)
            cz = z

            ma.markers.append(
                self.make_cube_marker(mid, "npc", cx, cy, cz, yaw, size, (0.0, 0.3, 1.0, 0.85))
            )
            mid += 1

        # pedestrian
        for ped in self.obj_msg.pedestrian_list:
            x, y, z = self.map_to_local(ped.position.x, ped.position.y, ped.position.z)
            if not self.in_crop(x, y):
                continue

            size = self.get_object_size(ped, self.default_ped_size)
            yaw = self.heading_to_local_yaw(ped.heading)

            ma.markers.append(
                self.make_cube_marker(mid, "ped", x, y, z, yaw, size, (0.0, 1.0, 0.0, 0.85))
            )
            mid += 1

        # obstacle
        for obs in self.obj_msg.obstacle_list:
            x, y, z = self.map_to_local(obs.position.x, obs.position.y, obs.position.z)
            if not self.in_crop(x, y):
                continue

            size = self.get_object_size(obs, self.default_obs_size)
            yaw = self.heading_to_local_yaw(getattr(obs, "heading", 0.0))

            ma.markers.append(
                self.make_cube_marker(mid, "obs", x, y, z, yaw, size, (0.6, 0.6, 0.6, 0.9))
            )
            mid += 1

        return ma

    def build_ego_marker(self):
        """
        simple arrow-like cube substitute not needed;
        ego already included in object markers
        """
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "ego_dummy"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 0.0
        m.lifetime = rospy.Duration(0.15)
        return m

    def publish_all(self):
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = self.frame_id

        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height

        # origin at bottom-left of local BEV box
        grid_msg.info.origin.position.x = -self.half_m
        grid_msg.info.origin.position.y = -self.half_m
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        grid_msg.data = self.build_grid()

        obj_markers = self.build_object_markers()
        ego_marker = self.build_ego_marker()

        self.grid_pub.publish(grid_msg)
        self.obj_pub.publish(obj_markers)
        self.ego_pub.publish(ego_marker)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.ego_msg is not None:
                self.publish_all()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = LocalBEVNav()
        node.run()
    except rospy.ROSInterruptException:
        pass