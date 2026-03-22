#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import rospy

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class HDMapVisualizer:
    def __init__(self):
        rospy.init_node("hdmap_visualizer", anonymous=False)

        # fixed path
        self.map_path = "/home/autonav/jang_ws/src/yolo_detector/data/R_KR_PG_KATRI"
        self.frame_id = "map"
        self.publish_rate = 1.0

        self.pub = rospy.Publisher("/mgeo_markers", MarkerArray, queue_size=1, latch=True)
        
        #global : 맵 전체 기준 정보
        #node: 도로 구조 연결점 정보
        #lane_node: 차선점 정보
        #lane_marking: 차선 표시선 데이터
        #crosswalk: 횡단보도 그룹정보
        #singlecrosswalk: 실제 횡단보도 형상데이터
        #surface_marking: 노면표시 데이터
        #intersection_controller: 교차로 제어 정보(시각화 의미 없을지도?)

        self.paths = {
            "global": os.path.join(self.map_path, "global_info.json"),
            "node": os.path.join(self.map_path, "node_set.json"),
            "lane_node": os.path.join(self.map_path, "lane_node_set.json"),
            "lane_marking": os.path.join(self.map_path, "lane_marking_set.json"),
            "crosswalk": os.path.join(self.map_path, "crosswalk_set.json"),
            "singlecrosswalk": os.path.join(self.map_path, "singlecrosswalk_set.json"),
            "surface_marking": os.path.join(self.map_path, "surface_marking_set.json"),
            "intersection_controller": os.path.join(self.map_path, "intersection_controller_set.json")
        }

        self._check_files()

        self.global_info = self._load_json(self.paths["global"])
        self.node_set = self._load_json(self.paths["node"])
        self.lane_node_set = self._load_json(self.paths["lane_node"])
        self.lane_marking_set = self._load_json(self.paths["lane_marking"])
        self.crosswalk_set = self._load_json(self.paths["crosswalk"])
        self.singlecrosswalk_set = self._load_json(self.paths["singlecrosswalk"])
        self.surface_marking_set = self._load_json(self.paths["surface_marking"])
        self.intersection_controller_set = self._load_json(self.paths["intersection_controller"])

        self.single_crosswalk_map = {item["idx"]: item for item in self.singlecrosswalk_set}
        self.crosswalk_map = {item["idx"]: item for item in self.crosswalk_set}
        self.node_map = {item["idx"]: item for item in self.node_set}
        self.lane_node_map = {item["idx"]: item for item in self.lane_node_set}

        rospy.loginfo("HD map loaded from: %s", self.map_path)
        rospy.loginfo("nodes=%d, lane_nodes=%d, lane_markings=%d",len(self.node_set), len(self.lane_node_set), len(self.lane_marking_set))
        rospy.loginfo("crosswalk=%d, singlecrosswalk=%d, surface_marking=%d",len(self.crosswalk_set), len(self.singlecrosswalk_set),
                    len(self.surface_marking_set))
        rospy.loginfo("intersection_controller=%d",
                    len(self.intersection_controller_set))

        self.marker_array = self.build_all_markers()

    def _check_files(self):
        for _, path in self.paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError("Missing file: {}".format(path))

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _make_point(self, xyz):
        p = Point()
        p.x = float(xyz[0])
        p.y = float(xyz[1])
        p.z = float(xyz[2]) if len(xyz) > 2 else 0.0
        return p

    def _make_color(self, r, g, b, a=1.0):
        c = ColorRGBA()
        c.r = r
        c.g = g
        c.b = b
        c.a = a
        return c

    def _new_marker(self, mid, ns, mtype):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = rospy.Duration(0)
        return m

    def _bbox_center(self, pts):
        if not pts:
            return [0.0, 0.0, 0.0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] if len(p) > 2 else 0.0 for p in pts]
        return [(min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5]

    def _dist2d(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    def _make_text_marker(self, mid, ns, xyz, text, scale=0.8, color=None):
        m = self._new_marker(mid, ns, Marker.TEXT_VIEW_FACING)
        m.pose.position.x = xyz[0]
        m.pose.position.y = xyz[1]
        m.pose.position.z = xyz[2] + 1.0
        m.scale.z = scale
        m.color = color if color else self._make_color(1.0, 1.0, 1.0, 1.0)
        m.text = text
        return m

    def _make_sphere_marker(self, mid, ns, xyz, scale_xyz=0.5, color=None):
        m = self._new_marker(mid, ns, Marker.SPHERE)
        m.pose.position.x = xyz[0]
        m.pose.position.y = xyz[1]
        m.pose.position.z = xyz[2]
        m.scale.x = scale_xyz
        m.scale.y = scale_xyz
        m.scale.z = scale_xyz
        m.color = color if color else self._make_color(1.0, 1.0, 1.0, 1.0)
        return m

    def _make_cube_marker(self, mid, ns, xyz, sx, sy, sz, color=None):
        m = self._new_marker(mid, ns, Marker.CUBE)
        m.pose.position.x = xyz[0]
        m.pose.position.y = xyz[1]
        m.pose.position.z = xyz[2]
        m.scale.x = sx
        m.scale.y = sy
        m.scale.z = sz
        m.color = color if color else self._make_color(1.0, 1.0, 1.0, 1.0)
        return m

    def _make_line_strip(self, mid, ns, pts, width, color):
        m = self._new_marker(mid, ns, Marker.LINE_STRIP)
        m.scale.x = width
        m.color = color
        for p in pts:
            m.points.append(self._make_point(p))
        return m

    def _make_line_list(self, mid, ns, seg_pts, width, color):
        m = self._new_marker(mid, ns, Marker.LINE_LIST)
        m.scale.x = width
        m.color = color
        for p in seg_pts:
            m.points.append(self._make_point(p))
        return m

    def _make_points_marker(self, mid, ns, pts, size, color):
        m = self._new_marker(mid, ns, Marker.POINTS)
        m.scale.x = size
        m.scale.y = size
        m.color = color
        for p in pts:
            m.points.append(self._make_point(p))
        return m

    def _make_triangle_list(self, mid, ns, pts, color):
        m = self._new_marker(mid, ns, Marker.TRIANGLE_LIST)
        m.color = color

        if len(pts) < 3:
            return m

        poly = pts[:]
        if poly[0] == poly[-1]:
            poly = poly[:-1]

        if len(poly) < 3:
            return m

        p0 = poly[0]
        for i in range(1, len(poly) - 1):
            m.points.append(self._make_point(p0))
            m.points.append(self._make_point(poly[i]))
            m.points.append(self._make_point(poly[i + 1]))
        return m

    def _lane_color_to_rgba(self, lane_color):
        lane_color = str(lane_color).lower()

        if lane_color == "white":
            return self._make_color(1.0, 1.0, 1.0, 1.0)
        elif lane_color == "yellow":
            return self._make_color(1.0, 0.95, 0.0, 1.0)
        elif lane_color == "blue":
            return self._make_color(0.1, 0.4, 1.0, 1.0)
        elif lane_color == "undefined":
            return self._make_color(0.0, 1.0, 0.0, 1.0)
        else:
            return self._make_color(0.8, 0.8, 0.8, 1.0)

    def _lane_type_name(self, lane_type):
        mapping = {
            501: "Center Line",
            503: "Lane Boundary",
            504: "Special / Bus Lane",
            505: "Lane Divider",
            506: "Lane Change Restriction",
            515: "Guide Line",
            525: "Stop Line",
            530: "Other Lane",
            531: "Other Lane",
            535: "Road Edge Line",
            599: "Unclassified Lane",
        }
        return mapping.get(int(lane_type), "Lane {}".format(lane_type))

    def _surface_color(self, type_code, sub_type):
        key = "{}_{}".format(type_code, sub_type)
        table = {
            "5_5321": self._make_color(1.0, 1.0, 0.2, 0.55),
            "5_534":  self._make_color(1.0, 0.5, 0.0, 0.55),
            "1_5371": self._make_color(0.2, 0.9, 0.9, 0.45),
            "1_5372": self._make_color(0.2, 0.8, 1.0, 0.45),
            "1_5373": self._make_color(0.0, 0.7, 1.0, 0.45),
            "1_5374": self._make_color(0.3, 1.0, 0.6, 0.45),
            "1_5379": self._make_color(1.0, 0.4, 0.7, 0.45),
            "1_5381": self._make_color(0.8, 0.5, 1.0, 0.45),
            "1_5382": self._make_color(0.6, 0.4, 1.0, 0.45),
            "1_5431": self._make_color(1.0, 0.2, 0.2, 0.45),
            "1_5432": self._make_color(1.0, 0.1, 0.1, 0.45),
        }
        return table.get(key, self._make_color(0.8, 0.8, 0.8, 0.4))

    def _surface_marking_label(self, item):
        t = str(item.get("type", ""))
        st = str(item.get("sub_type", ""))

        mapping = {
            ("5", "5321"): "Crosswalk-related Surface Marking",
            ("5", "534"): "Warning / Deceleration Surface Marking",
            ("1", "5371"): "Directional Guidance Marking",
            ("1", "5372"): "Directional Guidance Marking",
            ("1", "5373"): "Directional Guidance Marking",
            ("1", "5374"): "Directional Guidance Marking",
            ("1", "5379"): "Directional Guidance Marking",
            ("1", "5381"): "Lane Guidance Marking",
            ("1", "5382"): "Lane Guidance Marking",
            ("1", "5431"): "Stop / Warning Surface Marking",
            ("1", "5432"): "Stop / Warning Surface Marking",
        }
        return mapping.get((t, st), "Surface Marking {}-{}".format(t, st))

    def _traffic_light_color(self, tl_type):
        if tl_type == "pedestrian":
            return self._make_color(0.2, 1.0, 0.2, 1.0)
        elif tl_type == "bus":
            return self._make_color(1.0, 0.5, 0.0, 1.0)
        else:
            return self._make_color(1.0, 0.1, 0.1, 1.0)

    def _traffic_light_label(self, item):
        tl_type = item.get("type", "")
        sub_type = item.get("sub_type", [])

        if not isinstance(sub_type, list):
            sub_type = [str(sub_type)]

        if tl_type == "car":
            type_name = "Vehicle Traffic Light"
        elif tl_type == "pedestrian":
            type_name = "Pedestrian Traffic Light"
        elif tl_type == "bus":
            type_name = "Bus Traffic Light"
        else:
            type_name = "Traffic Light"

        sub_map = {
            "red": "Red",
            "yellow": "Yellow",
            "left": "Left Turn",
            "right": "Right Turn",
            "straight": "Straight",
            "uturn": "U-turn",
            "lowerleft": "Lower Left",
            "lowerright": "Lower Right",
        }

        names = [sub_map.get(str(s), str(s)) for s in sub_type]
        return "{}\n{}".format(type_name, ", ".join(names)) if names else type_name

    def _traffic_sign_color(self, sign_type):
        if sign_type == "1":
            return self._make_color(1.0, 0.2, 0.2, 1.0)
        elif sign_type == "2":
            return self._make_color(0.2, 0.4, 1.0, 1.0)
        elif sign_type == "3":
            return self._make_color(1.0, 0.9, 0.1, 1.0)
        elif sign_type == "4":
            return self._make_color(0.2, 1.0, 0.2, 1.0)
        else:
            return self._make_color(1.0, 1.0, 1.0, 1.0)

    def _traffic_sign_label(self, item):
        sign_type = str(item.get("type", ""))
        sub_type = str(item.get("sub_type", ""))

        type_name_map = {
            "1": "Warning Sign",
            "2": "Regulatory Sign",
            "3": "Instruction Sign",
            "4": "Supplementary Sign",
        }

        sub_name_map = {
            "199": "Warning Sign",
            "205": "Road Narrows",
            "211": "Children Protection",
            "212": "Bicycle Warning",
            "213": "Uneven Road",
            "214": "Slippery Road",
            "218": "Crosswalk",
            "221": "Merging Road",
            "222": "Traffic Signal Ahead",
            "224": "Speed Limit",
            "225": "Minimum Speed Limit",
            "226": "No Stopping or Parking",
            "227": "No Stopping",
            "228": "No Parking",
            "231": "No Entry",
            "299": "Regulatory Sign",
            "304": "Straight",
            "306": "Left Turn",
            "307": "Right Turn",
            "308": "Left or Right Turn",
            "309": "Straight or Left Turn",
            "310": "Straight or Right Turn",
            "312": "U-turn",
            "315": "Minimum Speed",
            "322": "One Way",
            "324": "Priority Road",
            "325": "U-turn Allowed",
            "332": "Bicycle Road",
            "399": "Instruction Sign",
            "499": "Supplementary Sign",
        }

        type_name = type_name_map.get(sign_type, "Traffic Sign")
        sub_name = sub_name_map.get(sub_type, "Code {}".format(sub_type))
        return "{}\n{}".format(type_name, sub_name)

    def _build_dashed_segments(self, pts, dash_len=1.0, gap_len=1.0):
        if len(pts) < 2:
            return []

        seg_points = []
        draw = True
        remain = dash_len
        cur = [pts[0][0], pts[0][1], pts[0][2]]

        i = 0
        while i < len(pts) - 1:
            p1 = cur
            p2 = pts[i + 1]
            seg_len = self._dist2d(p1, p2)

            if seg_len < 1e-6:
                i += 1
                if i < len(pts):
                    cur = list(pts[i])
                continue

            if seg_len <= remain:
                if draw:
                    seg_points.append([p1[0], p1[1], p1[2]])
                    seg_points.append([p2[0], p2[1], p2[2]])
                remain -= seg_len
                cur = list(p2)
                i += 1

                if remain <= 1e-6:
                    draw = not draw
                    remain = dash_len if draw else gap_len
            else:
                ratio = remain / seg_len
                nx = p1[0] + (p2[0] - p1[0]) * ratio
                ny = p1[1] + (p2[1] - p1[1]) * ratio
                nz = p1[2] + (p2[2] - p1[2]) * ratio
                split = [nx, ny, nz]
                if draw:
                    seg_points.append([p1[0], p1[1], p1[2]])
                    seg_points.append(split)
                cur = split
                draw = not draw
                remain = dash_len if draw else gap_len

        return seg_points

    def build_all_markers(self):
        ma = MarkerArray()
        mid = 0

        # 1) lane markings
        for item in self.lane_marking_set:
            pts = item.get("points", [])
            if len(pts) < 2:
                continue

            lane_color = item.get("lane_color", "undefined")
            lane_shape = item.get("lane_shape", [])
            lane_width = float(item.get("lane_width", 0.15))
            lane_type = item.get("lane_type", "")
            color = self._lane_color_to_rgba(lane_color)

            width = max(lane_width * 0.15, 0.10)

            if "Broken" in lane_shape:
                dash_l1 = float(item.get("dash_interval_L1", 1.0) or 1.0)
                dash_l2 = float(item.get("dash_interval_L2", 1.0) or 1.0)
                seg_pts = self._build_dashed_segments(pts, dash_len=dash_l1, gap_len=dash_l2)
                ma.markers.append(self._make_line_list(mid, "lane_markings", seg_pts, width, color))
                mid += 1
            else:
                ma.markers.append(self._make_line_strip(mid, "lane_markings", pts, width, color))
                mid += 1

            if lane_shape in [["Solid", "Solid"], ["Broken", "Solid"], ["Solid", "Broken"]]:
                c = self._bbox_center(pts)
                shape_txt = "/".join(lane_shape)
                label = "{}\n{}\n{}".format(
                    self._lane_type_name(lane_type),
                    lane_color,
                    shape_txt
                )
                ma.markers.append(
                    self._make_text_marker(
                        mid,
                        "lane_marking_labels",
                        c,
                        label,
                        scale=0.45,
                        color=self._make_color(1.0, 1.0, 0.0, 0.9),
                    )
                )
                mid += 1

        # 2) nodes
        node_points = [item["point"] for item in self.node_set if "point" in item]
        if node_points:
            ma.markers.append(
                self._make_points_marker(
                    mid, "nodes", node_points, 0.18, self._make_color(0.8, 0.8, 0.8, 0.6)
                )
            )
            mid += 1

        # 3) lane nodes
        lane_node_points = [item["point"] for item in self.lane_node_set if "point" in item]
        if lane_node_points:
            ma.markers.append(
                self._make_points_marker(
                    mid, "lane_nodes", lane_node_points, 0.08, self._make_color(0.6, 0.6, 1.0, 0.35)
                )
            )
            mid += 1

        # 4) single crosswalk
        for item in self.singlecrosswalk_set:
            pts = item.get("points", [])
            if len(pts) < 3:
                continue

            fill_color = self._make_color(1.0, 1.0, 1.0, 0.35)
            edge_color = self._make_color(1.0, 1.0, 1.0, 0.95)

            ma.markers.append(self._make_triangle_list(mid, "single_crosswalk_fill", pts, fill_color))
            mid += 1
            ma.markers.append(self._make_line_strip(mid, "single_crosswalk_edge", pts, 0.12, edge_color))
            mid += 1

        # 5) crosswalk group label
        for item in self.crosswalk_set:
            idx = item.get("idx", "")
            refs = item.get("single_crosswalk_list", [])
            all_pts = []

            for ref in refs:
                scw = self.single_crosswalk_map.get(ref)
                if scw:
                    all_pts.extend(scw.get("points", []))

            if not all_pts:
                continue

            c = self._bbox_center(all_pts)
            text = "Crosswalk\n{}\nParts: {}".format(idx, len(refs))
            ma.markers.append(
                self._make_text_marker(
                    mid,
                    "crosswalk_labels",
                    c,
                    text,
                    scale=0.75,
                    color=self._make_color(0.0, 1.0, 1.0, 0.95),
                )
            )
            mid += 1

        # 6) surface marking
        for item in self.surface_marking_set:
            pts = item.get("points", [])
            if len(pts) < 3:
                continue

            t = str(item.get("type", ""))
            st = str(item.get("sub_type", ""))
            fill_color = self._surface_color(t, st)

            ma.markers.append(self._make_triangle_list(mid, "surface_marking_fill", pts, fill_color))
            mid += 1
            ma.markers.append(
                self._make_line_strip(mid, "surface_marking_edge", pts, 0.08, self._make_color(1.0, 1.0, 1.0, 0.7))
            )
            mid += 1

            c = self._bbox_center(pts)
            ma.markers.append(
                self._make_text_marker(
                    mid,
                    "surface_marking_labels",
                    c,
                    self._surface_marking_label(item),
                    scale=0.42,
                    color=self._make_color(1.0, 0.85, 0.0, 0.95),
                )
            )
            mid += 1
        '''
        # 7) traffic lights
        for item in self.traffic_light_set:
            pt = item.get("point", None)
            if pt is None:
                continue

            tl_type = item.get("type", "")
            color = self._traffic_light_color(tl_type)

            width = max(float(item.get("width", 0.4) or 0.4), 0.25)
            height = max(float(item.get("height", 0.4) or 0.4), 0.25)
            z_offset = float(item.get("z_offset", 0.0) or 0.0)
            draw_pt = [pt[0], pt[1], pt[2] + z_offset]

            ma.markers.append(self._make_cube_marker(mid, "traffic_lights", draw_pt, width, width, height, color))
            mid += 1

            label = self._traffic_light_label(item)
            ma.markers.append(
                self._make_text_marker(
                    mid,
                    "traffic_light_labels",
                    draw_pt,
                    label,
                    scale=0.55,
                    color=self._make_color(1.0, 0.95, 0.95, 1.0),
                )
            )
            mid += 1
        
        # 8) traffic signs
        for item in self.traffic_sign_set:
            pt = item.get("point", None)
            if pt is None:
                continue

            sign_type = str(item.get("type", ""))
            color = self._traffic_sign_color(sign_type)

            z_offset = float(item.get("z_offset", 0.0) or 0.0)
            draw_pt = [pt[0], pt[1], pt[2] + z_offset]

            ma.markers.append(self._make_sphere_marker(mid, "traffic_signs", draw_pt, 0.45, color))
            mid += 1

            label = self._traffic_sign_label(item)
            ma.markers.append(
                self._make_text_marker(
                    mid,
                    "traffic_sign_labels",
                    draw_pt,
                    label,
                    scale=0.48,
                    color=self._make_color(1.0, 1.0, 0.6, 1.0),
                )
            )
            mid += 1
        '''

        # 9) empty file info
        if len(self.intersection_controller_set) == 0:
            ma.markers.append(
                self._make_text_marker(
                    mid,
                    "info",
                    [0.0, 0.0, 3.0],
                    "intersection_controller_set.json is empty",
                    scale=0.8,
                    color=self._make_color(1.0, 0.5, 0.5, 0.9),
                )
            )
            mid += 1

        '''
        if len(self.synced_traffic_light_set) == 0:
            ma.markers.append(
                self._make_text_marker(
                    mid,
                    "info",
                    [0.0, 0.0, 4.2],
                    "synced_traffic_light_set.json is empty",
                    scale=0.8,
                    color=self._make_color(1.0, 0.5, 0.5, 0.9),
                )
            )
            mid += 1
        '''
        rospy.loginfo("total markers built: %d", len(ma.markers))
        return ma

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            for m in self.marker_array.markers:
                m.header.stamp = now
            self.pub.publish(self.marker_array)
            rate.sleep()


if __name__ == "__main__":
    try:
        node = HDMapVisualizer()
        node.run()
    except Exception as e:
        rospy.logerr("hdmap_visualizer error: %s", str(e))