#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import tf

from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


class MoraiObjectsRviz:
    def __init__(self):
        rospy.init_node("morai_objects_rviz", anonymous=False)

        self.frame_id = "map"
        self.base_frame = "base_link"

        # Ego vehicle: 2023 Hyundai IONIQ 5
        # (length, width, height) in meters
        self.ego_size = (4.635, 1.890, 1.605)

        # MORAI pose is interpreted as rear-axle/base_link.
        # RViz CUBE pose is center-based, so we shift the cube center forward.
        self.ego_rear_to_center = 1.60

        # Fallback sizes when MORAI object size is missing or invalid
        self.default_npc_size = (4.4, 1.8, 1.5)
        self.default_ped_size = (0.6, 0.6, 1.7)
        self.default_obs_size = (1.0, 1.0, 1.0)

        # Heading correction if needed
        self.yaw_offset_deg = 0.0
        self.invert_heading = False

        self.ego_msg = None
        self.obj_msg = None

        self.marker_pub = rospy.Publisher("/morai_vehicle_markers", MarkerArray, queue_size=10)
        self.ego_axes_pub = rospy.Publisher("/ego_axes_marker", Marker, queue_size=10)
        self.odom_pub = rospy.Publisher("/ego_odom", Odometry, queue_size=10)

        self.tf_br = tf.TransformBroadcaster()

        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_callback)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.obj_callback)

        rospy.loginfo("morai_objects_rviz started")
        rospy.loginfo("ego vehicle set to 2023 Hyundai IONIQ 5")
        rospy.loginfo("ego size = L: %.3f, W: %.3f, H: %.3f",
                      self.ego_size[0], self.ego_size[1], self.ego_size[2])

    def ego_callback(self, msg):
        self.ego_msg = msg

    def obj_callback(self, msg):
        self.obj_msg = msg

    def heading_to_yaw_rad(self, heading_deg):
        h = heading_deg
        if self.invert_heading:
            h = -h
        return math.radians(h + self.yaw_offset_deg)

    def get_object_size(self, obj, default_size):
        """
        MORAI ObjectStatus.size ordering can vary by environment.
        User already verified that width/length were swapped once.
        This version assumes:
            size.x = length
            size.y = width
            size.z = height
        RViz CUBE uses:
            scale.x = length, scale.y = width, scale.z = height
        """
        try:
            length = float(obj.size.x)
            width = float(obj.size.y)
            height = float(obj.size.z)

            if width <= 0.01 or length <= 0.01 or height <= 0.01:
                return default_size

            return (length, width, height)
        except Exception:
            return default_size

    def get_rear_to_center_offset(self, size, obj_name=""):
        """
        Convert rear-axle/base_link pose to geometric vehicle center.
        size = (length, width, height)
        """
        length = size[0]
        name = str(obj_name).lower()

        # Heuristic by size
        if length >= 10.0:
            return 3.0       # bus
        elif length >= 6.0:
            return 2.2       # truck / large van
        elif length >= 5.0:
            return 1.7       # large SUV / large sedan
        elif length >= 4.3:
            return 1.5       # normal sedan
        else:
            return 1.2       # compact car

    def make_cube(self, mid, ns, x, y, z, yaw_rad, size, color):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id

        marker.ns = ns
        marker.id = mid
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw_rad)

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z + size[2] * 0.5

        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.scale.x = size[0]   # length
        marker.scale.y = size[1]   # width
        marker.scale.z = size[2]   # height

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        marker.lifetime = rospy.Duration(0.2)
        return marker

    def make_text(self, mid, ns, x, y, z, text, scale=0.8):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id

        marker.ns = ns
        marker.id = mid
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z + 2.2
        marker.pose.orientation.w = 1.0

        marker.scale.z = scale

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = text
        marker.lifetime = rospy.Duration(0.2)
        return marker

    def make_arrow(self, mid, ns, x, y, z, yaw_rad, length=2.8, shaft_d=0.18, head_d=0.35):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id

        marker.ns = ns
        marker.id = mid
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw_rad)

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z + 0.2

        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.scale.x = length
        marker.scale.y = shaft_d
        marker.scale.z = head_d

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.95

        marker.lifetime = rospy.Duration(0.2)
        return marker

    def publish_ego_tf_and_odom(self, ego):
        now = rospy.Time.now()

        x = ego.position.x
        y = ego.position.y
        z = ego.position.z
        yaw_rad = self.heading_to_yaw_rad(ego.heading)
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw_rad)

        # TF: map -> base_link (rear axle / MORAI pose)
        self.tf_br.sendTransform(
            (x, y, z),
            q,
            now,
            self.base_frame,
            self.frame_id
        )

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = ego.velocity.x
        odom.twist.twist.linear.y = ego.velocity.y
        odom.twist.twist.linear.z = ego.velocity.z

        self.odom_pub.publish(odom)

        # Additional visible arrow at base_link
        ego_arrow = self.make_arrow(
            0, "ego_axis",
            x, y, z,
            yaw_rad,
            length=2.8
        )
        self.ego_axes_pub.publish(ego_arrow)

    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            marker_array = MarkerArray()
            mid = 0

            # Ego vehicle
            if self.ego_msg is not None:
                ego = self.ego_msg
                yaw_rad = self.heading_to_yaw_rad(ego.heading)

                # base_link / TF / odometry use rear axle reference
                self.publish_ego_tf_and_odom(ego)

                # vehicle body cube center shifted forward from rear axle
                ego_cx = ego.position.x + self.ego_rear_to_center * math.cos(yaw_rad)
                ego_cy = ego.position.y + self.ego_rear_to_center * math.sin(yaw_rad)
                ego_cz = ego.position.z

                marker_array.markers.append(
                    self.make_cube(
                        mid, "ego",
                        ego_cx, ego_cy, ego_cz,
                        yaw_rad,
                        self.ego_size,
                        (1.0, 0.0, 0.0, 0.80)
                    )
                )
                mid += 1

                marker_array.markers.append(
                    self.make_text(
                        mid, "ego_label",
                        ego_cx, ego_cy, ego_cz,
                        "Ego (IONIQ 5)"
                    )
                )
                mid += 1

            # NPC / pedestrian / obstacle
            if self.obj_msg is not None:
                obj = self.obj_msg

                # NPC vehicles
                for npc in obj.npc_list:
                    yaw_rad = self.heading_to_yaw_rad(npc.heading)
                    npc_size = self.get_object_size(npc, self.default_npc_size)
                    npc_offset = self.get_rear_to_center_offset(npc_size, getattr(npc, "name", ""))

                    npc_cx = npc.position.x + npc_offset * math.cos(yaw_rad)
                    npc_cy = npc.position.y + npc_offset * math.sin(yaw_rad)
                    npc_cz = npc.position.z

                    marker_array.markers.append(
                        self.make_cube(
                            mid, "npc",
                            npc_cx, npc_cy, npc_cz,
                            yaw_rad,
                            npc_size,
                            (0.0, 0.3, 1.0, 0.80)
                        )
                    )
                    mid += 1

                    label = getattr(npc, "name", "NPC")
                    label_text = "{}\nL:{:.1f} W:{:.1f} H:{:.1f}".format(
                        label, npc_size[0], npc_size[1], npc_size[2]
                    )
                    marker_array.markers.append(
                        self.make_text(
                            mid, "npc_label",
                            npc_cx, npc_cy, npc_cz,
                            label_text,
                            scale=0.55
                        )
                    )
                    mid += 1

                # Pedestrians
                for ped in obj.pedestrian_list:
                    yaw_rad = self.heading_to_yaw_rad(ped.heading)
                    ped_size = self.get_object_size(ped, self.default_ped_size)

                    marker_array.markers.append(
                        self.make_cube(
                            mid, "pedestrian",
                            ped.position.x, ped.position.y, ped.position.z,
                            yaw_rad,
                            ped_size,
                            (0.0, 1.0, 0.0, 0.85)
                        )
                    )
                    mid += 1

                    label_text = "Pedestrian\nL:{:.1f} W:{:.1f} H:{:.1f}".format(
                        ped_size[0], ped_size[1], ped_size[2]
                    )
                    marker_array.markers.append(
                        self.make_text(
                            mid, "ped_label",
                            ped.position.x, ped.position.y, ped.position.z,
                            label_text,
                            scale=0.55
                        )
                    )
                    mid += 1

                # Obstacles
                for obs in obj.obstacle_list:
                    yaw_deg = getattr(obs, "heading", 0.0)
                    yaw_rad = self.heading_to_yaw_rad(yaw_deg)
                    obs_size = self.get_object_size(obs, self.default_obs_size)

                    marker_array.markers.append(
                        self.make_cube(
                            mid, "obstacle",
                            obs.position.x, obs.position.y, obs.position.z,
                            yaw_rad,
                            obs_size,
                            (0.6, 0.6, 0.6, 0.90)
                        )
                    )
                    mid += 1

                    label_text = "Obstacle\nL:{:.1f} W:{:.1f} H:{:.1f}".format(
                        obs_size[0], obs_size[1], obs_size[2]
                    )
                    marker_array.markers.append(
                        self.make_text(
                            mid, "obs_label",
                            obs.position.x, obs.position.y, obs.position.z,
                            label_text,
                            scale=0.55
                        )
                    )
                    mid += 1

            self.marker_pub.publish(marker_array)
            rate.sleep()


if __name__ == "__main__":
    try:
        node = MoraiObjectsRviz()
        node.run()
    except rospy.ROSInterruptException:
        pass