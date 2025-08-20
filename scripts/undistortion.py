#!/usr/bin/env python3
import os
import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
import message_filters
import ros2_numpy
import cv2
import time
import rosbag2_py
from rclpy.serialization import deserialize_message
import open3d as o3d
import numpy as np
import yaml


# QoS Profiles
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class BagReader(Node):
    def __init__(self):
        super().__init__("bag_reader")

        self.fisheye_pub = self.create_publisher(
            Image, '/dual_fisheye/image', qos_profile
        )
        self.reader = rosbag2_py.SequentialReader()
        bag_path = os.path.expanduser('/home/container_user/wheelchair2/src/records/calib/back_og/rosbag2_2025_08_20-06_53_01_0.db3')
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.reader.open(storage_options, converter_options)

        self.topic = {
            '/dual_fisheye/image': (Image, self.fisheye_pub)
        }

        self.create_timer(0.1, self.ros_play)

        self.played = False

    def ros_play(self):
        if self.played:
            return
        self.played = True
        self.get_logger().info('Started playing bag')
        wall_start_time = time.time()
        bag_start_time = None

        while self.reader.has_next():
            try:
                (topic, data, timestamp) = self.reader.read_next()
                if bag_start_time is None:
                    bag_start_time = timestamp
                intended_time = wall_start_time + (timestamp - bag_start_time) / 1e9
                now = time.time()
                sleep_time = intended_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if topic in self.topic:
                    message_type, publisher = self.topic[topic]
                    msg = deserialize_message(data, message_type)
                    publisher.publish(msg)

            except Exception as e:
                self.get_logger().error(f'Error reading bag: {e}')
                break
# ----- I have this so that i can run irl data and change it also


class DualFisheyeSplit(Node):
    def __init__(self):
        super().__init__("dual_fisheye_split")

        self.bridge = CvBridge()
        self.save_fisheye_front = '/home/container_user/wheelchair2/src/records/fisheye/fisheye_front'
        self.save_fisheye_back = '/home/container_user/wheelchair2/src/records/fisheye/fisheye_back'
        self.save_front = '/home/container_user/wheelchair2/src/records/fisheye/undistorted_front'
        self.save_back = '/home/container_user/wheelchair2/src/records/fisheye/undistorted_back'
        
        os.makedirs(self.save_front, exist_ok=True)
        os.makedirs(self.save_back, exist_ok=True)
        os.makedirs(self.save_fisheye_front, exist_ok=True)
        os.makedirs(self.save_fisheye_back, exist_ok=True)

        self.frame_count = 0

        self.fisheye_sub = self.create_subscription(
            Image,
            "/dual_fisheye/image",
            self.split_callback,
            10 
        )

        self.front_fisheye_pub = self.create_publisher(
            Image,
            "/front_fisheye/image",
            10
        )
        self.back_fisheye_pub = self.create_publisher(
            Image,
            "/back_fisheye/image",
            10
        )
            
    def split_callback(self, dual_fisheye_msg):
        dual_fisheye_img = self.bridge.imgmsg_to_cv2(dual_fisheye_msg, "bgr8")
            
        img_height, img_width_full, _ = dual_fisheye_img.shape
        midpoint = img_width_full // 2
        front_img_full = dual_fisheye_img[:, :midpoint]
        back_img_full = dual_fisheye_img[:, midpoint:]

        # front_img_full = cv2.rotate(front_img_full, cv2.ROTATE_90_CLOCKWISE)
        # back_img_full = cv2.rotate(back_img_full, cv2.ROTATE_90_COUNTERCLOCKWISE)

        filename_front_fisheye = os.path.join(self.save_fisheye_front, f"frame_{self.frame_count:06d}.png")
        filename_back_fisheye = os.path.join(self.save_fisheye_back, f"frame_{self.frame_count:06d}.png")

        cv2.imwrite(filename_front_fisheye, front_img_full)
        cv2.imwrite(filename_back_fisheye, back_img_full)

        K = np.array([
            [439.84214534, 0.0, 685.81748965], 
            [0.0, 424.55707757, 648.8049538], 
            [0.0, 0.0, 1.0]
            ])

        D = np.array([
            [-0.40391458],
            [0.17691515],  
            [0.01728955], 
            [-0.02034806]
            ])

        fh, fw, _ = front_img_full.shape
        bh, bw, _ = back_img_full.shape
        new_K = K.copy()
        mapf1, mapf2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (fw, fh), cv2.CV_32FC1)
        mapb1, mapb2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (bw, bh), cv2.CV_32FC1)
        undistorted_front_img = cv2.remap(front_img_full, mapf1, mapf2, interpolation=cv2.INTER_LINEAR)
        undistorted_back_img =cv2.remap(back_img_full, mapb1, mapb2, interpolation=cv2.INTER_LINEAR) 
        
        filename_front_undistorted = os.path.join(self.save_front, f"frame_{self.frame_count:06d}.png")
        filename_back_undistorted = os.path.join(self.save_back, f"frame_{self.frame_count:06d}.png")
        cv2.imwrite(filename_front_undistorted, undistorted_front_img)
        cv2.imwrite(filename_back_undistorted, undistorted_back_img)
        
        self.frame_count += 1                
        

def main(args=None):
    RUN_WITH_BAG = True

    rclpy.init(args=args)
    split = DualFisheyeSplit()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(split)
    bag_reader = None
    if RUN_WITH_BAG:
        print("RUNNING IN BAG PLAYBACK MODE")
        bag_reader = BagReader()
        executor.add_node(bag_reader)
    else:
        print("RUNNING IN LIVE MODE (DETECTOR ONLY)")

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if bag_reader is not None:
            bag_reader.destroy_node()
        split.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
