import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy

from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import Image

import message_filters
from message_filters import TimeSynchronizer, Subscriber

import cv2
from cv_bridge import CvBridge



class RLStreamer(Node):
    def __init__(self):
        super().__init__("rl_streamer")
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            liveliness=QoSLivelinessPolicy.RMW_QOS_POLICY_LIVELINESS_AUTOMATIC,
            depth=10,
        )
        
        self.control_pub = self.create_publisher(CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd_manual", 10)

        # self.bev_sub = Subscriber(self, Image, "/rg_mask", qos_profile=qos_profile)
        # self.event_sub = Subscriber(self, Image, "/floodfill_mask", qos_profile=qos_profile)
        self.bridge = CvBridge()
        self.bev_sub = self.create_subscription(Image, '/bev_publisher/bev_image', self.bev_callback, 1)
        self.bev_image = None
        self.bev_locked = False

    def bev_callback(self, msg):
        # Set Lock to prevent access during write
        self.bev_locked = True

        # Converting ROS image message to RGB
        self.bev_image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')

        # Remove lock as write is now finished
        self.bev_locked = False


    def pub_control(self, throttle = 0, steer = 0, brake = 0):
        control_msg = CarlaEgoVehicleControl()
        
        control_msg.brake = brake
        control_msg.steer = steer
        control_msg.throttle = throttle

        # Header
        control_msg.header.stamp = self.get_clock().now().to_msg()
        self.control_pub.publish(control_msg)