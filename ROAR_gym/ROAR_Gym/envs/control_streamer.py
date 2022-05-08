from ROAR.utilities_module import occupancy_map
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy

from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header

import message_filters
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer

import cv2
from cv_bridge import CvBridge

import numpy as np


class RLStreamer(Node):
    def __init__(self):
        super().__init__("rl_streamer")
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            liveliness=QoSLivelinessPolicy.RMW_QOS_POLICY_LIVELINESS_AUTOMATIC,
            depth=1,
        )
        
        self.ogm_shape = (84,84)
        self.ogm_channels = 3
        self.ogm_stack_size = 4


        self.control_pub = self.create_publisher(CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd_manual", 10)

        self.bridge = CvBridge()

        self.bev_image = np.zeros((self.ogm_shape[0], self.ogm_shape[1], 3))
        self.bev_locked = False

        self.ogm = np.zeros((self.ogm_stack_size, self.ogm_channels, self.ogm_shape[0], self.ogm_shape[1]))

        self.event = 0

        # self.bev_sub = self.create_subscription(Image, '/bev_publisher/bev_image', self.bev_callback, 1)
        # self.event_sub = self.create_subscription(Float32, '/state_streamer/event', self.reward_callback, 1)

        self.bev_sub   = Subscriber(self, Image, '/bev_publisher/bev_image', qos_profile=qos_profile)
        self.event_sub = Subscriber(self, Header, '/state_streamer/event_header', qos_profile=qos_profile)

        sync_msg = ApproximateTimeSynchronizer(
            [self.bev_sub, self.event_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=True
        )
        sync_msg.registerCallback(self.sync_callback)

    def sync_callback(self, bev_msg, event_msg):

        # ------------------------------- Handle Event ------------------------------- #
        self.event = float(event_msg.frame_id)
        print(self.event)
        self.event = 0

        # -------------------------------- Handle BEV -------------------------------- #
        # Set Lock to prevent access during write
        self.bev_locked = True

        # Converting ROS image message to BGR
        self.bev_image = self.bridge.imgmsg_to_cv2(bev_msg, desired_encoding='bgr8')

        print("\n\n\n-------------------\n the max of bev is:", np.max(self.bev_image), "\n\n\n-------------------")
        
        cv2.imshow("BEV from RL Streamer", 255*self.bev_image)
        cv2.waitKey(1)
        self.bev_image = np.array(self.bev_image, dtype='<f8')
        self.bev_image = np.stack(
            [
                self.bev_image[:,:,0],
                self.bev_image[:,:,1],
                self.bev_image[:,:,2]
            ],
            axis = 0,
        )
        assert self.bev_image.shape == (3,84,84), "Received BEV Image is not of the correct shape"
        self.update_occupancy_map()

        # Remove lock as write is now finished
        self.bev_locked = False



    # def bev_callback(self, msg):
    #     # Set Lock to prevent access during write
    #     self.bev_locked = True

    #     # Converting ROS image message to RGB
    #     self.bev_image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')

    #     # Remove lock as write is now finished
    #     self.bev_locked = False
        
    # def reward_callback(self, msg):

    #     if (msg == 0):
    #         self.crash = msg.data
    #     elif (msg.data == 1):
    #         self.reward = msg.data


    def pub_control(self, throttle = 0, steer = 0, brake = 0):
        control_msg = CarlaEgoVehicleControl()
        
        control_msg.brake = brake
        control_msg.steer = steer
        control_msg.throttle = throttle

        # Header
        control_msg.header.stamp = self.get_clock().now().to_msg()
        self.control_pub.publish(control_msg)

    def get_num_collision(self):
        if self.event == 2:
            return 1
        else:
            return 0

    def update_occupancy_map(self):

        # Push old bevs up the ogm stack
        self.ogm[1:self.ogm_stack_size-1, :, :, :] = self.ogm[0:self.ogm_stack_size-2, :, :, :]
        self.ogm[0, :, :, :] = self.bev_image
        self.ogm = np.array(self.ogm, dtype="<f8")
        
        assert self.ogm.shape == (self.ogm_stack_size, self.ogm_channels, self.ogm_shape[0], self.ogm_shape[1]), "Stacked OGM Shape incorrect"
        # return self.ogm

    def get_occupancy_map(self):
        return self.ogm
