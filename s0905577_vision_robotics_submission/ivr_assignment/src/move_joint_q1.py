#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from numpy.linalg import pinv
from copy import deepcopy
import os
from datetime import datetime
import time


class MoveNode:

    def __init__(self) -> None:

        self.bridge = CvBridge()
        rospy.init_node("move_links", anonymous=True)


        self.robot_joint2_pub = rospy.Publisher(
            "/robot/joint2_position_controller/command",
            Float64,
            queue_size=10)

        self.robot_joint3_pub = rospy.Publisher(
            "/robot/joint3_position_controller/command",
            Float64,
            queue_size=10)

        self.robot_joint4_pub = rospy.Publisher(
            "/robot/joint4_position_controller/command",
            Float64,
            queue_size=10
        )

        self.q1_angles = rospy.Publisher(
            "/q1_baseline_angles", Float64MultiArray, queue_size=10
        )

        self.current_time = rospy.get_time()

        self.rate = rospy.Rate(50)

    def _sinusodial_rotations(self):
        deltas = np.array([
            (np.pi /15) * (rospy.get_time() - self.current_time),
            (np.pi /20) * (rospy.get_time() - self.current_time),
            (np.pi /18) * (rospy.get_time() - self.current_time)
        ])

        return (np.pi/2)* np.sin(deltas)


    def move_joints(self):
        angle_deltas = self._sinusodial_rotations()
        # print(f"Joint angles deltas are {angle_deltas}")
        # print(f"current time = {self.current_time}")

        self._assign_joint_data(angle_deltas)

        self.robot_joint2_pub.publish(self.joint_2)
        self.robot_joint3_pub.publish(self.joint_3)
        self.robot_joint4_pub.publish(self.joint_4)

        angle_delta_pos = Float64MultiArray()
        angle_delta_pos.data = angle_deltas
        self.q1_angles.publish(angle_delta_pos)
        self.rate.sleep()


    def _assign_joint_data(self, deltas):
        self.joint_2 = Float64()
        self.joint_2.data = deltas[0]

        self.joint_3 = Float64()
        self.joint_3.data = deltas[1]

        self.joint_4 = Float64()
        self.joint_4.data = deltas[2]

def main(args):


    move_node = MoveNode()

    while not rospy.is_shutdown():
        try:
            # print("Executing rospy move nodes at main")
            move_node.move_joints()
        except KeyboardInterrupt:
            print("Shutting Down")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # print("Executing rospy move node at entry")
    main(sys.argv)