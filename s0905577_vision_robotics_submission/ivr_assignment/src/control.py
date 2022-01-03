#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge
LOWER_LINK_LENGTH = 4
MIDDLE_LINK_LENGTH = 3.2
UPPER_LINK_LENGTH = 2.8
DEGREE_ROT_90 = np.pi/2
class Control:

    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.joint_1_sub = rospy.Subscriber("/joint_angle_1", Float64, self.ja1_callback)
        self.joint_3_sub = rospy.Subscriber("/joint_angle_3", Float64, self.ja3_callback)
        self.joint_4_sub = rospy.Subscriber("/joint_angle_4", Float64, self.ja4_callback)
        self.end_effector_sub = rospy.Subscriber("/end_effector_pos", Float64MultiArray, self.end_effector_callback)
        self.ja_1 = None
        self.ja_3 = None
        self.ja_4 = None
        self.end_effector = None

    def ja1_callback(self, ja1):
        self.ja_1 = ja1
        self.kinematics()

    def ja3_callback(self, ja3):
        self.ja_3 = ja3
        self.kinematics()

    def ja4_callback(self, ja4):
        self.ja_4 = ja4
        self.kinematics()

    def end_effector_callback(self, end_effector_pos):
        self.end_effector = end_effector_pos
        self.kinematics()

    def kinematics(self):
        if any([not self.ja_1, not self.ja_3, not self.ja_4, not self.end_effector]):
            print("Subscribers not publishing data: waiting")
        else:
            joint_angles = np.array([self.ja_1.data, self.ja_3.data, self.ja_4.data])
            end_effector = np.array(self.end_effector.data)
            print(f"Published JA1 {self.ja_1.data} JA3 {self.ja_3.data} JA4 {self.ja_4.data}")
            z_matrix = self.z_matrix(joint_angles[0], 4)
            x_matrix = self.x_y_matrix(DEGREE_ROT_90, joint_angles[1], 3.2)
            y_matrix = self.x_y_matrix(DEGREE_ROT_90, joint_angles[2], 2.8)
            homogenous_transformation = z_matrix @ x_matrix @ y_matrix
            coords = homogenous_transformation[:,3][:-1]
            print(f"End effector position baseline {end_effector}: predicted coords: {coords}")

    def x_y_matrix(self, rot_z, rot_x, disp_x):
        return np.array([
            [np.cos(rot_z), -np.sin(rot_z) * np.cos(rot_x),  np.sin(rot_z) * np.sin(rot_x), disp_x * np.cos(rot_z)],
            [np.sin(rot_z),  np.cos(rot_z) * np.cos(rot_x), -np.cos(rot_z) * np.sin(rot_x), disp_x * np.sin(rot_z)],
            [0, np.sin(rot_x), np.cos(rot_x), 0],
            [0,0,0,1]])

    def z_matrix(self, rot_z, disp_z):
        return np.array([
            [np.cos(rot_z), 0,  np.sin(rot_z), 0],
            [np.sin(rot_z),  np.cos(rot_z), -np.cos(rot_z), 0],
            [0, 1, 0, disp_z],
            [0,0,0,1]])

def main(args):
    np.set_printoptions(suppress=True)

    control = Control()
    rospy.init_node("control", anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")


if __name__ == "__main__":
    main(sys.argv)