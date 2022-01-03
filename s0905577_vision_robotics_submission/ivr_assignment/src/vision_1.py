#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

BOTTOM_LINK_LENGTH = 4
MIDDLE_LINK_LENGTH = 3.2
UPPER_LINK_LENGTH = 2.8

RED_MASK = [
    (0, 0, 100),
    (0, 0, 255)
]

YELLOW_MASK = [
    (0, 100, 100),
    (0, 255, 255)
]

BLUE_MASK = [
    (100, 0, 0),
    (255, 0, 0)
]

GREEN_MASK = [
    (0, 100, 0),
    (0, 255, 0)
]

BASIS_X_VECTOR = np.array([1, 0, 0])
BASIS_Y_VECTOR = np.array([0, 1, 0])
BASIS_Z_VECTOR = np.array([0, 0, 1])

class BlobDetection:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.image_x_sub = rospy.Subscriber("image_topic1", Image, self.im1callback)
        self.image_y_sub = rospy.Subscriber("image_topic2", Image, self.im2callback)
        self.joint_2 = rospy.Publisher("joint_angle_2", Float64, queue_size=10)
        self.joint_3 = rospy.Publisher("joint_angle_3", Float64, queue_size=10)
        self.joint_4 = rospy.Publisher("joint_angle_4", Float64, queue_size=10)

        self.camera_x = None
        self.camera_y = None
        self.joint_state = None

    def im1callback(self, image):
        self.camera_x = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.blob_detect()

    def im2callback(self, image):
        self.camera_y = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.blob_detect()

    def blob_detect(self):
        if np.any(self.camera_x) and np.any(self.camera_y):
            angles = self._detect_angles(self.camera_x, self.camera_y)
            ja2, ja3, ja4 = Float64(), Float64(), Float64()
            ja2.data = angles[0]
            ja3.data = angles[1]
            ja4.data = angles[2]

            self.joint_2.publish(ja2)
            self.joint_3.publish(ja3)
            self.joint_4.publish(ja4)
        else:
            print("Susbcribers not synchronized yet: waiting")

    def _calculate_moment(self, image_x, image_y, mask):
        blob_x, blob_y = (
            self._detect_blob(image_x, mask[0], mask[1]),
            self._detect_blob(image_y, mask[0], mask[1])
        )
        loc = np.array([
            blob_y[0],
            blob_x[0],
            (blob_y[1] + blob_x[1]) /2
        ])

        return loc

    def _identify_locations(self, image_x, image_y):
        try:
            yellow_loc = self._calculate_moment(image_x, image_y, YELLOW_MASK)
            self.last_yellow_success = yellow_loc
        except:
            print(f"yellow detection failed. reverting to {self.last_yellow_success}")
            yellow_loc = self.last_yellow_success

        try:
            blue_loc = self._calculate_moment(image_x, image_y, BLUE_MASK)
            self.last_blue_success = blue_loc
        except:
            print(f"blue detection failed. reverting to {self.last_blue_success}")
            blue_loc = self.last_blue_success

        try:
            red_loc = self._calculate_moment(image_x, image_y, RED_MASK)
            self.last_red_success = red_loc
        except:
            print(f"red detection failed. reverting to {self.last_red_success}")
            red_loc = self.last_red_success

        return yellow_loc, blue_loc, red_loc

    def _scale_to_meter(self, upstream, downstream, scaler):
        distance = upstream - downstream
        scaler = scaler / np.sqrt(np.sum(np.square(downstream - upstream)))
        return distance * scaler

    def _detect_angles(self, image_x, image_y):

        yellow_loc, blue_loc, red_loc = self._identify_locations(image_x, image_y)

        dist_yellow_blue = self._scale_to_meter(yellow_loc, blue_loc, MIDDLE_LINK_LENGTH)
        dist_blue_red = self._scale_to_meter(blue_loc, red_loc, UPPER_LINK_LENGTH)

        ja2 = self._calculate_jablue(dist_yellow_blue, BASIS_X_VECTOR, BASIS_Y_VECTOR)
        ja3 = self._calculate_jablue(dist_yellow_blue, BASIS_Y_VECTOR, BASIS_X_VECTOR) - np.pi
        ja4 = self._calculate_ja4(dist_blue_red, dist_yellow_blue)

        if blue_loc[0] < yellow_loc[0]:
            ja2 = ja2*-1

        if blue_loc[1] < yellow_loc[1]:
            ja3 = ja3*-1

        if red_loc[0] < blue_loc[0]:
            ja4 = ja4*-1

        return [ja2, ja3, ja4]

    def _calculate_angle(self, vector_1, identity_vector):
        vector_1, identity_vector = (
            vector_1 / np.linalg.norm(vector_1),
            identity_vector / np.linalg.norm(identity_vector)
        )
        return np.arccos(vector_1 @ identity_vector)

    def _calculate_ja4(self, vector_1, vector_2):
        angle = self._calculate_angle(vector_1.copy(), vector_2.copy())
        return angle

    def _detect_blob(self, image, mask_lower, mask_upper):
        mask = cv2.inRange(image, mask_lower, mask_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            result = np.array([cx, cy])
            return result
        except:
            raise Exception()

def main(args):
    np.set_printoptions(suppress=True)

    blob_detect = BlobDetection()
    rospy.init_node("blob_detection", anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")


if __name__ == "__main__":
    main(sys.argv)