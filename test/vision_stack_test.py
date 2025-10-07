#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# import rospy
# from image_geometry import PinholeCameraModel
# from mil_ros_tools import (
#     Image_Subscriber,
# )


from vision_stack import ResizeLayer, UnderWaterImageEnhancementLayer, VisionStack


class ObjectDetectionTest(Node):
    def __init__(self):

        super().__init__('Vision_Stack_Subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/front_cam/image_raw',
            self.listener_callback,
            10)
        
        # camera = rospy.get_param("~image_topic", "/front_cam/image_raw")
        self.vs = VisionStack(
            layers=[
                ResizeLayer(960, 608),
                UnderWaterImageEnhancementLayer(),
            ],
        )

        print("Finishes vision stack initialization\n")
        # self.image_sub = Image_Subscriber(camera, self.detection_callback)
        # self.camera_info = self.image_sub.wait_for_camera_info()
        # assert self.camera_info is not None
        # self.cam = PinholeCameraModel()
        # self.cam.fromCameraInfo(self.camera_info)


    def listener_callback(self, msg):
        # print("Was called")
        print(f'Image: {msg.width}x{msg.height}, encoding: {msg.encoding}')

        # Create Image from array
        # self.vs.run(msg, True)
        # print(f"I heard: {msg.data}")


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ObjectDetectionTest()

    print("Starting the subscriber")
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # minimal_subscriber.destroy_node()
    # rclpy.shutdown()


if __name__ == "__main__":
    # rospy.init_node("vision_pipeline_test")
    # ObjectDetectionTest()
    # rospy.spin()
    main()