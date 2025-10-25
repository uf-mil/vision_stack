#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
vision_stack_base_path = str(Path(__file__).parent.parent)

from vision_stack import VisionStack, BinThresholdingLayer, CannyLayer, ColorMagnificationLayer, CustomLayer, GaussianLayer, GrayscaleLayer, HoughTransformLayer, MinMaxNormalizationLayer, ZScoreNormalizationLayer, RobustScalingLayer, ObjectDetectionLayer, ResizeLayer, RGBMagnificationLayer, SobelLayer, UnderWaterImageEnhancementLayer

bridge = CvBridge()

class ObjectDetectionTest(Node):
    def __init__(self):

        super().__init__('Vision_Stack_Subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/front_cam/image_raw',
            self.listener_callback,
            10)
        
        self.vs = VisionStack(
            layers=[
                # Include as many layers in any combination as you need
            

                # Stole the comments and layer initializations below from Daniel's README on the VisionStack repo

                ObjectDetectionLayer(conf_thres=0.5, weights_file=f"{vision_stack_base_path}/vision_stack/weights/test.pt", pass_post_detection_img = False), # Access YOLO weights and make predictions on the image provided by the previous layer. Setting pass_post_processing_img will push the image with bounding boxes to the next layer.
                BinThresholdingLayer(150,250), # Converts image to grayscale if image is not grayscale and extracts pixels with values between 150 and 250.
                CannyLayer(50,100), # Simplified canny filter that uses cv2.Canny to pass a canny filter over an image with the low value (50) threshold for soft edge detection and the high value (100) for strong edges detection.
                ColorMagnificationLayer((23,156,234)), # Highlights objects with this color (23,156,234) in an image.
                GaussianLayer((11,11), 50), # Pass a gaussian filter of kernal size (11,11) (kernal size must be of odd numbered dimensions) with a sigma value of 50.
                GrayscaleLayer(), # Convert image to grayscale
                HoughTransformLayer(threshold=100, min_line_length=20, max_line_gap=10, pass_post_processing_img=True), # Pass a Hough Transform filter over an image to extract lines. Setting pass_post_processing_img will push the image with hough transform lines to the next layer.
                MinMaxNormalizationLayer(),
                ZScoreNormalizationLayer(),
                ResizeLayer(960, 608), # Resize the image from the previous layer.
                RGBMagnificationLayer('G'), # Magnifies the provided channel respectively (options are 'R', 'G', 'B')
                UnderWaterImageEnhancementLayer(), # Uses a generative AI model to improve underwater images (good for murky waters).
                
            

                # **********Layers with issues****************
                # RobustScalingLayer(), This layer produces a floating point encoding type which is not allowed by the publisher, so don't use this alyer
                # SobelLayer(5), @ Fails for the same reason as RobustScalingLayer
            ],
        )


    # Runs vision stack on every callback
    def listener_callback(self, msg):

        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.vs.run(cv_image, True)


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = ObjectDetectionTest()
    rclpy.spin(minimal_subscriber)



if __name__ == "__main__":
    main()