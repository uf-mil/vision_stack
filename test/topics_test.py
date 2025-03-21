import rclpy
import pytest
from PIL import Image
import numpy as np

from vision_stack import VisionStack
from vision_stack import BinThresholdingLayer, CannyLayer, ColorMagnificationLayer, CustomLayer, GaussianLayer, GrayscaleLayer, HoughTransformLayer, MinMaxNormalizationLayer, ZScoreNormalizationLayer, ObjectDetectionLayer, ResizeLayer, RGBMagnificationLayer, RGBtoBGRLayer, SobelLayer, UnderWaterImageEnhancementLayer 

def test_topic_creation():
    rclpy.init()
    print("here")
    def custom_process(img, args):
        return (img, None)

    vs = VisionStack([
        UnderWaterImageEnhancementLayer(),
        ObjectDetectionLayer(0.5, 'test.pt'),
        BinThresholdingLayer(0,255),
        CannyLayer(0, 255),
        ColorMagnificationLayer((125,34,56)),
        CustomLayer("custom", custom_process),
        GaussianLayer((5,5), 0.5),
        GrayscaleLayer(),
        HoughTransformLayer(),
        MinMaxNormalizationLayer(),
        ZScoreNormalizationLayer(),
        ResizeLayer(400,600),
        RGBMagnificationLayer('R'),
        RGBtoBGRLayer(),
        SobelLayer(5)
    ], 'test')

    img = np.array(Image.open('test/test.jpg'))
    vs.run(img, True)
    print("here2")

    topics = [t[0] for t in vs.get_topic_names_and_types()]
    print("Active topics:", topics)  # Debug print

    expected_topic = [
        "/vs_test/test/underwaterImageEnhancement_0",
        "/vs_test/test/objectDetection_test_1",
        "/vs_test/test/binThresholding_2",
        "/vs_test/test/canny_3",
        "/vs_test/test/colorMagnification_4",
        "/vs_test/test/custom_5",
        "/vs_test/test/gaussian_6",
        "/vs_test/test/grayscale_7",
        "/vs_test/test/houghTransform_8",
        "/vs_test/test/minMaxNorm_9",
        "/vs_test/test/zScoreNorm_10",
        "/vs_test/test/resize_11",
        "/vs_test/test/rgbMagnification_12",
        "/vs_test/test/rgb_to_bgr_13",
        "/vs_test/test/sobel_14",
        ]
    
    for t in expected_topic:
        assert t in topics, f"Expected topic {expected_topic} not found!"

    vs.destroy_node()
    rclpy.shutdown()



