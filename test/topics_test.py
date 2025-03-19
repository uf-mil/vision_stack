import rclpy
import pytest
from PIL import Image
import numpy as np

from vision_stack import VisionStack
from vision_stack import ObjectDetectionLayer

def test_topic_creation():
    rclpy.init()
    print("here")
    vs = VisionStack([
        ObjectDetectionLayer(0.5, 'test.pt'),
    ], 'test')
    img = np.array(Image.open('test/test.jpg'))
    vs.run(img, True)

    topics = [t[0] for t in vs.get_topic_names_and_types()]
    print("Active topics:", topics)  # Debug print

    expected_topic = "/vs_test/test/objectDetection_test_0"  # Replace with the actual topic name
    assert expected_topic in topics, f"Expected topic {expected_topic} not found!"

    vs.destroy_node()
    rclpy.shutdown()



