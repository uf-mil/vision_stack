from .layers.Layer import Layer
from typing import List
import time
import numpy as np

import copy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

try:
    import rclpy
    from rclpy.publisher import Publisher
    from rclpy.node import Node
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
except:
    print("mil_ros_tools package is not available")

NUM_COLS = 3

class Image_Publisher:
    """
    Publishes OpenCV image mats directly to a ROS2 topic, avoiding the need for
    continual conversion.

    Attributes:
        bridge (CvBridge): The ROS2 bridge to OpenCV. Created upon instantiation.
        encoding (str): The encoding of the images. Supplied upon creation.
            Defaults to ``bgr8``.
        im_pub (rclpy.Publisher): The ROS2 publisher responsible for publishing
            images to a ROS topic. The topic name and queue size are supplied
            through the constructor.
    """

    def __init__(self, topic: str, node:Node, encoding: str = "infer", queue_size: int = 1):
        print(f"Topic Name: {topic}")
        self.bridge = CvBridge()
        self.encoding = encoding
        self.node:Node = node
        self.im_pub:Publisher = node.create_publisher(Image, topic, qos_profile=queue_size)

    def get_num_connections(self) -> int:
        return self.im_pub.get_subscription_count()

    
    # used to infer the encoding of an image
    # useful when publishing different image encodings from same Image_Publisher instance
    @staticmethod
    def guess_ros_encoding(img: np.ndarray) -> str:
        """
        Guess the appropriate ROS image encoding string based on a NumPy image array.

        Parameters:
            img (np.ndarray): The input image as a NumPy array.

        Returns:
            str: A string representing the ROS-compatible image encoding.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        if img.dtype != np.uint8:
            raise ValueError(f"Unsupported dtype: {img.dtype}. Only uint8 (8-bit) is supported.")

        shape = img.shape

        # Grayscale image (H, W)
        if len(shape) == 2:
            return "mono8"

        # Color image (H, W, C)
        if len(shape) == 3:
            channels = shape[2]
            if channels == 1:
                return "mono8"
            elif channels == 3:
                return "bgr8"  # Assuming OpenCV-style (BGR)
            elif channels == 4:
                return "bgra8"
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        
        raise ValueError(f"Unsupported image shape: {shape}")

    def publish(self, cv_image: np.ndarray):
        """
        Publishes an OpenCV image mat to the ROS topic. :class:`CvBridgeError`
        exceptions are caught and logged.
        """
        try:
            encoding = self.encoding if self.encoding != "infer" else Image_Publisher.guess_ros_encoding(cv_image)
            image_message = self.bridge.cv2_to_imgmsg(cv_image, encoding)
            self.im_pub.publish(image_message)
        except CvBridgeError as e:
            # Intentionally absorb CvBridge Errors
            self.node.get_logger().error(str(e))

class VisionStack(Node):
    static_id = 0
    def __init__(self, layers:List[Layer], unique_name = ""):
        """
        An array like object that holds layers that are processed in order from index: 0 to the end of the array.
        """
        super().__init__(f"vs_{unique_name}")
        self.layers = layers
        self.analysis_dict = {
            "updated_at": time.localtime()
        }
        self.processed_image = None
        VisionStack.static_id += 1
        self.instance_id = VisionStack.static_id
        self.unique_name = unique_name
    
    def __getitem__(self, index):
        return self.layers[index]
    
    def __setitem__(self, index, layer:Layer):
        self.layers[index] = layer

    def insert(self, index, layer:Layer):
        self.layers.insert(index, layer)
    
    def push(self, layer:Layer):
        self.layers.append(layer)
    
    def pop(self, index = -1):
        self.layers.pop(index)
    
    def run(self, in_image, verbose = False):
        processed_image = copy.copy(in_image)
        self.analysis_dict["updated_at"] = time.localtime()

        num_rows = -(-len(self.layers) // NUM_COLS)

        ros_is_running = False

        for i, layer in enumerate(self.layers):
            layer_process = layer.process(processed_image)
            processed_image = layer_process[0]
            topic_name = f"/{self.instance_id if self.unique_name == '' else self.unique_name}/{layer.name}_{i}"

            if layer_process[1] is not None:
                self.analysis_dict[f"{layer.name}_{i}"] = layer_process[1]

                # Try publishing message from layer
                if layer.msg:
                    try:
                        analysis_pub:Publisher = self.create_publisher(type(layer.msg), topic_name+"/analysis", queue_size=10)
                        analysis_pub.publish(layer.msg)
                    except Exception as e:
                        print(f"Could not publish ros message:\n{e}")

            if verbose: # Create a display showing how each layer processes the image before it
                try:
                    # when debugging we expect different image encodings (maybe there's an RGB layer, then BW, etc.)
                    verbose_layer_pub = Image_Publisher(f"/front_cam/{layer.__class__.__name__}", self)
                    verbose_layer_pub.publish(processed_image)
                    ros_is_running = True
                except:
                    print("ROS is not running")
                    fig, axes = plt.subplots(num_rows, NUM_COLS)
                    row_index = i // NUM_COLS
                    col_index = i % NUM_COLS

                    if num_rows == 1:
                        axes[col_index].imshow(processed_image)
                        axes[col_index].set_title(layer.name + "_" + str(i))
                    else:
                        axes[row_index, col_index].imshow(processed_image)
                        axes[row_index, col_index].set_title(layer.name + "_" + str(i))                
                    if num_rows == 1:
                        axes[col_index].axis('off')
                    else:
                        axes[row_index, col_index].axis('off')
            
            else:
                print("Not creating publisher")

        self.processed_image = processed_image

        if verbose and not ros_is_running:
            plt.tight_layout()
            plt.show()
    
    def visualize(self):
        for layer in self.layers:
            print(layer.name)
