from .layers.Layer import Layer
from typing import List
from datetime import datetime
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

try:
    import rclpy
    from rclpy.publisher import Publisher
    from rclpy.node import Node
except:
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
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

    def __init__(self, topic: str, node:Node, encoding: str = "bgr8", queue_size: int = 1):
        self.bridge = CvBridge()
        self.encoding = encoding
        self.node:Node = node
        self.im_pub:Publisher = node.create_publisher(Image, topic, queue_size=queue_size)

    def get_num_connections(self) -> int:
        return self.im_pub.get_subscription_count()

    def publish(self, cv_image: np.ndarray):
        """
        Publishes an OpenCV image mat to the ROS topic. :class:`CvBridgeError`
        exceptions are caught and logged.
        """
        try:
            image_message = self.bridge.cv2_to_imgmsg(cv_image, self.encoding)
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
        super.__init__(f"vs_{unique_name}")
        self.layers = layers
        self.analysis_dict = {
            "updated_at": datetime.now()
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
        processed_image = in_image.copy()
        self.analysis_dict["updated_at"] = datetime.now()

        num_rows = -(-len(self.layers) // NUM_COLS)

        ros_is_running = False

        for i, layer in enumerate(self.layers):
            layer_process = layer.process(processed_image)
            processed_image = layer_process[0]
            topic_name = f"~{self.instance_id if self.unique_name == '' else self.unique_name}/{layer.name}_{i}"

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
                    verbose_layer_pub = Image_Publisher(topic_name)
                    verbose_layer_pub.publish(processed_image)
                    ros_is_running = True
                except:
                    print("ROS is not running")
                    fig, axes = plt.subplots(num_rows, NUM_COLS)
                    row_index = i // NUM_COLS
                    col_index = i % NUM_COLS

                    if num_rows == 1:
                        axes[col_index].imshow(processed_image)
                        axes[col_index].set_title(layer.name + "_" + i)
                    else:
                        axes[row_index, col_index].imshow(processed_image)
                        axes[row_index, col_index].set_title(layer.name + "_" + i)                
                    if num_rows == 1:
                        axes[col_index].axis('off')
                    else:
                        axes[row_index, col_index].axis('off')
            
        self.processed_image = processed_image

        if verbose and not ros_is_running:
            plt.tight_layout()
            plt.show()
    
    def visualize(self):
        for layer in self.layers:
            print(layer.name)
