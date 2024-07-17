from .layers.Layer import Layer
from typing import List
from datetime import datetime

try:
    import rospy
    from mil_ros_tools import Image_Publisher
except:
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    print("mil_ros_tools package is not available")

NUM_COLS = 3

class VisionStack:
    static_id = 0
    def __init__(self, layers:List[Layer], unique_name = ""):
        """
        An array like object that holds layers that are processed in order from index: 0 to the end of the array.
        """
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
                        analysis_pub = rospy.Publisher(topic_name+"/analysis", type(layer.msg), queue_size=10)
                        analysis_pub.publish(layer.msg)
                    except Exception as e:
                        print(f"Could not publish ros message:\n{e}")

            if verbose: # Create a display showing how each layer processes the image before it
                try:
                    verbose_layer_pub = Image_Publisher(topic_name)
                    verbose_layer_pub.publish(processed_image)
                    ros_is_running = True
                except:
                    print("ros is not running")
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
