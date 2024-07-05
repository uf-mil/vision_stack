from .layers.Layer import Layer
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt

try:
    from mil_ros_tools import Image_Publisher
except:
    plt.switch_backend('TkAgg')
    print("mil_ros_tools package is not available")

NUM_COLS = 3

class VisionStack:
    def __init__(self, layers:List[Layer]):
        """
        An array like object that holds layers that are processed in order from index: 0 to the end of the array.
        """
        self.layers = layers
        self.analysis_dict = {
            "updated_at": datetime.now()
        }
        self.processed_image = None
    
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
        fig, axes = plt.subplots(num_rows, NUM_COLS)

        ros_is_running = False

        for i, layer in enumerate(self.layers):
            layer_process = layer.process(processed_image)
            processed_image = layer_process[0]

            if layer_process[1] is not None:
                self.analysis_dict[layer.name] = layer_process[1]

            if verbose: # Create a display showing how each layer processes the image before it
                try:
                    verbose_layer_pub = Image_Publisher(f"~{layer.name}_{i}")
                    verbose_layer_pub.publish(processed_image)
                    ros_is_running = True
                except:
                    print("ros is not running")
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
