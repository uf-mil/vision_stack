import sys
import os

print(os.getcwd())
sys.path.append(os.getcwd() + "/layers")

from layers import ResizeLayer, GaussianLayer, GrayscaleLayer, BinThresholdingLayer, HoughTransformLayer, RGBMagnificationLayer, UnderwaterEnhancementLayer, CustomLayer, ObjectDetectionLayer
from layers.Layer import Layer
from typing import List, Tuple
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

NUM_COLS = 3

class VisionStack:
    def __init__(self, layers:List[Layer], input_size:Tuple[int,int]):
        """
        An array like object that holds layers that are processed in order from index: 0 to the end of the array.
        """
        self.layers = layers
        self.input_size = input_size,
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

        for i, layer in enumerate(self.layers):
            layer_process = layer.process(processed_image)
            processed_image = layer_process[0]

            if layer_process[1] is not None:
                self.analysis_dict[layer.name] = layer_process[1]

            if verbose: # Create a display showing how each layer processes the image before it
                row_index = i // NUM_COLS
                col_index = i % NUM_COLS

                if num_rows == 1:
                    axes[col_index].imshow(processed_image)
                    axes[col_index].set_title(layer.name)
                else:
                    axes[row_index, col_index].imshow(processed_image)
                    axes[row_index, col_index].set_title(layer.name)                
                if num_rows == 1:
                    axes[col_index].axis('off')
                else:
                    axes[row_index, col_index].axis('off')
            
        self.processed_image = processed_image

        if verbose:
            plt.tight_layout()
            plt.show()
    
    def visualize(self):
        for layer in self.layers:
            print(layer.name)

if __name__ == "__main__":
    SIZE = (960,608)
    stack = VisionStack([ResizeLayer.ResizeLayer((0,0), 960, 608), 
                        #  GaussianLayer.GaussianLayer(SIZE, (5,5), 10)
                         ], SIZE)
    img = Image.open(os.path.join(os.path.dirname(__file__), 'imgs/sample.png'))
    CLASSES = [
        "buoy_abydos_serpenscaput",
        "buoy_abydos_taurus",
        "buoy_earth_auriga",
        "buoy_earth_cetus",
        "gate_abydos",
        "gate_earth",
    ]
    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 155, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    # img.show()
    # stack.visualize()
    def funcToMyLayer(img, args):
        return(img, None)
    
    # stack.push(CustomLayer.CustomLayer(SIZE, SIZE, "myLayer", funcToMyLayer, []))
    stack.push(UnderwaterEnhancementLayer.UnderWaterImageEnhancementLayer(SIZE))
    # stack.push(GrayscaleLayer.GrayscaleLayer(SIZE))
    # stack.push(BinThresholdingLayer.BinThresholdingLayer(SIZE, 150, 255))
    # stack.push(HoughTransformLayer.HoughTransformLayer(SIZE, 100, 20, 10, True))
    # stack.push(GaussianLayer.GaussianLayer(SIZE, (5,5), 15))
    stack.push(ObjectDetectionLayer.ObjectDetectionLayer(SIZE,SIZE, "../ml/weights/robosub24.pt", 0.5, 0.5, CLASSES, COLORS, True))
    # print()
    # stack.visualize()
    stack.run(np.array(img), True)
