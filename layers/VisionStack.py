import ResizeLayer, GaussianLayer, GrayscaleLayer, BinThresholdingLayer, HoughTransformLayer, RGBMagnificationLayer, UnderwaterEnhancementLayer, CustomLayer
from Layer import Layer
from typing import List, Tuple
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

NUM_COLS = 3

class VisionStack:
    def __init__(self, layers:List[Layer], input_size:Tuple[int,int]):
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
    SIZE = (900,600)
    stack = VisionStack([ResizeLayer.ResizeLayer((0,0), 900, 400), 
                         GaussianLayer.GaussianLayer(SIZE, (5,5), 10)], SIZE)
    img = Image.open("../imgs/image1.jpg")
    # img.show()
    # stack.visualize()
    def funcToMyLayer(img, args):
        return(img, None)
    
    stack.push(CustomLayer.CustomLayer(SIZE, SIZE, "myLayer", funcToMyLayer, []))
    stack.push(UnderwaterEnhancementLayer.UnderWaterImageEnhancementLayer(SIZE))
    stack.push(GrayscaleLayer.GrayscaleLayer(SIZE))
    stack.push(BinThresholdingLayer.BinThresholdingLayer(SIZE, 150, 255))
    stack.push(HoughTransformLayer.HoughTransformLayer(SIZE, 100, 20, 10, True))
    stack.push(GaussianLayer.GaussianLayer(SIZE, (5,5), 15))
    # print()
    # stack.visualize()
    stack.run(np.array(img), True)
