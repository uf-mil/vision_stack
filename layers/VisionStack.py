import ResizeLayer, GaussianLayer, GrayscaleLayer, BinThresholdingLayer, HoughTransformLayer
from Layer import Layer
from typing import List, Tuple
from datetime import datetime
from PIL import Image
import numpy as np


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
    
    def run(self, in_image, verbose = False):
        processed_image = in_image.copy()
        self.analysis_dict["updated_at"] = datetime.now()
        for layer in self.layers:
            layer_process = layer.process(processed_image)
            processed_image = layer_process[0]
            if layer_process[1] is not None:
                self.analysis_dict[layer.name] = layer_process[1]
        self.processed_image = processed_image
    
    def visualize(self):
        for layer in self.layers:
            print(layer.name)

if __name__ == "__main__":
    SIZE = (400,400)
    stack = VisionStack([ResizeLayer.ResizeLayer((0,0), 400, 400), 
                         GaussianLayer.GaussianLayer(SIZE, (5,5), 15), 
                         GrayscaleLayer.GrayscaleLayer(SIZE), 
                         BinThresholdingLayer.BinThresholdingLayer(SIZE, 100, 255), 
                         HoughTransformLayer.HoughTransformLayer(SIZE, 10, 1, 5, True)], SIZE)
    img = Image.open("imgs/original.jpg")
    img.show()
    stack.run(np.array(img))
    processed_img = Image.fromarray(stack.processed_image)
    processed_img.show()
