import cv2
from Layer import PreprocessLayer
import numpy as np

class BinThresholdingLayer(PreprocessLayer):

    def __init__(self, size, low, high) -> None:
        if low < 0 or high > 255 or low > high:
            raise Exception(f"Threshold values are invalid:\nLow: {low}  High: {high}\nRequirements:\n- Low < High\n- Low > 0\n- High < 255")
        super().__init__(size, "binThresholdingLayer")
        self.low = low
        self.high = high

    def process(self, image):
        if not self.__is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bin_image = np.where((image >= self.low) & (image <= self.high), 255, 0).astype(np.uint8)
        return (bin_image, None)

    
    def __is_grayscale(self, image):
        if len(image.shape) == 2:
            return True 
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                return True 
            elif image.shape[2] == 3:
                if (image[:,:,0] == image[:,:,1]).all() and (image[:,:,1] == image[:,:,2]).all():
                    return True 
        return False 