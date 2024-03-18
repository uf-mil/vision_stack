import cv2
import numpy as np
from Layer import PreprocessLayer

class MinMaxNormalizationLayer(PreprocessLayer):
    def __init__(self, size) -> None:
        super().__init__(size)
    
    def process(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / (max_val - min_val)
        return (normalized_image, None)
    

class ZScoreNormalizationLayer(PreprocessLayer):
    def __init__(self, size) -> None:
        super().__init__(size)
    
    def process(self, image):
        mean = np.mean(image)
        std_dev = np.std(image)
        normalized_image = (image - mean) / std_dev
        return (normalized_image, None)

class RobustScalingLayer(PreprocessLayer):
    def __init__(self, size) -> None:
        super().__init__(size)
    
    def process(self, image):
        median = np.median(image)
        iqr = np.percentile(image, 75) - np.percentile(image, 25)
        normalized_image = (image - median) / iqr
        return (normalized_image, None)