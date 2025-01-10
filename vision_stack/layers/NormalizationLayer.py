import cv2
import numpy as np
from .Layer import PreprocessLayer

class MinMaxNormalizationLayer(PreprocessLayer):
    def __init__(self) -> None:
        """
        Normalizes an image through by getting the minimum pixel value, the maximum pixel value, subtracting the minimum value from all pixels, and dividing by the difference between the maximum pixel value and the minimum pixel value.
        """
        super().__init__("minMaxNorm")
    
    def process(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / (max_val - min_val)
        return (normalized_image, None)
    

class ZScoreNormalizationLayer(PreprocessLayer):
    def __init__(self) -> None:
        """
        Normalizes the image by calculating the mean and standard deviation of the pixels in the image, then subtracting the mean from all pixel values and dividing by the standard deviation.
        """
        super().__init__()
    
    def process(self, image):
        mean = np.mean(image)
        std_dev = np.std(image)
        normalized_image = (image - mean) / std_dev
        return (normalized_image, None)

class RobustScalingLayer(PreprocessLayer):
    def __init__(self) -> None:
        """
        Normalizes the image by calculating the median and the IQR of all the pixels, then subtracting the median form the pixels and dividing by the IQR.
        """
        super().__init__()
    
    def process(self, image):
        median = np.median(image)
        iqr = np.percentile(image, 75) - np.percentile(image, 25)
        normalized_image = (image - median) / iqr
        return (normalized_image, None)