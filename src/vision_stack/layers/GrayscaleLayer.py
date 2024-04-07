import cv2
from .Layer import PreprocessLayer

class GrayscaleLayer(PreprocessLayer):

    def __init__(self, size) -> None:
        """
        Converts image to grayscale.
        """
        super().__init__(size, "grayscale")

    
    def process(self, image):
        return (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)