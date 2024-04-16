import cv2
from .Layer import PreprocessLayer

class GrayscaleLayer(PreprocessLayer):

    def __init__(self) -> None:
        """
        Converts image to grayscale.
        """
        super().__init__("grayscale")

    
    def process(self, image):
        return (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)