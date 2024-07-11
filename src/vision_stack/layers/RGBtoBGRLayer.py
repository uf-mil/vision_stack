import cv2
from .Layer import PreprocessLayer

class RGBtoBGRLayer(PreprocessLayer):

    def __init__(self) -> None:
        """
        Converts image to grayscale.
        """
        super().__init__("grayscale")

    
    def process(self, image):
        if len(image.shape) == 2: # Grayscale image (height, width)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),None)