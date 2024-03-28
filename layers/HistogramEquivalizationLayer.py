import cv2
from Layer import PreprocessLayer

class HistogramEqualizaionLayer(PreprocessLayer):

    def __init__(self, size) -> None:
        """
        Equalizes the pixels between pixel intesities based on the pixel histogram

        Requires Grayscale images to process
        """
        super().__init__(size, "histogram-equalization")


    def process(self, image):
        return (cv2.equalizeHist(image), None)

class HistogramAdaptiveEqualizationLayer(PreprocessLayer):
    def __init__(self, size) -> None:
        """
        Adaptively equalizes the pixel histogram

        Requires Grayscale images to process
        """
        super().__init__(size, "adaptive-histogram-equalization")

    def process(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return (clahe.apply(image), None)
