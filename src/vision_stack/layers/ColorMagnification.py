import numpy as np
import cv2

from .Layer import PreprocessLayer

class ColorMagnificationLayer(PreprocessLayer):
    def __init__ (self, color_tuple):
        """
        Given an RGB tuple (e.g. (125,34,56)), get the absolute difference between that color and the pixels on the image by subtracting the provided color_tuple from each channel of the pixel. Then the absolute error is subtracted from an all white image which provides an image detailing similarity. The values of the similarity image are then squared to get 'squared similarity'.

        Parameters:
            color_tuple: e.g. (125, 34, 56)
        """
        super().__init__(name="colorMagnification")
        
        if not (len(color_tuple) != 3):
            raise Exception(f"Invalid color_tuple received [{color_tuple}], expected a color_tuple that is of size 3") # Allowing 3-tuple with values outside of 0-255 for experimental use
        else:
            self.color = color_tuple

    def process(self, image):
        if len(image.shape) == 2: # Grayscale image (height, width)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        height, width, channels = image.shape

        color_image = np.ones((height, width, channels), dtype=np.uint8) * np.array(self.color, dtype=np.uint8)

        absolute_error = cv2.subtract(image, color_image)

        all_white_image = np.ones((height, width, 3), dtype=np.uint8) * 255

        all_white_image = all_white_image[:height, :width, :]

        similarity_image = np.square(all_white_image.astype(np.float32) - absolute_error)

        similarity_image = (similarity_image - np.min(similarity_image)) / (np.max(similarity_image) - np.min(similarity_image)) * 255

        similarity_image = similarity_image.astype(np.uint8)

        return (similarity_image, None)