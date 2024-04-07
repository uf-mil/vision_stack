import cv2
from Layer import PreprocessLayer

class ResizeLayer(PreprocessLayer):

    def __init__(self, size, new_width, new_height) -> None:
        """
        Resizes the image to be of new dimensions.

        Parameters:
            new_width: Image width will be resized to the new pixel width value.
            new_height: Image height will be resized to the new pixel height value.
        """
        super().__init__(size, "resize")
        self.new_width = new_width
        self.new_height = new_height

    
    def process(self, image):
        resized_image = cv2.resize(image, (self.new_width, self.new_height))
        return (resized_image, None)