import cv2
from Layer import PreprocessLayer

class ResizeLayer(PreprocessLayer):

    def __init__(self, size, new_width, new_height) -> None:
        super().__init__(size, "resize")
        self.new_width = new_width
        self.new_height = new_height

    
    def process(self, image):
        resized_image = cv2.resize(image, (self.new_width, self.new_height))
        return (resized_image, None)