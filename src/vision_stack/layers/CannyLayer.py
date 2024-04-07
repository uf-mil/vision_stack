import cv2

from .Layer import PreprocessLayer

class CannyLayer(PreprocessLayer):

    def __init__(self, size, low, high) -> None:
        """
        Applies a canny edge detection layer over a grayscaled image if not already grayscaled.
        
        Parameters:
            low: This value is used to identify weak edges in the image. Any edge with a gradient value below this threshold is discarded. Setting this threshold too low can result in detecting a lot of noise as edges.

            high: This value is used to identify strong edges in the image. Any edge with a gradient value above this threshold is considered a strong edge. These strong edges are the ones that will be finally selected as edges. Setting this threshold too high may result in missing weak edges.
        """
        super().__init__(size, "canny")
        self.low = low
        self.high = high

    def process(self, image):
        if not self.__is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, self.low, self.high)
        return (edges, None)
    

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