import cv2

from Layer import PreprocessLayer

class CannyLayer(PreprocessLayer):

    def __init__(self, size, ksize) -> None:
        """
            ksize: parameter to change the size of the Sobel kernel. A larger kernel size can capture larger gradients but may also result in more noise.
        """
        super().__init__(size, "sobel")
        self.ksize = ksize

    def process(self, image):
        if not self.__is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.ksize)

        edges_image = cv2.magnitude(sobel_x, sobel_y)

        return (edges_image, None)
    

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