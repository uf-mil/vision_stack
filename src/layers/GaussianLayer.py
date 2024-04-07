import cv2
from Layer import PreprocessLayer

class GaussianLayer(PreprocessLayer):

    def __init__(self, size, kernel_size, sigma) -> None:
        """
        Passes a cv2.GaussianBlure over image.

        Parameters:
            kernel_size (tuple): Size of the kernel for the Gaussian filter
            sigma (float): Standard deviation of the Gaussian kernel
        """
        super().__init__(size, "gaussian")
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def process(self, image):
        return (cv2.GaussianBlur(image, self.kernel_size, self.sigma), None)