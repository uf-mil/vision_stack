import cv2

from .Layer import PreprocessLayer


class HistogramEqualizationLayer(PreprocessLayer):
    
    def __init__(self, grid_size=8) -> None:
        """
        Passes an adaptive cv2 histogram equalization over the image.
        
        Paramters:
            gridsize: Size of grid for histogram equalization. Input image will be divided into 
            equally sized rectangular tiles. gridsize defines the number of tiles in row and column.
        """
        super().__init__("histogramequalization")
        self.grid_size = grid_size
        
    def process(self, image):
        if len(image.shape) == 2: #Basic Histogram Equalization for grayscale image
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(self.grid_size,self.grid_size))
            equalized_image = clahe.apply(image)
            return (equalized_image, None)
        else: # Histogram Equalization for colored images
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(self.grid_size,self.grid_size))
            equalized_l = clahe.apply(l)
            
            equalized_lab_image = cv2.merge((equalized_l, a, b))
            equalized_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)
            
            return(equalized_image, None)
