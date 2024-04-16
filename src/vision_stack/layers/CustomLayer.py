from .Layer import AnalysisLayer
import numpy as np

class CustomLayer(AnalysisLayer):
    def __init__(self, name, process:callable, *args) -> None:
        """
        Allows you to perform any custom image manipulation.
        
        Parameters:
            in_size: Input size of the image before processing. Ex. (400,400)
            out_size: Output size of the image post processing. Ex. (300,300)
            name: The name of the layer appears over the image after running 'run()' on a vision stack and appears when collecting data after processing images in the analysis dict of the vision stack object, given information in the second slot of the tuple returned by the custom process is not None.
            process: A function passed in that should be formatted:
                     def customProcess(img, args) -> (img, Any)
        """
        image_size = (40, 40)
        random_image = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
        self.args = args
        super().__init__(name)

        if callable(process):
            try:
                output = process(random_image, args)
                assert isinstance(output[0], np.ndarray)
                assert len(output) == 2
                self.in_process = lambda img: process(img, self.args)
            except Exception as e:
                raise Exception(f"The return value of the process is invalid, must be tuple with:\ntuple[0] -> 'numpy.ndarray'\ntuple[1] -> Some value (Default to None)\nException:{e}")
    
    def process(self, image):
        return self.in_process(image)