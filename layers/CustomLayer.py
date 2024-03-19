from Layer import AnalysisLayer
import numpy as np

class CustomLayer(AnalysisLayer):
    def __init__(self, in_size, out_size, name, process:callable, *args) -> None:
        image_size = (40, 40)
        # Create a random image array
        random_image = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
        self.args = args
        super().__init__(in_size, out_size, name)

        if callable(process):
            try:
                output = process(random_image, args)
                assert isinstance(output[0], np.ndarray)
                assert len(output) == 2
                self.process = lambda img: process(img, self.args)
            except Exception as e:
                raise Exception(f"The return value of the process is invalid, must be tuple with:\ntuple[0] -> 'numpy.ndarray'\ntuple[1] -> Some value (Default to None)\nException:{e}")