from PIL import Image
from abc import ABC, abstractmethod

class Layer(ABC):

    @abstractmethod
    def input_size(self):
        pass

    @abstractmethod
    def output_size(self):
        pass

    @abstractmethod
    def process(self, image):
        pass

class PreprocessLayer(Layer):
    def __init__(self, size) -> None:
        self.out_dim = size
        self.in_dim = size
    
    def input_size(self):
        return self.in_dim
    
    def output_size(self):
        return self.out_dim
    
class AnalysisLayer(Layer):
    def __init__(self, in_size, out_size) -> None:
        self.out_dim = out_size
        self.in_dim = in_size
    
    def input_size(self):
        return self.in_dim
    
    def output_size(self):
        return self.out_dim
    
