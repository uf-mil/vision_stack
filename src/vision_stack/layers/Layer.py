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

    @property
    @abstractmethod
    def name(self):
        pass

class PreprocessLayer(Layer):
    def __init__(self, size, name) -> None:
        self.out_dim = size
        self.in_dim = size
        self._name = name
    
    def input_size(self):
        return self.in_dim
    
    def output_size(self):
        return self.out_dim
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
    
class AnalysisLayer(Layer):
    def __init__(self, in_size, out_size, name) -> None:
        self.out_dim = out_size
        self.in_dim = in_size
        self.name = name
    
    def input_size(self):
        return self.in_dim
    
    def output_size(self):
        return self.out_dim
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
    
