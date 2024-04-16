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
    def __init__(self, name) -> None:
        self._name = name
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
    
class AnalysisLayer(Layer):
    def __init__(self, name) -> None:
        self.name = name
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
    
