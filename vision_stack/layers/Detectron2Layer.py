# TODO:
# - implement _setup_config to build a detectron2 cfg
# - implement _lazy_init_predictor to create DefaultPredictor lazily
# - implement process to run inference and return detections + optional visualization
# - implement extraction/visualization helpers

from typing import Optional, List, Dict, Tuple
import numpy as np

from .Layer import AnalysisLayer


class Detectron2Layer(AnalysisLayer):
    def __init__(self, name: str = "Detectron2", model_name: str = "faster_rcnn_R_50_FPN", conf_threshold: float = 0.5, device: Optional[str] = None) -> None:
        self._name = name
        self._model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self.cfg = None
        self.predictor = None
        self.metadata = None

    @property
    def name(self) -> str:
        return self._name

    def _setup_config(self):
        raise NotImplementedError

    def _lazy_init_predictor(self):
        raise NotImplementedError

    def process(self, image: np.ndarray, visualize: Optional[bool] = None, save_visualized_path: Optional[str] = None) -> Tuple[List[Dict], Optional[np.ndarray], Optional[object], float]:
        raise NotImplementedError

    def extract_detections(self, outputs) -> List[Dict]:
        raise NotImplementedError

    def visualize_predictions(self, image: np.ndarray, outputs) -> np.ndarray:
        raise NotImplementedError

    def set_model(self, model_name: str, weights: Optional[str] = None, conf_threshold: Optional[float] = None) -> None:
        self._model_name = model_name
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        self.cfg = None
        self.predictor = None
        self.metadata = None

    def close(self) -> None:
        self.predictor = None

    def __repr__(self) -> str:
        return f"<Detectron2Layer name={self._name} model={self._model_name}>"
