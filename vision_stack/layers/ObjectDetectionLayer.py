import os
import numpy as np

try:
    from mil_msgs.msg import ObjectDetection, ObjectDetections
except Exception as e:
    print(f"Could not import rclpy or messages because:\n{e}")

from PIL import Image
from ultralytics import YOLO

from .Layer import AnalysisLayer

class ObjectDetectionLayer(AnalysisLayer):
    def __init__(self, conf_thres, weights_file:str, absolute_path_to_weights_directory = "", pass_post_detection_img = False) -> None:
        print(os.getcwd())
        absolute_path_to_weights_directory = absolute_path_to_weights_directory if absolute_path_to_weights_directory != "" else os.path.join(os.getcwd(), "vision_stack", "weights")
        self.path_to_weights = os.path.join(absolute_path_to_weights_directory, weights_file)
        # Find file for weights and extract name
        self.weights_name = os.path.splitext(os.path.basename(self.path_to_weights))[0] 

        # Pass post processing image
        self.pass_post_detection_img = pass_post_detection_img

        # Store parameters
        self.model = YOLO(self.path_to_weights)
        self.labels = self.model.names
        self.conf_thres = conf_thres
        self.msg = None

        super().__init__(f"objectDetection_{self.weights_name}")
    
    def process(self, image):
        img = Image.fromarray(image)
        results = self.model.predict(img, conf=self.conf_thres)
        detections = results[0].boxes

        unprocessed_image = image
        bbox_image = results[0].plot()

        object_detections = []

        if detections:
            for bbox in detections:
                x, y, w, h = bbox.xywh.numpy()[0]
                conf = bbox.conf.item()
                class_index = bbox.cls.item()
                class_name = self.labels[class_index]
                object_detections.append(ObjectDetection(center_x=x, center_y=y, width=w, height=h, confidence=conf, class_index=class_index, class_name=class_name))
        
        self.msg = ObjectDetections(detections=object_detections)

        return (bbox_image if self.pass_post_detection_img else unprocessed_image, object_detections)
