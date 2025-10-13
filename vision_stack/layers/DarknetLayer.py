from pathlib import Path
import numpy as np
import cv2
import darknet

try:
    from mil_msgs.msg import ObjectDetection, ObjectDetections
except Exception as e:
    print(f"Could not import rclpy or messages because:\n{e}")

from .Layer import AnalysisLayer


class DarknetLayer(AnalysisLayer):
    def __init__(self, conf_thres, weights_file: str, absolute_path_to_weights_directory = "", pass_post_detection_img = False) -> None:
        print(Path.cwd())
        weights_dir = Path(absolute_path_to_weights_directory) if absolute_path_to_weights_directory != "" else Path.cwd() / "vision_stack" / "weights"
        
        # Extract base name for cfg and names files (assumes same base name)
        weights_path = weights_dir / weights_file
        weights_base = weights_path.stem
        
        self.path_to_weights = weights_path
        self.path_to_cfg = weights_dir / f"{weights_base}.cfg"
        self.path_to_names = weights_dir / f"{weights_base}.names"
        
        # Find file for weights and extract name
        self.weights_name = self.path_to_weights.stem
        
        # Pass post processing image
        self.pass_post_detection_img = pass_post_detection_img
        
        # Store parameters
        self.network = darknet.load_net_custom(
            str(self.path_to_cfg).encode("ascii"),
            str(self.path_to_weights).encode("ascii"),
            0, 1
        )
        
        self.labels = self.path_to_names.read_text().splitlines()
        self.colours = darknet.class_colors(self.labels)
        self.conf_thres = conf_thres
        
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        
        self.msg = None
        super().__init__(f"darknet_{self.weights_name}")
    
    def process(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        # Create darknet image
        darknet_image = darknet.make_image(self.width, self.height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        
        # Run detection
        detections = darknet.detect_image(self.network, self.labels, darknet_image, thresh=self.conf_thres)
        darknet.free_image(darknet_image)
        
        unprocessed_image = image
        bbox_image = darknet.draw_boxes(detections, image_resized, self.colours)
        bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
        bbox_image = cv2.resize(bbox_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        object_detections = []
        
        if detections:
            # Scale factors to convert from network size to original image size
            scale_x = image.shape[1] / self.width
            scale_y = image.shape[0] / self.height
            
            for detection in detections:
                class_name, conf, bbox = detection
                x, y, w, h = bbox
                
                # Scale coordinates to original image size
                x = x * scale_x
                y = y * scale_y
                w = w * scale_x
                h = h * scale_y
                
                class_index = self.labels.index(class_name)
                object_detections.append(ObjectDetection(center_x=x, center_y=y, width=w, height=h, confidence=conf, class_index=class_index, class_name=class_name))
        
        self.msg = ObjectDetections(detections=object_detections)
        return (bbox_image if self.pass_post_detection_img else unprocessed_image, object_detections)