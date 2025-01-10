#!/usr/bin/env python

import cv2

import os

import numpy as np
import torch
import torch.nn as nn

try:
    import rclpy
    from vision_stack.msg import ObjectDetection, ObjectDetections
except Exception as e:
    print(f"Could not import rclpy or messages because:\n{e}")

try:
    import tensorflow as tf
except Exception as e:
    print("Tensorflow could not be imported...")

from PIL import Image
from torchvision import transforms

from .helpers.ml_funcs import attempt_load, plot_one_box, non_max_suppression

from .Layer import AnalysisLayer

class ObjectDetectionLayer(AnalysisLayer):
    def __init__(self, path_to_weights, conf_thres, iou_thres, class_names_array = [], colors_array = [], pass_post_processing_img = False) -> None:

        # Find file for weights and extract name
        _, file_type = os.path.splitext(path_to_weights)
        self.weights_name = os.path.splitext(os.path.basename(path_to_weights))[0]

        # Check if the number of colors provided matches the number of classes
        if len(colors_array) != len(class_names_array):
            raise Exception(f"ERROR -> ObjectDetectionLayer -> __init__: colors_array must be same length as class_names_array ({len(colors_array)} != {len(class_names_array)})")

        self.class_names_array = class_names_array
        self.colors_array = colors_array

        # Choose a processor based on the provided weight's file type
        self.processor = self.PTWeightsProcessor(path_to_weights, conf_thres, iou_thres, class_names_array, colors_array, pass_post_processing_img) if file_type == ".pt" else self.TFLiteWeightsProcessor(path_to_weights, conf_thres, class_names_array, colors_array, pass_post_processing_img) if file_type == ".tflite" else None

        if not self.processor:
            raise Exception("ERROR -> ObjectDetectionLayer -> __init__: Invalid file type, weights file supported are *.pt or *.tflite") 

        # Store parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.msg = None

        super().__init__(f"objectDetection_{self.weights_name}")
    
    def process(self, image):
        img, detections = self.processor.process(image)
        object_detections = []
        for detection in detections:
            object_detections.append(ObjectDetection(center_x=detection[0], center_y=detection[1], width=detection[2], height=detection[3], confidence=detection[4], class_index=detection[5], class_name=self.class_names_array[detection[5]]))
        self.msg = ObjectDetections(detections=object_detections)
        return (img,detections)
    
    class PTWeightsProcessor():
        def __init__(self, weights_path, conf_thres, iou_thres, classes, colors, input_shape = ((960, 608)), pass_post_processing_img = False) -> None:
            self.weights_path = weights_path
            self.pass_post_processing_img = pass_post_processing_img
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            self.conf_thres = conf_thres
            self.iou_thres = iou_thres

            self.classes = classes
            self.colors = colors
            self.model_input_dims = input_shape

            absolute_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), self.weights_path),
            )

            self.__MODEL = attempt_load(
                absolute_file_path,
                map_location=self.device,
            )

        def process(self, img):
            image = Image.fromarray(img)
            img_transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = img_transform(image).to(self.device).unsqueeze(0)

            print(img_tensor.size())

            with torch.inference_mode():
                pred_results = self.__MODEL(img_tensor)[0]

            detections = non_max_suppression(
                pred_results,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )

            unprocessed_image = img.copy()

            processed_detections = []

            if detections:
                detections = detections[0]
                print(detections)
                for x1, y1, x2, y2, conf, cls in detections:
                    class_index = int(cls.cpu().item())
                    print(f"{self.classes[class_index]} => {conf}")
                    plot_one_box(
                        [x1, y1, x2, y2],
                        img,
                        label=f"{self.classes[class_index]}",
                        color=self.colors[class_index],
                        line_thickness=2,
                    )
                    processed_detections.append(self.get_center_and_dims(x1, y1, x2, y2) + [conf, class_index])

            return (img if self.pass_post_processing_img else img, processed_detections)
        
        def get_center_and_dims(self, x1, y1, x2, y2):
            center_x = ( x2 - x1 ) / 2 + x1
            center_y = ( y2 - y1 ) / 2 + y1
            w = x2 - x1
            h = y2 - y1
            return [center_x, center_y, w, h]
    
    class TFLiteWeightsProcessor():
        def __init__(self, weights_path, conf_thres, classes, colors, pass_post_processing_img = False) -> None:
            self.weights_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), weights_path),
            )
            self.conf_thres = conf_thres
            self.classes = classes
            self.colors = colors
            self.pass_post_processing_img = pass_post_processing_img
            self.interpreter = tf.lite.Interpreter(model_path=self.weights_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_dims = (self.input_details[0]["shape"][-1], self.input_details[0]["shape"][-2])

        def process(self, img):
            COLORS = self.colors
            CLASSES = self.classes
            CONFIDENCE_THRESHOLD = self.conf_thres

            # Prepare input data (replace with your input data)
            input_data = Image.fromarray(img)
            unprocessed_image = img.copy()
            img_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            input_data = img_transform(input_data).unsqueeze(0)


            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)


            # Run inference
            self.interpreter.invoke()


            # Get output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            processed_detections = []


            # Process output (interpret predictions)
            # Replace this with your post-processing logic
            if len(output_data) != 0:
                detections = output_data[0]
                for x1, y1, w, h, conf, cls1, cls2, cls3, cls4, cls5, cls6 in detections:
                    class_index = np.argmax([cls1,cls2,cls3,cls4,cls5,cls6])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    else:
                        pass
                        print(f'{CLASSES[class_index]} => {conf}')
                        print(f'Center: ({x1, y1})')


                    x1, y1, x2, y2 = self.__bounding_box_coordinates(x1, y1, w, h)
                    processed_detections.append([x1, y1, w, h, conf, class_index])
                    plot_one_box([x1, y1, x2, y2], img, label=f'{CLASSES[class_index]}', color=COLORS[class_index], line_thickness=2)
            else:
                print("No Detections Made")

            return (img if self.pass_post_processing_img else unprocessed_image, processed_detections)
        
        def __bounding_box_coordinates(self, center_x, center_y, width, height):
                half_width = width / 2
                half_height = height / 2
                x1 = center_x - half_width
                y1 = center_y - half_height
                x2 = center_x + half_width
                y2 = center_y + half_height
                return x1, y1, x2, y2
