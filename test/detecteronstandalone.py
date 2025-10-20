#!/usr/bin/env python3
"""
Standalone Detectron2 testing script.
Tests different pre-trained models and evaluates their performance.

Usage:
    python3 standalone_detectron2_test.py <path_to_image>
    python3 standalone_detectron2_test.py --webcam
    python3 standalone_detectron2_test.py --video <path_to_video>
"""

import cv2
import numpy as np
import torch
import time
import argparse
from pathlib import Path

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class Detectron2Tester:
    """Class to test and benchmark different Detectron2 models."""
    
    AVAILABLE_MODELS = {
        'faster_rcnn_R_50_FPN': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'faster_rcnn_R_101_FPN': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
        'faster_rcnn_X_101_FPN': 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        'mask_rcnn_R_50_FPN': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'mask_rcnn_R_101_FPN': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'retinanet_R_50_FPN': 'COCO-Detection/retinanet_R_50_FPN_3x.yaml',
        'retinanet_R_101_FPN': 'COCO-Detection/retinanet_R_101_FPN_3x.yaml',
    }
    
    def __init__(self, model_name='faster_rcnn_R_50_FPN', conf_threshold=0.5):
        """
        Initialize the Detectron2 tester.
        
        Args:
            model_name: Name of the model to use (see AVAILABLE_MODELS)
            conf_threshold: Confidence threshold for detections (0-1)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.cfg = self._setup_config(model_name, conf_threshold)
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TRAIN[0] if len(self.cfg.DATASETS.TRAIN) > 0 
            else "coco_2017_train"
        )
        
        print(f"\nModel initialized: {model_name}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Available classes: {len(self.metadata.thing_classes)}")
    
    def _setup_config(self, model_name, conf_threshold):
        """Setup Detectron2 configuration."""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(self.AVAILABLE_MODELS[model_name])
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            self.AVAILABLE_MODELS[model_name]
        )
        cfg.MODEL.DEVICE = self.device
        
        return cfg
    
    def predict_image(self, image):
        """
        Run prediction on a single image.
        
        Args:
            image: numpy array (BGR format from cv2)
        
        Returns:
            outputs: Detectron2 outputs dictionary
            inference_time: Time taken for inference (seconds)
        """
        start_time = time.time()
        outputs = self.predictor(image)
        inference_time = time.time() - start_time
        
        return outputs, inference_time
    
    def extract_detections(self, outputs):
        """
        Extract detection information from Detectron2 outputs.
        
        Returns:
            list of dicts with detection info
        """
        instances = outputs["instances"].to("cpu")
        
        detections = []
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center_x': float((x1 + x2) / 2),
                    'center_y': float((y1 + y2) / 2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'confidence': float(score),
                    'class_id': int(cls),
                    'class_name': self.metadata.thing_classes[int(cls)]
                }
                detections.append(detection)
        
        return detections
    
    def visualize_predictions(self, image, outputs):
        """
        Create visualization of predictions.
        
        Args:
            image: Original image (BGR)
            outputs: Detectron2 outputs
        
        Returns:
            Visualized image (BGR)
        """
        v = Visualizer(
            image[:, :, ::-1],  
            self.metadata,
            scale=1.0
        )
        vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return vis_output.get_image()[:, :, ::-1] 
    
    def test_single_image(self, image_path, save_output=True):
        """Test on a single image."""
        print(f"\n{'='*60}")
        print(f"Testing on image: {image_path}")
        print(f"{'='*60}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        outputs, inference_time = self.predict_image(image)
        detections = self.extract_detections(outputs)
        
        print(f"\nInference time: {inference_time*1000:.2f} ms")
        print(f"FPS: {1/inference_time:.2f}")
        print(f"Detections found: {len(detections)}")
        
        if detections:
            print("\nDetected objects:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']}: "
                      f"{det['confidence']:.2f} confidence, "
                      f"bbox: ({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                      f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})")
        
        vis_image = self.visualize_predictions(image, outputs)
        
        cv2.imshow('Detectron2 Predictions', vis_image)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if save_output:
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{self.model_name}_{Path(image_path).name}"
            cv2.imwrite(str(output_path), vis_image)
            print(f"Saved output to: {output_path}")
        
        return detections, inference_time
    
    def test_webcam(self):
        """Test on webcam feed."""
        print("\n{'='*60}")
        print("Testing on webcam")
        print("Press 'q' to quit")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        fps_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            outputs, inference_time = self.predict_image(frame)
            detections = self.extract_detections(outputs)
            
            fps = 1.0 / inference_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            vis_frame = self.visualize_predictions(frame, outputs)
            
            cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Detectron2 Webcam', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def test_video(self, video_path, save_output=True):
        """Test on video file."""
        print(f"\n{'='*60}")
        print(f"Testing on video: {video_path}")
        print("Press 'q' to quit")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        if save_output:
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{self.model_name}_{Path(video_path).name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        inference_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            outputs, inference_time = self.predict_image(frame)
            detections = self.extract_detections(outputs)
            inference_times.append(inference_time)
            
            vis_frame = self.visualize_predictions(frame, outputs)
            
            avg_inference = np.mean(inference_times[-30:])
            cv2.putText(vis_frame, 
                       f"Frame: {frame_count}/{total_frames}", 
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, 
                       f"Inference: {avg_inference*1000:.1f}ms", 
                       (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, 
                       f"Detections: {len(detections)}", 
                       (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Detectron2 Video', vis_frame)
            
            if save_output:
                out.write(vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_output:
            out.release()
            print(f"\nSaved output to: {output_path}")
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average inference time: {np.mean(inference_times)*1000:.2f} ms")
        print(f"  Average FPS: {1/np.mean(inference_times):.2f}")
        print(f"{'='*60}")


def benchmark_models(image_path, models_to_test=None, num_runs=10):
    """
    Benchmark multiple models on the same image.
    
    Args:
        image_path: Path to test image
        models_to_test: List of model names (None = test all)
        num_runs: Number of runs for averaging
    """
    if models_to_test is None:
        models_to_test = [
            'faster_rcnn_R_50_FPN',
            'faster_rcnn_R_101_FPN',
            'mask_rcnn_R_50_FPN',
            'retinanet_R_50_FPN',
        ]
    
    print(f"\n{'='*60}")
    print("BENCHMARK MODE")
    print(f"Image: {image_path}")
    print(f"Models to test: {len(models_to_test)}")
    print(f"Runs per model: {num_runs}")
    print(f"{'='*60}\n")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    results = []
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            tester = Detectron2Tester(model_name=model_name)
            
            tester.predict_image(image)
            
            times = []
            detection_counts = []
            
            for i in range(num_runs):
                outputs, inference_time = tester.predict_image(image)
                detections = tester.extract_detections(outputs)
                times.append(inference_time)
                detection_counts.append(len(detections))
                print(f"  Run {i+1}/{num_runs}: {inference_time*1000:.2f}ms, "
                      f"{len(detections)} detections")
            
            result = {
                'model': model_name,
                'avg_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000,
                'fps': 1.0 / np.mean(times),
                'avg_detections': np.mean(detection_counts)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Avg Time (ms)':<15} {'Std (ms)':<12} {'FPS':<10} {'Detections'}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['model']:<30} "
              f"{r['avg_time_ms']:<15.2f} "
              f"{r['std_time_ms']:<12.2f} "
              f"{r['fps']:<10.2f} "
              f"{r['avg_detections']:.1f}")
    
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test Detectron2 models standalone'
    )
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--model', type=str, 
                       default='faster_rcnn_R_50_FPN',
                       help='Model to use (default: faster_rcnn_R_50_FPN)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark on multiple models')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable models:")
        for model in Detectron2Tester.AVAILABLE_MODELS.keys():
            print(f"  - {model}")
        return
    
    if args.benchmark:
        if not args.image:
            print("Error: --benchmark requires --image")
            return
        benchmark_models(args.image)
        return
    
    tester = Detectron2Tester(
        model_name=args.model,
        conf_threshold=args.conf_threshold
    )
    
    if args.webcam:
        tester.test_webcam()
    elif args.video:
        tester.test_video(args.video)
    elif args.image:
        tester.test_single_image(args.image)
    else:
        print("Error: Must specify --image, --video, or --webcam")
        parser.print_help()


if __name__ == '__main__':
    main()