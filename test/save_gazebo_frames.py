#!/usr/bin/env python3
"""
Simple script to save frames from Gazebo camera feed.
This lets you test Detectron2 on actual sim images before doing real-time processing.

Usage:
    python3 save_gazebo_frames.py /camera/topic/name
    python3 save_gazebo_frames.py /front_camera/image_raw --num-frames 10
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import argparse
from pathlib import Path
import time


class FrameSaver(Node):
    """Simple node to save frames from a camera topic."""
    
    def __init__(self, topic_name, num_frames=10, output_dir='gazebo_frames'):
        super().__init__('frame_saver')
        
        self.bridge = CvBridge()
        self.num_frames = num_frames
        self.frames_saved = 0
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.get_logger().info(f'Saving {num_frames} frames to {output_dir}/')
        self.get_logger().info(f'Subscribing to: {topic_name}')
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10
        )
    
    def image_callback(self, msg):
        """Save incoming images."""
        if self.frames_saved >= self.num_frames:
            self.get_logger().info('Target number of frames saved. Shutting down...')
            rclpy.shutdown()
            return
        
        try:
            # Convert ROS Image message to OpenCV image
            # Try 'bgr8' first (most common), fall back to 'rgb8'
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # Save frame
            self.frames_saved += 1
            filename = self.output_dir / f'frame_{self.frames_saved:04d}.jpg'
            cv2.imwrite(str(filename), cv_image)
            
            self.get_logger().info(
                f'Saved frame {self.frames_saved}/{self.num_frames}: {filename}'
            )
            
            # Small delay to avoid saving duplicate frames
            time.sleep(0.5)
            
        except Exception as e:
            self.get_logger().error(f'Error saving frame: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Save frames from Gazebo camera topic'
    )
    parser.add_argument('topic', type=str, 
                       help='Camera topic name (e.g., /front_camera/image_raw)')
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Number of frames to save (default: 10)')
    parser.add_argument('--output-dir', type=str, default='gazebo_frames',
                       help='Output directory (default: gazebo_frames)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        node = FrameSaver(
            topic_name=args.topic,
            num_frames=args.num_frames,
            output_dir=args.output_dir
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()