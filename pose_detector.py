"""
Pose Detection Module using MediaPipe

This module provides a PoseDetector class for real-time human pose detection
and analysis using Google's MediaPipe framework.

Author: [Pradosh P. Dash]
License: MIT
"""

import cv2
import mediapipe as mp
import math
import time
from typing import List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseDetector:
    """
    A class for detecting and analyzing human poses in images and video streams.
    
    This class wraps MediaPipe's pose detection functionality and provides
    additional methods for landmark extraction and angle calculation.
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the PoseDetector.
        
        Args:
            static_image_mode: Whether to treat input as static images
            model_complexity: Complexity of pose model (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            enable_segmentation: Whether to predict segmentation mask
            smooth_segmentation: Whether to smooth segmentation across frames
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe components
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        self.results = None
        self.landmark_list = []
        
    def detect_pose(self, image: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Detect pose landmarks in the given image.
        
        Args:
            image: Input image in BGR format
            draw: Whether to draw pose landmarks on the image
            
        Returns:
            Image with pose landmarks drawn (if draw=True)
        """
        try:
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(image_rgb)
            
            # Draw landmarks if requested and detected
            if self.results.pose_landmarks and draw:
                self.mp_draw.draw_landmarks(
                    image, 
                    self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
            
        return image
    
    def get_landmarks(self, image: cv2.Mat, draw: bool = True) -> List[List[int]]:
        """
        Extract pose landmark positions from the image.
        
        Args:
            image: Input image
            draw: Whether to draw circles at landmark positions
            
        Returns:
            List of landmarks in format [id, x, y]
        """
        self.landmark_list = []
        
        if self.results and self.results.pose_landmarks:
            height, width, channels = image.shape
            
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                self.landmark_list.append([landmark_id, x, y])
                
                # Draw landmark points if requested
                if draw:
                    cv2.circle(image, (x, y), 5, (255, 0, 0), cv2.FILLED)
                    
        return self.landmark_list
    
    def calculate_angle(
        self, 
        image: cv2.Mat, 
        point1: int, 
        point2: int, 
        point3: int, 
        draw: bool = True
    ) -> float:
        """
        Calculate the angle between three pose landmarks.
        
        Args:
            image: Input image
            point1: First landmark ID
            point2: Vertex landmark ID (center point)
            point3: Third landmark ID
            draw: Whether to draw the angle visualization
            
        Returns:
            Angle in degrees
        """
        if len(self.landmark_list) == 0:
            logger.warning("No landmarks available for angle calculation")
            return 0.0
            
        try:
            # Get landmark coordinates
            x1, y1 = self.landmark_list[point1][1:]
            x2, y2 = self.landmark_list[point2][1:]
            x3, y3 = self.landmark_list[point3][1:]
            
            # Calculate angle using atan2
            angle = math.degrees(
                math.atan2(y3 - y2, x3 - x2) - 
                math.atan2(y1 - y2, x1 - x2)
            )
            
            # Normalize angle to 0-360 degrees
            if angle < 0:
                angle += 360
                
            # Draw angle visualization if requested
            if draw:
                self._draw_angle(image, (x1, y1), (x2, y2), (x3, y3), angle)
                
            return angle
            
        except (IndexError, ValueError) as e:
            logger.error(f"Error calculating angle: {e}")
            return 0.0
    
    def _draw_angle(
        self, 
        image: cv2.Mat, 
        point1: Tuple[int, int], 
        point2: Tuple[int, int], 
        point3: Tuple[int, int], 
        angle: float
    ) -> None:
        """
        Draw angle visualization on the image.
        
        Args:
            image: Input image
            point1: First point coordinates
            point2: Vertex point coordinates
            point3: Third point coordinates
            angle: Calculated angle
        """
        # Draw lines
        cv2.line(image, point1, point2, (255, 255, 255), 3)
        cv2.line(image, point3, point2, (255, 255, 255), 3)
        
        # Draw circles at points
        for point in [point1, point2, point3]:
            cv2.circle(image, point, 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, point, 15, (0, 0, 255), 2)
        
        # Display angle text
        cv2.putText(
            image, 
            f"{int(angle)}°", 
            (point2[0] - 50, point2[1] + 50),
            cv2.FONT_HERSHEY_PLAIN, 
            2, 
            (0, 0, 255), 
            2
        )


class FPSCounter:
    """Helper class for calculating and displaying FPS."""
    
    def __init__(self):
        self.prev_time = 0
        
    def update(self) -> float:
        """Update and return current FPS."""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        return fps
    
    def draw_fps(self, image: cv2.Mat, fps: float) -> None:
        """Draw FPS on the image."""
        cv2.putText(
            image, 
            f"FPS: {int(fps)}", 
            (70, 50), 
            cv2.FONT_HERSHEY_PLAIN, 
            3,
            (255, 0, 0), 
            3
        )


def demo_video_analysis(video_path: str) -> None:
    """
    Demonstrate pose detection on a video file.
    
    Args:
        video_path: Path to the video file
    """
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    fps_counter = FPSCounter()
    
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return
    
    logger.info(f"Starting video analysis: {video_path}")
    
    while True:
        success, image = cap.read()
        if not success:
            logger.info("End of video or failed to read frame")
            break
            
        # Detect pose
        image = detector.detect_pose(image)
        landmark_list = detector.get_landmarks(image, draw=False)
        
        # Highlight specific landmark (right wrist - landmark 16)
        if len(landmark_list) > 16:
            x, y = landmark_list[16][1], landmark_list[16][2]
            cv2.circle(image, (x, y), 15, (0, 255, 0), cv2.FILLED)
        
        # Calculate and display FPS
        fps = fps_counter.update()
        fps_counter.draw_fps(image, fps)
        
        # Display image
        cv2.imshow("Pose Detection - Video", image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def demo_webcam() -> None:
    """Demonstrate real-time pose detection using webcam."""
    cap = cv2.VideoCapture(0)  # Use default camera
    detector = PoseDetector()
    fps_counter = FPSCounter()
    
    if not cap.isOpened():
        logger.error("Error opening webcam")
        return
    
    logger.info("Starting webcam pose detection (Press 'q' to quit)")
    
    while True:
        success, image = cap.read()
        if not success:
            logger.error("Failed to read from webcam")
            break
        
        # Detect pose and get landmarks
        image = detector.detect_pose(image)
        landmark_list = detector.get_landmarks(image)
        
        # Example: Calculate elbow angle (landmarks 11, 13, 15 for left arm)
        if len(landmark_list) > 15:
            angle = detector.calculate_angle(image, 11, 13, 15)
            
        # Display FPS
        fps = fps_counter.update()
        fps_counter.draw_fps(image, fps)
        
        # Show image
        cv2.imshow("Pose Detection - Webcam", image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function with options for different demo modes."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "webcam":
            demo_webcam()
        else:
            demo_video_analysis(sys.argv[1])
    else:
        print("Usage:")
        print("  python pose_detector.py webcam        # Use webcam")
        print("  python pose_detector.py <video_path>  # Use video file")
        print("\nRunning webcam demo by default...")
        demo_webcam()


if __name__ == "__main__":
    main()
