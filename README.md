# Pose Detection with MediaPipe

A professional Python implementation for real-time human pose detection and analysis using Google's MediaPipe framework.

## Features

- **Real-time pose detection** from webcam or video files
- **Landmark extraction** with pixel-perfect coordinates
- **Angle calculation** between any three pose landmarks
- **FPS monitoring** for performance tracking
- **Professional code structure** with proper error handling and logging
- **Type hints** for better code maintainability
- **Comprehensive documentation** with docstrings

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV
- MediaPipe

### Install Dependencies

```bash
pip install opencv-python mediapipe
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### Using Webcam

```python
from pose_detector import demo_webcam

# Start real-time pose detection
demo_webcam()
```

### Using Video File

```python
from pose_detector import demo_video_analysis

# Analyze poses in a video file
demo_video_analysis("path/to/your/video.mp4")
```

### Command Line Usage

```bash
# Use webcam
python pose_detector.py webcam

# Use video file
python pose_detector.py path/to/video.mp4
```

## Usage Examples

### Basic Pose Detection

```python
import cv2
from pose_detector import PoseDetector

# Initialize detector
detector = PoseDetector()

# Load image
image = cv2.imread("person.jpg")

# Detect pose
result_image = detector.detect_pose(image, draw=True)

# Get landmark coordinates
landmarks = detector.get_landmarks(image)
print(f"Detected {len(landmarks)} landmarks")
```

### Calculate Joint Angles

```python
import cv2
from pose_detector import PoseDetector

detector = PoseDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect pose
    frame = detector.detect_pose(frame)
    landmarks = detector.get_landmarks(frame)
    
    # Calculate elbow angle (shoulder-elbow-wrist)
    if len(landmarks) > 15:
        elbow_angle = detector.calculate_angle(frame, 11, 13, 15)
        print(f"Left elbow angle: {elbow_angle:.1f}°")
    
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## MediaPipe Pose Landmarks

The pose model detects 33 landmarks:

| ID | Landmark Name | ID | Landmark Name |
|----|---------------|----|---------------|
| 0  | nose | 17 | left_pinky |
| 1  | left_eye_inner | 18 | right_pinky |
| 2  | left_eye | 19 | left_index |
| 3  | left_eye_outer | 20 | right_index |
| 4  | right_eye_inner | 21 | left_thumb |
| 5  | right_eye | 22 | right_thumb |
| 6  | right_eye_outer | 23 | left_hip |
| 7  | left_ear | 24 | right_hip |
| 8  | right_ear | 25 | left_knee |
| 9  | mouth_left | 26 | right_knee |
| 10 | mouth_right | 27 | left_ankle |
| 11 | left_shoulder | 28 | right_ankle |
| 12 | right_shoulder | 29 | left_heel |
| 13 | left_elbow | 30 | right_heel |
| 14 | right_elbow | 31 | left_foot_index |
| 15 | left_wrist | 32 | right_foot_index |
| 16 | right_wrist |

## API Reference

### PoseDetector Class

#### Constructor

```python
PoseDetector(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

**Parameters:**
- `static_image_mode`: Whether to treat input as static images
- `model_complexity`: Complexity of pose model (0=lite, 1=full, 2=heavy)
- `smooth_landmarks`: Whether to smooth landmarks across frames
- `enable_segmentation`: Whether to predict segmentation mask
- `smooth_segmentation`: Whether to smooth segmentation across frames
- `min_detection_confidence`: Minimum confidence for pose detection (0.0-1.0)
- `min_tracking_confidence`: Minimum confidence for pose tracking (0.0-1.0)

#### Methods

##### `detect_pose(image, draw=True)`
Detect pose landmarks in the given image.

**Parameters:**
- `image`: Input image in BGR format
- `draw`: Whether to draw pose landmarks on the image

**Returns:** Image with pose landmarks drawn (if draw=True)

##### `get_landmarks(image, draw=True)`
Extract pose landmark positions from the image.

**Parameters:**
- `image`: Input image
- `draw`: Whether to draw circles at landmark positions

**Returns:** List of landmarks in format [id, x, y]

##### `calculate_angle(image, point1, point2, point3, draw=True)`
Calculate the angle between three pose landmarks.

**Parameters:**
- `image`: Input image
- `point1`: First landmark ID
- `point2`: Vertex landmark ID (center point)
- `point3`: Third landmark ID
- `draw`: Whether to draw the angle visualization

**Returns:** Angle in degrees (0-360)

## Performance Tips

1. **Reduce model complexity** for faster processing on slower devices
2. **Disable landmark smoothing** for static images
3. **Turn off drawing** when only landmark coordinates are needed
4. **Use appropriate confidence thresholds** based on your use case

## Common Use Cases

- **Fitness applications**: Monitor exercise form and count repetitions
- **Physical therapy**: Track range of motion and rehabilitation progress
- **Sports analysis**: Analyze athlete performance and technique
- **Gaming**: Create motion-controlled games and interfaces
- **Security**: Monitor for specific poses or gestures

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google's MediaPipe team for the pose detection model
- OpenCV community for computer vision tools
- Contributors and users of this project

## Troubleshooting

### Common Issues

**ImportError: No module named 'mediapipe'**
```bash
pip install mediapipe
```

**Camera not working**
- Check if camera is being used by another application
- Try different camera indices (0, 1, 2, etc.)
- Verify camera permissions

**Low FPS performance**
- Reduce model complexity
- Lower input image resolution
- Disable unnecessary drawing operations

## Support

If you encounter any problems or have questions, please:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information about your problem
