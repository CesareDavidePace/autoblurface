# AutoBlurFace
## Description
AutoFaceBlur is a Python-based tool designed to enhance privacy and anonymity in video content. Utilizing the advanced capabilities of the YOLO  model, AutoFaceBlur efficiently detects faces in real-time within video streams and applies a seamless blurring effect. 

![alt text](https://github.com/[cesaredavidepace]/[autoblurface]/face.jpg?raw=true)

## Features
- **Real-time face detection**: Leveraging the YOLO model for accurate and fast detection.
- **Face blurring**: Enhances privacy by blurring detected faces.
- **Compatibility**: Works with video files and webcam streams.
- **Adjustable blurring intensity**: Customize the intensity of the blur effect.
- **Output saving**: Option to save the processed video output.

## Installation

### Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/cesaredavidepace/autoblurface.git
   ```
2. **Navigate to the cloned directory**:
   ```bash
   cd AutoBlurFace
   ```
3. **Install required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download models**
   [YoloV8 Face](https://github.com/akanametov/yolov8-face)

## Basic Usage 
```python
from AutoBlurFace import AutoBlurFace

auto_blur_face = AutoBlurFace(
    model_path='yolov8n-face.pt',
    output_video_path='output.mp4',
    show_video=True,
    save_video=True,
    apply_blur=True
)

auto_blur_face.run('face-video.mp4')
```

## Using Webcam
```python
auto_blur_face.run(use_webcam=True)
```

## Acknowledgments
- YOLO and Ultralytics for the face detection model.
- OpenCV community for the image processing tools.


