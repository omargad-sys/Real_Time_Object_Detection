# Real-Time Object Detection with YOLOv8 and OpenCV

This project uses a YOLOv8 model with OpenCV in Python to perform real-time object detection through a webcam. The program recognizes common objects — like pens, bottles, laptops, or people — and outlines them on the video feed with labeled boxes.

## How It Works

*   The webcam feed is captured using OpenCV.
*   Each frame is passed to the YOLOv8 model for object detection.
*   The model predicts object locations and labels.
*   The results are drawn directly onto the live video feed.

When you run the script, the program opens a camera window and automatically labels what it detects in real time.

Press “q” to quit the window.

## Tech Stack

*   Python 3.8+
*   Ultralytics YOLOv8 (pretrained yolov8n.pt model)
*   OpenCV for image and video handling
