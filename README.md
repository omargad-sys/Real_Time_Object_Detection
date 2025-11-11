# Real-Time Object Detection with YOLOv8 and OpenCV

This project uses a YOLOv8 model with OpenCV in Python to perform real-time object detection through a webcam. The program recognizes common objects such as pens, bottles, laptops, or people and outlines them on the video feed with labeled boxes.

## Features

*   **Real-time object detection** using YOLOv8
*   **GPU acceleration** (automatically uses CUDA if available)
*   **Object tracking** with persistent IDs across frames
*   **FPS counter** and live detection statistics
*   **Class-specific color coding** for bounding boxes
*   **Pause/Resume** functionality (SPACE key)
*   **Screenshot capture** (S key)
*   **Frame skipping** for performance optimization
*   **Window scaling** for different display sizes
*   **Configuration file** support for easy setup
*   **Automatic webcam error recovery**
*   **Thread-safe frame processing**
*   **Comprehensive logging**

## How It Works

*   The webcam feed is captured using OpenCV in a separate thread for optimal performance.
*   Each frame is passed to the YOLOv8 model for object detection.
*   Objects are tracked across frames using IoU (Intersection over Union) matching.
*   The model predicts object locations and labels with confidence scores.
*   Results are drawn onto the live video feed with color-coded bounding boxes and tracking IDs.
*   An info panel displays FPS, GPU/CPU mode, detection counts, and keyboard controls.

When you run the script, the program opens a camera window and automatically labels what it detects in real time.

**Keyboard Controls:**
*   Press `q` to quit
*   Press `SPACE` to pause/resume
*   Press `s` to save a screenshot

## Tech Stack

*   Python 3.8+
*   Ultralytics YOLOv8 (pretrained yolov8m.pt model)
*   OpenCV for image and video handling
*   PyTorch for GPU acceleration
*   NumPy for numerical operations

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the object detection script from the command line:

```bash
python main.py [OPTIONS]
```

### Options

*   `--device`: Specify the webcam device ID. Default is `0`.
*   `--confidence-threshold`: Set the minimum confidence threshold for displaying detections. Default is `0.5`.
*   `--output-file`: Specify a file name to save the output video (e.g., `output.mp4`).
*   `--classes`: Specify a list of class IDs to detect (e.g., `0` for person, `41` for cup). Default is to detect all classes.
*   `--frame-skip`: Process every N frames (0 = process all frames). Useful for performance on slower systems. Default is `0`.
*   `--window-scale`: Scale factor for the display window (e.g., `0.5` for half size, `2.0` for double size). Default is `1.0`.
*   `--config`: Path to a JSON configuration file. Command-line arguments override config file values.
*   `--model`: Path to YOLO model file. Default is `yolov8m.pt`.

### Examples

*   To run with the default settings:
    ```bash
    python main.py
    ```

*   To use a different webcam (e.g., device ID 1):
    ```bash
    python main.py --device 1
    ```

*   To set a higher confidence threshold (e.g., 0.7):
    ```bash
    python main.py --confidence-threshold 0.7
    ```

*   To save the output to a file named `output.mp4`:
    ```bash
    python main.py --output-file output.mp4
    ```

*   To detect only persons (class ID 0) and bottles (class ID 39):
    ```bash
    python main.py --classes 0 39
    ```

*   To use frame skipping for better performance (process every 2nd frame):
    ```bash
    python main.py --frame-skip 1
    ```

*   To scale the display window to half size:
    ```bash
    python main.py --window-scale 0.5
    ```

*   To use a configuration file:
    ```bash
    python main.py --config config.json
    ```

### Configuration File

You can create a JSON configuration file to set default options. Example `config.json`:

```json
{
  "device": 0,
  "confidence_threshold": 0.6,
  "frame_skip": 1,
  "window_scale": 0.75,
  "classes": [0, 39, 41],
  "model": "yolov8m.pt"
}
```

Command-line arguments will override values from the configuration file.

## Testing

Run the unit tests with:

```bash
python -m unittest test_main.py
```

Or for verbose output:

```bash
python -m unittest test_main.py -v
```

## Performance Tips

*   **GPU Acceleration**: The program automatically detects and uses CUDA if available. Check the info panel to see if GPU mode is active.
*   **Frame Skipping**: Use `--frame-skip 1` or higher on slower systems to process fewer frames and improve performance.
*   **Window Scaling**: Use `--window-scale 0.5` to reduce the display window size if you experience lag.
*   **Class Filtering**: Use `--classes` to detect only specific objects, which can improve performance.

## Troubleshooting

*   **Webcam not opening**: Make sure no other application is using the webcam. Try different device IDs with `--device 1`, `--device 2`, etc.
*   **Low FPS**: Enable frame skipping with `--frame-skip 1` or reduce window scale with `--window-scale 0.5`.
*   **Model file not found**: Ensure `yolov8m.pt` is in the same directory as `main.py`, or specify the path with `--model`.
*   **CUDA errors**: If you have GPU issues, the program will automatically fall back to CPU mode.
