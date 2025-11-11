import cv2
import argparse
import threading
import logging
import time
import os
import json
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, deque
from ultralytics import YOLO
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectTracker:
    """Simple object tracker using IoU (Intersection over Union) matching."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracked_objects: Dict[int, Tuple[int, int, int, int]] = {}
        self.max_age = 30  # frames to keep object without detection
        self.object_ages: Dict[int, int] = {}

    def _calculate_iou(self, box1: Tuple[int, int, int, int],
                       box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[int]:
        """Update tracker with new detections and return object IDs."""
        # Match detections to tracked objects
        matched_ids = []
        unmatched_detections = list(range(len(detections)))

        for obj_id, tracked_box in list(self.tracked_objects.items()):
            best_match_idx = -1
            best_iou = self.iou_threshold

            for det_idx in unmatched_detections:
                iou = self._calculate_iou(tracked_box, detections[det_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = det_idx

            if best_match_idx >= 0:
                # Update tracked object
                self.tracked_objects[obj_id] = detections[best_match_idx]
                self.object_ages[obj_id] = 0
                matched_ids.append(obj_id)
                unmatched_detections.remove(best_match_idx)
            else:
                # Increment age for unmatched tracked object
                self.object_ages[obj_id] = self.object_ages.get(obj_id, 0) + 1
                matched_ids.append(-1)  # Placeholder for unmatched

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.tracked_objects[self.next_id] = detections[det_idx]
            self.object_ages[self.next_id] = 0
            matched_ids.append(self.next_id)
            self.next_id += 1

        # Remove old tracks
        for obj_id in list(self.tracked_objects.keys()):
            if self.object_ages.get(obj_id, 0) > self.max_age:
                del self.tracked_objects[obj_id]
                del self.object_ages[obj_id]

        return matched_ids


class ObjectDetector:
    def __init__(self, model_path: str, device_id: int, confidence_threshold: float,
                 output_file: Optional[str], classes_to_detect: List[int],
                 frame_skip: int = 0, window_scale: float = 1.0, config_file: Optional[str] = None):
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)

        # GPU acceleration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        self.device_id = device_id
        self.confidence_threshold = confidence_threshold
        self.output_file = output_file
        self.classes_to_detect = classes_to_detect
        self.frame_skip = frame_skip
        self.window_scale = window_scale

        # Thread safety
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False
        self.video_writer: Optional[cv2.VideoWriter] = None

        # FPS counter
        self.fps_history: deque = deque(maxlen=30)
        self.last_time = time.time()

        # Object tracking
        self.tracker = ObjectTracker()

        # Detection statistics
        self.detection_counts: Dict[str, int] = defaultdict(int)

        # Class colors (generate consistent colors for each class)
        np.random.seed(42)
        self.class_colors: Dict[int, Tuple[int, int, int]] = {}
        for i in range(80):  # COCO has 80 classes
            self.class_colors[i] = tuple(np.random.randint(0, 255, 3).tolist())

        # Screenshot counter
        self.screenshot_count = 0

        # Frame counter for skipping
        self.frame_counter = 0

    def _read_frames(self):
        """Background thread for reading frames from webcam."""
        retry_count = 0
        max_retries = 5

        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
                retry_count = 0
            else:
                # Webcam error recovery
                retry_count += 1
                logger.warning(f"Failed to read frame. Retry {retry_count}/{max_retries}")

                if retry_count >= max_retries:
                    logger.error("Max retries reached. Webcam connection lost.")
                    self.is_running = False
                    break

                time.sleep(0.5)
                # Try to reconnect
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.device_id)

    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if current_time > self.last_time else 0
        self.last_time = current_time
        self.fps_history.append(fps)
        return sum(self.fps_history) / len(self.fps_history)

    def _draw_info_panel(self, frame: np.ndarray, fps: float):
        """Draw information panel with FPS and detection counts."""
        panel_height = 60
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        # FPS and Device on same line
        cv2.putText(panel, f'FPS: {fps:.1f}', (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        device_text = f'Device: {self.device.upper()}'
        cv2.putText(panel, device_text, (150, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Detection counts and controls on same line
        y_offset = 45
        if self.detection_counts:
            counts_text = ', '.join([f'{k}: {v}' for k, v in sorted(self.detection_counts.items())])
            cv2.putText(panel, counts_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Combine panel with frame
        return np.vstack([panel, frame])

    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot."""
        self.screenshot_count += 1
        filename = f'screenshot_{self.screenshot_count:04d}.png'
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")

    def run_detection(self):
        """Main detection loop."""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logger.error(f"Could not open webcam on device {self.device_id}.")
            return

        logger.info(f"Webcam opened successfully on device {self.device_id}")

        # Video writer setup with error handling
        if self.output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Default FPS if unable to detect
                logger.warning(f"Unable to detect FPS, using default: {fps}")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Adjust for window scaling and info panel
            scaled_width = int(width * self.window_scale)
            scaled_height = int(height * self.window_scale) + 60  # +60 for info panel

            self.video_writer = cv2.VideoWriter(self.output_file, fourcc, fps,
                                               (scaled_width, scaled_height))

            if not self.video_writer.isOpened():
                logger.error(f"Failed to initialize video writer for {self.output_file}")
                self.video_writer = None
            else:
                logger.info(f"Video writer initialized: {self.output_file}")

        self.is_running = True
        thread = threading.Thread(target=self._read_frames, daemon=True)
        thread.start()

        logger.info("Starting detection loop. Press 'q' to quit, SPACE to pause, 's' for screenshot")

        while self.is_running:
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Quit key pressed")
                self.is_running = False
                break
            elif key == ord(' '):
                self.is_paused = not self.is_paused
                logger.info(f"{'Paused' if self.is_paused else 'Resumed'}")
            elif key == ord('s'):
                with self.frame_lock:
                    if self.latest_frame is not None:
                        self._save_screenshot(self.latest_frame)

            if self.is_paused:
                continue

            # Get latest frame with thread safety
            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()

            frame = cv2.flip(frame, 1)  # Flip horizontally

            # Frame skipping logic
            self.frame_counter += 1
            should_process = (self.frame_skip == 0) or (self.frame_counter % (self.frame_skip + 1) == 0)

            if should_process:
                # Calculate FPS
                fps = self._calculate_fps()

                # Reset detection counts
                self.detection_counts.clear()

                # Perform object detection
                results = self.model(frame, stream=True, verbose=False)

                # Collect detections for tracking
                detections = []
                detection_data = []

                # Process results and collect detection data
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if self.classes_to_detect and cls not in self.classes_to_detect:
                            continue

                        conf = round(float(box.conf[0]), 2)
                        if conf < self.confidence_threshold:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = self.model.names[cls]

                        detections.append((x1, y1, x2, y2))
                        detection_data.append((cls, label, conf, (x1, y1, x2, y2)))

                        # Update detection counts
                        self.detection_counts[label] = self.detection_counts.get(label, 0) + 1

                # Update tracker
                if detections:
                    object_ids = self.tracker.update(detections)
                else:
                    object_ids = []

                # Draw bounding boxes with tracking IDs
                for i, (cls, label, conf, (x1, y1, x2, y2)) in enumerate(detection_data):
                    # Get color for this class
                    color = self.class_colors.get(cls, (255, 0, 0))

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with tracking ID
                    obj_id = object_ids[i] if i < len(object_ids) else -1
                    text = f'{label} {conf}'
                    if obj_id >= 0:
                        text += f' ID:{obj_id}'

                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10),
                                (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Add info panel
                frame_with_info = self._draw_info_panel(frame, fps)

                # Window resizing
                if self.window_scale != 1.0:
                    new_width = int(frame_with_info.shape[1] * self.window_scale)
                    new_height = int(frame_with_info.shape[0] * self.window_scale)
                    frame_with_info = cv2.resize(frame_with_info, (new_width, new_height))

                # Display the frame
                cv2.imshow('YOLOv8 Object Detection', frame_with_info)

                # Write to video file
                if self.video_writer:
                    self.video_writer.write(frame_with_info)
            else:
                # If skipping, just display the last processed frame
                time.sleep(0.001)

        thread.join(timeout=2.0)
        self.release_resources()

    def release_resources(self):
        """Release all resources."""
        logger.info("Releasing resources")
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()


def load_config(config_file: str) -> Dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_file):
        logger.warning(f"Config file not found: {config_file}")
        return {}

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection with Enhanced Features")
    parser.add_argument("--device", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output video file name")
    parser.add_argument("--classes", type=int, nargs='+', default=[],
                       help="List of class IDs to detect")
    parser.add_argument("--frame-skip", type=int, default=0,
                       help="Process every N frames (0 = process all frames)")
    parser.add_argument("--window-scale", type=float, default=1.0,
                       help="Scale factor for display window (e.g., 0.5 for half size)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON configuration file")
    parser.add_argument("--model", type=str, default='yolov8m.pt',
                       help="Path to YOLO model file")

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Command-line args override config file
    device = args.device if args.device != 0 else config.get('device', 0)
    confidence_threshold = args.confidence_threshold if args.confidence_threshold != 0.5 else config.get('confidence_threshold', 0.5)
    output_file = args.output_file or config.get('output_file')
    classes_to_detect = args.classes or config.get('classes', [])
    frame_skip = args.frame_skip if args.frame_skip != 0 else config.get('frame_skip', 0)
    window_scale = args.window_scale if args.window_scale != 1.0 else config.get('window_scale', 1.0)
    model_path = args.model if args.model != 'yolov8m.pt' else config.get('model', 'yolov8m.pt')

    try:
        detector = ObjectDetector(
            model_path=model_path,
            device_id=device,
            confidence_threshold=confidence_threshold,
            output_file=output_file,
            classes_to_detect=classes_to_detect,
            frame_skip=frame_skip,
            window_scale=window_scale,
            config_file=args.config
        )
        detector.run_detection()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
