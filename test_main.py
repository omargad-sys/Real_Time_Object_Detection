import unittest
import os
import json
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from main import ObjectDetector, ObjectTracker, load_config


class TestObjectTracker(unittest.TestCase):
    """Test cases for ObjectTracker class."""

    def setUp(self):
        self.tracker = ObjectTracker(iou_threshold=0.3)

    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.next_id, 0)
        self.assertEqual(len(self.tracker.tracked_objects), 0)
        self.assertEqual(self.tracker.iou_threshold, 0.3)

    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        iou = self.tracker._calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_calculate_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (0, 0, 10, 10)
        iou = self.tracker._calculate_iou(box1, box2)
        self.assertEqual(iou, 1.0)

    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        iou = self.tracker._calculate_iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_update_new_detection(self):
        """Test updating tracker with new detection."""
        detections = [(10, 10, 50, 50)]
        ids = self.tracker.update(detections)
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 0)
        self.assertEqual(len(self.tracker.tracked_objects), 1)

    def test_update_multiple_detections(self):
        """Test updating tracker with multiple detections."""
        detections = [(10, 10, 50, 50), (60, 60, 100, 100)]
        ids = self.tracker.update(detections)
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(self.tracker.tracked_objects), 2)

    def test_update_existing_detection(self):
        """Test updating tracker with existing detection (tracking continuity)."""
        # First frame
        detections1 = [(10, 10, 50, 50)]
        ids1 = self.tracker.update(detections1)

        # Second frame with slightly moved detection
        detections2 = [(12, 12, 52, 52)]
        ids2 = self.tracker.update(detections2)

        # Should maintain the same ID
        self.assertEqual(ids1[0], ids2[0])


class TestObjectDetector(unittest.TestCase):
    """Test cases for ObjectDetector class."""

    def setUp(self):
        # Create a temporary model file for testing
        self.temp_model = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        self.temp_model.close()
        self.model_path = self.temp_model.name

    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    @patch('main.YOLO')
    @patch('main.torch.cuda.is_available')
    def test_initialization_cpu(self, mock_cuda, mock_yolo):
        """Test detector initialization with CPU."""
        mock_cuda.return_value = False
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(
            model_path=self.model_path,
            device_id=0,
            confidence_threshold=0.5,
            output_file=None,
            classes_to_detect=[]
        )

        self.assertEqual(detector.device, 'cpu')
        mock_model.to.assert_called_once_with('cpu')

    @patch('main.YOLO')
    @patch('main.torch.cuda.is_available')
    def test_initialization_gpu(self, mock_cuda, mock_yolo):
        """Test detector initialization with GPU."""
        mock_cuda.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(
            model_path=self.model_path,
            device_id=0,
            confidence_threshold=0.5,
            output_file=None,
            classes_to_detect=[]
        )

        self.assertEqual(detector.device, 'cuda')
        mock_model.to.assert_called_once_with('cuda')

    def test_initialization_missing_model(self):
        """Test detector initialization with missing model file."""
        with self.assertRaises(FileNotFoundError):
            ObjectDetector(
                model_path='nonexistent_model.pt',
                device_id=0,
                confidence_threshold=0.5,
                output_file=None,
                classes_to_detect=[]
            )

    @patch('main.YOLO')
    @patch('main.torch.cuda.is_available')
    def test_confidence_threshold_setting(self, mock_cuda, mock_yolo):
        """Test confidence threshold is set correctly."""
        mock_cuda.return_value = False
        mock_yolo.return_value = MagicMock()

        detector = ObjectDetector(
            model_path=self.model_path,
            device_id=0,
            confidence_threshold=0.7,
            output_file=None,
            classes_to_detect=[]
        )

        self.assertEqual(detector.confidence_threshold, 0.7)

    @patch('main.YOLO')
    @patch('main.torch.cuda.is_available')
    def test_frame_skip_setting(self, mock_cuda, mock_yolo):
        """Test frame skip is set correctly."""
        mock_cuda.return_value = False
        mock_yolo.return_value = MagicMock()

        detector = ObjectDetector(
            model_path=self.model_path,
            device_id=0,
            confidence_threshold=0.5,
            output_file=None,
            classes_to_detect=[],
            frame_skip=2
        )

        self.assertEqual(detector.frame_skip, 2)

    @patch('main.YOLO')
    @patch('main.torch.cuda.is_available')
    def test_window_scale_setting(self, mock_cuda, mock_yolo):
        """Test window scale is set correctly."""
        mock_cuda.return_value = False
        mock_yolo.return_value = MagicMock()

        detector = ObjectDetector(
            model_path=self.model_path,
            device_id=0,
            confidence_threshold=0.5,
            output_file=None,
            classes_to_detect=[],
            window_scale=0.5
        )

        self.assertEqual(detector.window_scale, 0.5)

    @patch('main.YOLO')
    @patch('main.torch.cuda.is_available')
    def test_calculate_fps(self, mock_cuda, mock_yolo):
        """Test FPS calculation."""
        mock_cuda.return_value = False
        mock_yolo.return_value = MagicMock()

        detector = ObjectDetector(
            model_path=self.model_path,
            device_id=0,
            confidence_threshold=0.5,
            output_file=None,
            classes_to_detect=[]
        )

        # First call
        fps1 = detector._calculate_fps()
        self.assertGreater(fps1, 0)

        # Second call should return a different value
        import time
        time.sleep(0.01)
        fps2 = detector._calculate_fps()
        self.assertGreater(fps2, 0)

    @patch('main.YOLO')
    @patch('main.torch.cuda.is_available')
    def test_draw_info_panel(self, mock_cuda, mock_yolo):
        """Test info panel drawing."""
        mock_cuda.return_value = False
        mock_yolo.return_value = MagicMock()

        detector = ObjectDetector(
            model_path=self.model_path,
            device_id=0,
            confidence_threshold=0.5,
            output_file=None,
            classes_to_detect=[]
        )

        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw info panel
        result = detector._draw_info_panel(frame, 30.0)

        # Result should be taller than original frame
        self.assertGreater(result.shape[0], frame.shape[0])
        self.assertEqual(result.shape[1], frame.shape[1])


class TestConfigLoader(unittest.TestCase):
    """Test cases for configuration file loading."""

    def test_load_config_valid_file(self):
        """Test loading valid config file."""
        config_data = {
            'device': 1,
            'confidence_threshold': 0.7,
            'frame_skip': 2
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = load_config(config_file)
            self.assertEqual(config['device'], 1)
            self.assertEqual(config['confidence_threshold'], 0.7)
            self.assertEqual(config['frame_skip'], 2)
        finally:
            os.unlink(config_file)

    def test_load_config_nonexistent_file(self):
        """Test loading nonexistent config file."""
        config = load_config('nonexistent_config.json')
        self.assertEqual(config, {})

    def test_load_config_invalid_json(self):
        """Test loading invalid JSON config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content {')
            config_file = f.name

        try:
            config = load_config(config_file)
            self.assertEqual(config, {})
        finally:
            os.unlink(config_file)


if __name__ == '__main__':
    unittest.main()
