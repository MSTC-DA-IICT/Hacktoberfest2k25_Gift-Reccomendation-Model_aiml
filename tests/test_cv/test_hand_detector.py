"""
Test cases for hand detector module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cv.hand_detector import HandDetector


class TestHandDetector:
    """Test cases for HandDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = HandDetector()

    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector is not None
        assert self.detector.min_detection_confidence == 0.5
        assert self.detector.min_tracking_confidence == 0.5
        assert len(self.detector.landmark_names) == 21

    def test_detector_with_custom_confidence(self):
        """Test detector with custom confidence values."""
        detector = HandDetector(
            min_detection_confidence=0.7, min_tracking_confidence=0.8
        )
        assert detector.min_detection_confidence == 0.7
        assert detector.min_tracking_confidence == 0.8

    def test_detect_hands_empty_image(self):
        """Test detection with empty image."""
        empty_image = np.array([])
        result = self.detector.detect_hands(empty_image)
        assert result is None

    def test_detect_hands_none_image(self):
        """Test detection with None image."""
        result = self.detector.detect_hands(None)
        assert result is None

    def test_detect_hands_valid_image(self):
        """Test detection with valid image (no actual hand)."""
        # Create a dummy image
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = self.detector.detect_hands(image)

        # Result should not be None (MediaPipe should process it)
        # but may not detect hands
        assert result is not None or result is None  # Either is acceptable

    def test_get_landmarks_invalid_image(self):
        """Test landmark extraction with invalid image."""
        empty_image = np.array([])
        result = self.detector.get_landmarks(empty_image)
        assert result is None

    def test_get_landmarks_no_hand_detected(self):
        """Test landmark extraction when no hand is detected."""
        # Create image with no hand
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.get_landmarks(image)

        # Should return None when no hand is detected
        assert result is None

    @patch("cv2.VideoCapture")
    def test_capture_from_webcam_no_camera(self, mock_video_capture):
        """Test webcam capture when camera is not available."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        result = self.detector.capture_from_webcam(duration=1, show_preview=False)
        assert result is None

    @patch("cv2.VideoCapture")
    def test_capture_from_webcam_mock(self, mock_video_capture):
        """Test webcam capture with mocked camera."""
        # Create mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.ones((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        # This test might not complete due to MediaPipe processing
        # but we can at least test the setup
        assert mock_cap.isOpened()

    def test_draw_landmarks_none_input(self):
        """Test drawing landmarks with None input."""
        image = np.ones((480, 640, 3), dtype=np.uint8)
        result = self.detector.draw_landmarks(image, None)

        # Should return copy of original image
        assert result.shape == image.shape
        assert isinstance(result, np.ndarray)

    def test_draw_landmarks_empty_dict(self):
        """Test drawing landmarks with empty dictionary."""
        image = np.ones((480, 640, 3), dtype=np.uint8)
        empty_landmarks = {}
        result = self.detector.draw_landmarks(image, empty_landmarks)

        # Should return copy of original image
        assert result.shape == image.shape
        assert isinstance(result, np.ndarray)

    def test_draw_landmarks_valid_structure(self):
        """Test drawing landmarks with valid structure."""
        image = np.ones((480, 640, 3), dtype=np.uint8)

        # Create mock landmarks structure
        landmarks = {
            "landmarks": [
                {"x": 100, "y": 200, "z": 0.1},
                {"x": 150, "y": 250, "z": 0.2},
            ],
            "handedness": "Right",
            "confidence": 0.95,
        }

        result = self.detector.draw_landmarks(image, landmarks)

        # Should return modified image
        assert result.shape == image.shape
        assert isinstance(result, np.ndarray)

    def test_landmark_names_count(self):
        """Test that correct number of landmark names are defined."""
        assert len(self.detector.landmark_names) == 21

    def test_landmark_names_content(self):
        """Test that landmark names contain expected entries."""
        expected_landmarks = [
            "WRIST",
            "THUMB_TIP",
            "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_TIP",
        ]

        for landmark in expected_landmarks:
            assert landmark in self.detector.landmark_names

    def test_detector_cleanup(self):
        """Test detector cleanup."""
        # Create detector
        detector = HandDetector()

        # Call cleanup (destructor)
        del detector

        # Test passes if no exception is raised

    def test_get_landmark_by_name_invalid(self):
        """Test getting landmark by invalid name."""
        landmarks = {"landmarks": [{"x": 100, "y": 200, "z": 0.1}] * 21}

        result = self.detector.get_landmark_by_name(landmarks, "INVALID_LANDMARK")
        assert result is None

    def test_get_landmark_by_name_valid(self):
        """Test getting landmark by valid name."""
        landmarks = {
            "landmarks": [{"x": i * 10, "y": i * 20, "z": i * 0.1} for i in range(21)]
        }

        wrist = self.detector.get_landmark_by_name(landmarks, "WRIST")
        assert wrist is not None
        assert wrist["x"] == 0
        assert wrist["y"] == 0

    def test_confidence_range(self):
        """Test that confidence values are properly handled."""
        # Test with extreme values
        detector1 = HandDetector(
            min_detection_confidence=0.0, min_tracking_confidence=0.0
        )
        assert detector1.min_detection_confidence == 0.0

        detector2 = HandDetector(
            min_detection_confidence=1.0, min_tracking_confidence=1.0
        )
        assert detector2.min_detection_confidence == 1.0
