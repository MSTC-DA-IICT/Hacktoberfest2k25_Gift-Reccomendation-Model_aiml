"""
Hand detection module using MediaPipe.

This module provides hand detection and landmark extraction functionality
using Google's MediaPipe framework.
"""

import time
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)


class HandDetector:
    """
    Hand detection using MediaPipe.

    This class provides methods to detect hands in images and extract
    landmark coordinates for further analysis.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the hand detector.

        Parameters
        ----------
        min_detection_confidence : float, default=0.5
            Minimum confidence for hand detection
        min_tracking_confidence : float, default=0.5
            Minimum confidence for hand tracking
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Hand landmark names for reference
        self.landmark_names = [
            "WRIST",
            "THUMB_CMC",
            "THUMB_MCP",
            "THUMB_IP",
            "THUMB_TIP",
            "INDEX_FINGER_MCP",
            "INDEX_FINGER_PIP",
            "INDEX_FINGER_DIP",
            "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP",
            "MIDDLE_FINGER_PIP",
            "MIDDLE_FINGER_DIP",
            "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP",
            "RING_FINGER_PIP",
            "RING_FINGER_DIP",
            "RING_FINGER_TIP",
            "PINKY_MCP",
            "PINKY_PIP",
            "PINKY_DIP",
            "PINKY_TIP",
        ]

        logger.info("HandDetector initialized successfully")

    def detect_hands(self, image: np.ndarray) -> Optional[Any]:
        """
        Detect hands in image and return raw MediaPipe results.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format

        Returns
        -------
        Optional[Any]
            MediaPipe results object or None if no hands detected

        Examples
        --------
        >>> detector = HandDetector()
        >>> image = cv2.imread('hand_image.jpg')
        >>> results = detector.detect_hands(image)
        >>> if results and results.multi_hand_landmarks:
        ...     print("Hand detected!")
        """
        if image is None or image.size == 0:
            logger.warning("Invalid image provided to detect_hands")
            return None

        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = self.hands.process(rgb_image)

            return results

        except Exception as e:
            logger.error(f"Error in hand detection: {e}")
            return None

    def get_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract hand landmark coordinates from image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format

        Returns
        -------
        Optional[Dict]
            Dictionary with landmark coordinates or None if no hands detected

        Examples
        --------
        >>> detector = HandDetector()
        >>> image = cv2.imread('hand_image.jpg')
        >>> landmarks = detector.get_landmarks(image)
        >>> if landmarks:
        ...     print(f"Wrist position: {landmarks['landmarks'][0]}")
        """
        results = self.detect_hands(image)

        if not results or not results.multi_hand_landmarks:
            return None

        try:
            # Use the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            height, width = image.shape[:2]

            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z  # Depth (relative to wrist)
                landmarks.append({"x": x, "y": y, "z": z})

            # Get hand classification (Left/Right)
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label

            landmark_dict = {
                "landmarks": landmarks,
                "handedness": handedness,
                "image_shape": (height, width),
                "confidence": (
                    results.multi_handedness[0].classification[0].score
                    if results.multi_handedness
                    else 0.0
                ),
            }

            logger.debug(f"Extracted {len(landmarks)} landmarks from {handedness} hand")
            return landmark_dict

        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None

    def get_landmark_by_name(
        self, landmarks_dict: Dict, landmark_name: str
    ) -> Optional[Dict]:
        """
        Get specific landmark by name.

        Parameters
        ----------
        landmarks_dict : Dict
            Landmarks dictionary from get_landmarks()
        landmark_name : str
            Name of the landmark (e.g., 'WRIST', 'THUMB_TIP')

        Returns
        -------
        Optional[Dict]
            Landmark coordinates or None if not found

        Examples
        --------
        >>> landmarks = detector.get_landmarks(image)
        >>> wrist = detector.get_landmark_by_name(landmarks, 'WRIST')
        >>> print(wrist)  # {'x': 320, 'y': 240, 'z': 0.0}
        """
        if not landmarks_dict or landmark_name not in self.landmark_names:
            return None

        try:
            index = self.landmark_names.index(landmark_name)
            return landmarks_dict["landmarks"][index]
        except (IndexError, KeyError):
            return None

    def capture_from_webcam(
        self, duration: int = 5, show_preview: bool = True
    ) -> Optional[np.ndarray]:
        """
        Capture hand image from webcam.

        Parameters
        ----------
        duration : int, default=5
            Maximum duration to wait for stable hand detection (seconds)
        show_preview : bool, default=True
            Whether to show camera preview window

        Returns
        -------
        Optional[np.ndarray]
            Captured image with detected hand or None if failed

        Examples
        --------
        >>> detector = HandDetector()
        >>> hand_image = detector.capture_from_webcam(duration=10)
        >>> if hand_image is not None:
        ...     cv2.imwrite('captured_hand.jpg', hand_image)
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Could not open webcam")
            return None

        logger.info(
            f"Starting webcam capture. Looking for stable hand detection for {duration} seconds..."
        )

        start_time = time.time()
        best_image = None
        best_confidence = 0.0
        stable_detections = 0
        required_stable_detections = 10  # Number of consecutive stable detections

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame from webcam")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Detect hands
                results = self.detect_hands(frame)
                current_time = time.time()

                if results and results.multi_hand_landmarks:
                    # Get confidence score
                    confidence = 0.0
                    if results.multi_handedness:
                        confidence = results.multi_handedness[0].classification[0].score

                    # Draw landmarks on frame for preview
                    annotated_frame = frame.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                        )

                    # Check if this is a good detection
                    if confidence > self.min_detection_confidence:
                        stable_detections += 1

                        # Update best image if confidence is higher
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_image = frame.copy()

                        # Add confidence text
                        cv2.putText(
                            annotated_frame,
                            f"Confidence: {confidence:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            annotated_frame,
                            f"Stable: {stable_detections}/{required_stable_detections}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        # If we have enough stable detections, we can finish
                        if stable_detections >= required_stable_detections:
                            logger.info(
                                f"Captured stable hand with confidence {best_confidence:.2f}"
                            )
                            break

                    else:
                        stable_detections = (
                            0  # Reset counter if detection is not confident
                        )

                    if show_preview:
                        cv2.imshow("Hand Detection (Press q to quit)", annotated_frame)

                else:
                    stable_detections = 0  # Reset counter if no hand detected

                    if show_preview:
                        # Show waiting message
                        waiting_frame = frame.copy()
                        cv2.putText(
                            waiting_frame,
                            "Place your hand in view...",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow("Hand Detection (Press q to quit)", waiting_frame)

                # Check for timeout
                if current_time - start_time > duration:
                    logger.warning(f"Webcam capture timed out after {duration} seconds")
                    break

                # Check for quit key
                if show_preview and cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Webcam capture interrupted by user")
                    break

        except Exception as e:
            logger.error(f"Error during webcam capture: {e}")

        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        if best_image is not None:
            logger.info(
                f"Successfully captured hand image with confidence {best_confidence:.2f}"
            )
        else:
            logger.warning("No suitable hand image captured")

        return best_image

    def draw_landmarks(self, image: np.ndarray, landmarks_dict: Dict) -> np.ndarray:
        """
        Draw hand landmarks on image.

        Parameters
        ----------
        image : np.ndarray
            Input image
        landmarks_dict : Dict
            Landmarks dictionary from get_landmarks()

        Returns
        -------
        np.ndarray
            Image with drawn landmarks

        Examples
        --------
        >>> detector = HandDetector()
        >>> landmarks = detector.get_landmarks(image)
        >>> if landmarks:
        ...     annotated_image = detector.draw_landmarks(image, landmarks)
        ...     cv2.imshow('Hand Landmarks', annotated_image)
        """
        if not landmarks_dict:
            return image.copy()

        annotated_image = image.copy()

        try:
            landmarks = landmarks_dict["landmarks"]

            # Draw landmarks as circles
            for i, landmark in enumerate(landmarks):
                x, y = landmark["x"], landmark["y"]

                # Different colors for different parts of the hand
                if i == 0:  # Wrist
                    color = (255, 0, 0)  # Red
                elif i in [1, 2, 3, 4]:  # Thumb
                    color = (0, 255, 0)  # Green
                elif i in [5, 6, 7, 8]:  # Index finger
                    color = (0, 0, 255)  # Blue
                elif i in [9, 10, 11, 12]:  # Middle finger
                    color = (255, 255, 0)  # Cyan
                elif i in [13, 14, 15, 16]:  # Ring finger
                    color = (255, 0, 255)  # Magenta
                else:  # Pinky
                    color = (0, 255, 255)  # Yellow

                cv2.circle(annotated_image, (x, y), 5, color, -1)

                # Add landmark index
                cv2.putText(
                    annotated_image,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )

            # Draw connections (simplified version)
            # This is a basic implementation - for full connections use MediaPipe's drawing utils
            connections = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),  # Thumb
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),  # Index
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),  # Middle
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),  # Ring
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),  # Pinky
            ]

            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (landmarks[start_idx]["x"], landmarks[start_idx]["y"])
                    end_point = (landmarks[end_idx]["x"], landmarks[end_idx]["y"])
                    cv2.line(
                        annotated_image, start_point, end_point, (255, 255, 255), 2
                    )

            # Add handedness label
            handedness = landmarks_dict.get("handedness", "Unknown")
            confidence = landmarks_dict.get("confidence", 0.0)
            cv2.putText(
                annotated_image,
                f"{handedness} Hand (Conf: {confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")

        return annotated_image

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, "hands"):
            self.hands.close()
