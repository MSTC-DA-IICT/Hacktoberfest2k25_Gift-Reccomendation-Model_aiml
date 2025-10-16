"""
Hand size estimation module.

This module provides functionality to estimate hand size from MediaPipe landmarks
and classify hands into size categories.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HandSizeEstimator:
    """
    Estimate hand size from MediaPipe landmarks.

    This class calculates various hand measurements and classifies
    hands into size categories (Small, Medium, Large).
    """

    def __init__(self, small_threshold: int = 150, medium_threshold: int = 200):
        """
        Initialize the hand size estimator.

        Parameters
        ----------
        small_threshold : int, default=150
            Maximum pixel measurement for small hands
        medium_threshold : int, default=200
            Maximum pixel measurement for medium hands
        """
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold

        # MediaPipe landmark indices for key points
        self.landmark_indices = {
            "wrist": 0,
            "thumb_tip": 4,
            "index_tip": 8,
            "middle_tip": 12,
            "ring_tip": 16,
            "pinky_tip": 20,
            "thumb_mcp": 2,
            "index_mcp": 5,
            "middle_mcp": 9,
            "ring_mcp": 13,
            "pinky_mcp": 17,
        }

        logger.info(
            f"HandSizeEstimator initialized with thresholds: small<={small_threshold}, medium<={medium_threshold}"
        )

    def calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """
        Calculate Euclidean distance between two landmarks.

        Parameters
        ----------
        point1 : Dict
            First landmark with 'x' and 'y' keys
        point2 : Dict
            Second landmark with 'x' and 'y' keys

        Returns
        -------
        float
            Euclidean distance in pixels

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> p1 = {'x': 100, 'y': 200}
        >>> p2 = {'x': 150, 'y': 250}
        >>> distance = estimator.calculate_distance(p1, p2)
        >>> print(f"Distance: {distance:.2f}")
        """
        try:
            dx = point2["x"] - point1["x"]
            dy = point2["y"] - point1["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            return distance
        except (KeyError, TypeError) as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0

    def calculate_finger_length(self, landmarks_dict: Dict) -> Dict[str, float]:
        """
        Calculate finger lengths from tip to MCP joint.

        Parameters
        ----------
        landmarks_dict : Dict
            Landmarks dictionary from HandDetector.get_landmarks()

        Returns
        -------
        Dict[str, float]
            Dictionary with finger lengths

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> finger_lengths = estimator.calculate_finger_length(landmarks)
        >>> print(finger_lengths)
        {'thumb': 85.4, 'index': 102.3, 'middle': 115.7, 'ring': 108.1, 'pinky': 78.9}
        """
        if not landmarks_dict or "landmarks" not in landmarks_dict:
            return {}

        landmarks = landmarks_dict["landmarks"]

        try:
            finger_lengths = {}

            # Calculate each finger length (tip to MCP)
            fingers = [
                ("thumb", "thumb_tip", "thumb_mcp"),
                ("index", "index_tip", "index_mcp"),
                ("middle", "middle_tip", "middle_mcp"),
                ("ring", "ring_tip", "ring_mcp"),
                ("pinky", "pinky_tip", "pinky_mcp"),
            ]

            for finger_name, tip_name, mcp_name in fingers:
                tip_idx = self.landmark_indices[tip_name]
                mcp_idx = self.landmark_indices[mcp_name]

                if tip_idx < len(landmarks) and mcp_idx < len(landmarks):
                    tip_point = landmarks[tip_idx]
                    mcp_point = landmarks[mcp_idx]
                    length = self.calculate_distance(tip_point, mcp_point)
                    finger_lengths[finger_name] = length

            logger.debug(f"Calculated finger lengths: {finger_lengths}")
            return finger_lengths

        except Exception as e:
            logger.error(f"Error calculating finger lengths: {e}")
            return {}

    def calculate_palm_width(self, landmarks_dict: Dict) -> float:
        """
        Calculate palm width from thumb MCP to pinky MCP.

        Parameters
        ----------
        landmarks_dict : Dict
            Landmarks dictionary from HandDetector.get_landmarks()

        Returns
        -------
        float
            Palm width in pixels

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> palm_width = estimator.calculate_palm_width(landmarks)
        >>> print(f"Palm width: {palm_width:.2f} pixels")
        """
        if not landmarks_dict or "landmarks" not in landmarks_dict:
            return 0.0

        landmarks = landmarks_dict["landmarks"]

        try:
            thumb_mcp_idx = self.landmark_indices["thumb_mcp"]
            pinky_mcp_idx = self.landmark_indices["pinky_mcp"]

            if thumb_mcp_idx < len(landmarks) and pinky_mcp_idx < len(landmarks):
                thumb_mcp = landmarks[thumb_mcp_idx]
                pinky_mcp = landmarks[pinky_mcp_idx]
                width = self.calculate_distance(thumb_mcp, pinky_mcp)

                logger.debug(f"Calculated palm width: {width:.2f}")
                return width

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating palm width: {e}")
            return 0.0

    def calculate_hand_span(self, landmarks_dict: Dict) -> float:
        """
        Calculate hand span from thumb tip to pinky tip.

        Parameters
        ----------
        landmarks_dict : Dict
            Landmarks dictionary from HandDetector.get_landmarks()

        Returns
        -------
        float
            Hand span in pixels

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> hand_span = estimator.calculate_hand_span(landmarks)
        >>> print(f"Hand span: {hand_span:.2f} pixels")
        """
        if not landmarks_dict or "landmarks" not in landmarks_dict:
            return 0.0

        landmarks = landmarks_dict["landmarks"]

        try:
            thumb_tip_idx = self.landmark_indices["thumb_tip"]
            pinky_tip_idx = self.landmark_indices["pinky_tip"]

            if thumb_tip_idx < len(landmarks) and pinky_tip_idx < len(landmarks):
                thumb_tip = landmarks[thumb_tip_idx]
                pinky_tip = landmarks[pinky_tip_idx]
                span = self.calculate_distance(thumb_tip, pinky_tip)

                logger.debug(f"Calculated hand span: {span:.2f}")
                return span

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating hand span: {e}")
            return 0.0

    def calculate_hand_length(self, landmarks_dict: Dict) -> float:
        """
        Calculate hand length from wrist to middle finger tip.

        Parameters
        ----------
        landmarks_dict : Dict
            Landmarks dictionary from HandDetector.get_landmarks()

        Returns
        -------
        float
            Hand length in pixels

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> hand_length = estimator.calculate_hand_length(landmarks)
        >>> print(f"Hand length: {hand_length:.2f} pixels")
        """
        if not landmarks_dict or "landmarks" not in landmarks_dict:
            return 0.0

        landmarks = landmarks_dict["landmarks"]

        try:
            wrist_idx = self.landmark_indices["wrist"]
            middle_tip_idx = self.landmark_indices["middle_tip"]

            if wrist_idx < len(landmarks) and middle_tip_idx < len(landmarks):
                wrist = landmarks[wrist_idx]
                middle_tip = landmarks[middle_tip_idx]
                length = self.calculate_distance(wrist, middle_tip)

                logger.debug(f"Calculated hand length: {length:.2f}")
                return length

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating hand length: {e}")
            return 0.0

    def calculate_all_measurements(self, landmarks_dict: Dict) -> Dict[str, float]:
        """
        Calculate all hand measurements.

        Parameters
        ----------
        landmarks_dict : Dict
            Landmarks dictionary from HandDetector.get_landmarks()

        Returns
        -------
        Dict[str, float]
            Dictionary with all measurements

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> measurements = estimator.calculate_all_measurements(landmarks)
        >>> print(measurements)
        {
            'palm_width': 120.5,
            'hand_span': 180.3,
            'hand_length': 165.7,
            'middle_finger_length': 78.2
        }
        """
        if not landmarks_dict:
            return {}

        measurements = {}

        try:
            # Basic measurements
            measurements["palm_width"] = self.calculate_palm_width(landmarks_dict)
            measurements["hand_span"] = self.calculate_hand_span(landmarks_dict)
            measurements["hand_length"] = self.calculate_hand_length(landmarks_dict)

            # Finger lengths
            finger_lengths = self.calculate_finger_length(landmarks_dict)
            measurements.update(finger_lengths)

            # Add middle finger length separately as it's often the primary measurement
            if "middle" in finger_lengths:
                measurements["middle_finger_length"] = finger_lengths["middle"]

            logger.debug(f"Calculated all measurements: {measurements}")

        except Exception as e:
            logger.error(f"Error calculating measurements: {e}")

        return measurements

    def classify_size(
        self, measurements: Dict[str, float], primary_metric: str = "hand_length"
    ) -> str:
        """
        Classify hand size based on measurements.

        Parameters
        ----------
        measurements : Dict[str, float]
            Dictionary with hand measurements
        primary_metric : str, default='hand_length'
            Primary measurement to use for classification

        Returns
        -------
        str
            Size classification: 'small', 'medium', or 'large'

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> measurements = {'hand_length': 180.5, 'palm_width': 85.2}
        >>> size = estimator.classify_size(measurements)
        >>> print(size)  # 'medium'
        """
        if not measurements or primary_metric not in measurements:
            logger.warning(
                f"Primary metric '{primary_metric}' not found in measurements"
            )
            # Fallback to any available measurement
            if measurements:
                primary_metric = next(iter(measurements))
            else:
                return "unknown"

        try:
            primary_value = measurements[primary_metric]

            if primary_value <= self.small_threshold:
                size = "small"
            elif primary_value <= self.medium_threshold:
                size = "medium"
            else:
                size = "large"

            logger.info(
                f"Classified hand as '{size}' based on {primary_metric}={primary_value:.2f}"
            )
            return size

        except Exception as e:
            logger.error(f"Error classifying hand size: {e}")
            return "unknown"

    def classify_size_multi_metric(
        self, measurements: Dict[str, float], weights: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float]:
        """
        Classify hand size using multiple metrics with weights.

        Parameters
        ----------
        measurements : Dict[str, float]
            Dictionary with hand measurements
        weights : Optional[Dict[str, float]]
            Weights for different measurements

        Returns
        -------
        Tuple[str, float]
            Size classification and confidence score

        Examples
        --------
        >>> estimator = HandSizeEstimator()
        >>> measurements = {'hand_length': 180.5, 'palm_width': 85.2}
        >>> size, confidence = estimator.classify_size_multi_metric(measurements)
        >>> print(f"Size: {size}, Confidence: {confidence:.2f}")
        """
        if not measurements:
            return "unknown", 0.0

        # Default weights - hand_length is most reliable
        if weights is None:
            weights = {
                "hand_length": 0.4,
                "middle_finger_length": 0.3,
                "palm_width": 0.2,
                "hand_span": 0.1,
            }

        try:
            weighted_scores = {"small": 0.0, "medium": 0.0, "large": 0.0}
            total_weight = 0.0

            for metric, value in measurements.items():
                if metric in weights:
                    weight = weights[metric]
                    total_weight += weight

                    # Calculate scores for each size category
                    if value <= self.small_threshold:
                        weighted_scores["small"] += weight * (
                            1 - value / self.small_threshold
                        )
                        weighted_scores["medium"] += weight * (
                            value / self.small_threshold * 0.5
                        )
                    elif value <= self.medium_threshold:
                        small_distance = abs(value - self.small_threshold) / (
                            self.medium_threshold - self.small_threshold
                        )
                        weighted_scores["small"] += weight * max(
                            0, 1 - small_distance * 2
                        )
                        weighted_scores["medium"] += weight * (1 - small_distance)
                        weighted_scores["large"] += weight * small_distance * 0.5
                    else:
                        large_factor = min(
                            1.0, (value - self.medium_threshold) / self.medium_threshold
                        )
                        weighted_scores["medium"] += weight * max(
                            0, 1 - large_factor * 2
                        )
                        weighted_scores["large"] += weight * (1 - large_factor * 0.5)

            if total_weight == 0:
                return "unknown", 0.0

            # Normalize scores
            for size in weighted_scores:
                weighted_scores[size] /= total_weight

            # Find best classification
            best_size = max(weighted_scores, key=weighted_scores.get)
            confidence = weighted_scores[best_size]

            logger.info(
                f"Multi-metric classification: {best_size} (confidence: {confidence:.3f})"
            )
            return best_size, confidence

        except Exception as e:
            logger.error(f"Error in multi-metric classification: {e}")
            return "unknown", 0.0

    def estimate_real_world_size(
        self, measurements: Dict[str, float], reference_distance: float = 60.0
    ) -> Dict[str, float]:
        """
        Estimate real-world hand measurements in centimeters.

        Parameters
        ----------
        measurements : Dict[str, float]
            Pixel measurements
        reference_distance : float, default=60.0
            Assumed distance from camera in centimeters

        Returns
        -------
        Dict[str, float]
            Real-world measurements in centimeters

        Note
        ----
        This is a rough approximation and requires calibration for accurate results.
        """
        if not measurements:
            return {}

        # This is a simplified conversion - would need camera calibration for accuracy
        # Assuming average pixel-to-cm ratio based on typical webcam setup
        pixel_to_cm_ratio = 0.1  # This would need calibration

        real_measurements = {}
        for key, pixel_value in measurements.items():
            real_measurements[f"{key}_cm"] = pixel_value * pixel_to_cm_ratio

        logger.debug(f"Estimated real-world measurements: {real_measurements}")
        return real_measurements

    def get_size_statistics(self, measurements: Dict[str, float]) -> Dict[str, any]:
        """
        Get detailed statistics about the hand size estimation.

        Parameters
        ----------
        measurements : Dict[str, float]
            Hand measurements

        Returns
        -------
        Dict[str, any]
            Statistics dictionary with various metrics
        """
        if not measurements:
            return {}

        stats = {
            "measurements": measurements,
            "primary_classification": self.classify_size(measurements),
            "thresholds": {
                "small_max": self.small_threshold,
                "medium_max": self.medium_threshold,
            },
        }

        # Multi-metric classification
        multi_size, confidence = self.classify_size_multi_metric(measurements)
        stats["multi_metric"] = {"classification": multi_size, "confidence": confidence}

        # Size ratios relative to thresholds
        if "hand_length" in measurements:
            hand_length = measurements["hand_length"]
            stats["size_ratios"] = {
                "vs_small_threshold": hand_length / self.small_threshold,
                "vs_medium_threshold": hand_length / self.medium_threshold,
            }

        return stats
