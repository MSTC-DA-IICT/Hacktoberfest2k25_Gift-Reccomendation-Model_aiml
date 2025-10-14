"""
Hand visualization module.

This module provides functionality to visualize hand landmarks,
measurements, and size classifications on images.
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HandVisualizer:
    """
    Visualize hand landmarks and measurements.

    This class provides methods to draw landmarks, measurements,
    and annotations on hand images for analysis and display.
    """

    def __init__(self):
        """Initialize the hand visualizer with default colors and fonts."""
        # Color scheme for different parts
        self.colors = {
            'wrist': (255, 0, 0),      # Red
            'thumb': (0, 255, 0),      # Green
            'index': (0, 0, 255),      # Blue
            'middle': (255, 255, 0),   # Cyan
            'ring': (255, 0, 255),     # Magenta
            'pinky': (0, 255, 255),    # Yellow
            'connection': (255, 255, 255),  # White
            'measurement': (0, 255, 127),   # Spring green
            'text': (255, 255, 255),        # White
            'background': (0, 0, 0)         # Black
        }

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Landmark indices for connections
        self.connections = [
            # Thumb
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            # Index finger
            [(0, 5), (5, 6), (6, 7), (7, 8)],
            # Middle finger
            [(0, 9), (9, 10), (10, 11), (11, 12)],
            # Ring finger
            [(0, 13), (13, 14), (14, 15), (15, 16)],
            # Pinky
            [(0, 17), (17, 18), (18, 19), (19, 20)]
        ]

        logger.info("HandVisualizer initialized successfully")

    def draw_landmarks(self, image: np.ndarray, landmarks_dict: Dict,
                      show_connections: bool = True,
                      show_indices: bool = False) -> np.ndarray:
        """
        Draw hand landmarks and connections on image.

        Parameters
        ----------
        image : np.ndarray
            Input image
        landmarks_dict : Dict
            Landmarks dictionary from HandDetector.get_landmarks()
        show_connections : bool, default=True
            Whether to draw connections between landmarks
        show_indices : bool, default=False
            Whether to show landmark indices as text

        Returns
        -------
        np.ndarray
            Image with drawn landmarks

        Examples
        --------
        >>> visualizer = HandVisualizer()
        >>> annotated_image = visualizer.draw_landmarks(image, landmarks)
        >>> cv2.imshow('Hand Landmarks', annotated_image)
        """
        if not landmarks_dict or 'landmarks' not in landmarks_dict:
            logger.warning("No landmarks provided to draw_landmarks")
            return image.copy()

        annotated_image = image.copy()
        landmarks = landmarks_dict['landmarks']

        try:
            # Draw connections first (so they appear behind landmarks)
            if show_connections:
                annotated_image = self._draw_connections(annotated_image, landmarks)

            # Draw landmarks
            for i, landmark in enumerate(landmarks):
                x, y = landmark['x'], landmark['y']
                color = self._get_landmark_color(i)

                # Draw landmark point
                cv2.circle(annotated_image, (x, y), 5, color, -1)
                cv2.circle(annotated_image, (x, y), 6, (0, 0, 0), 1)  # Black border

                # Draw index number if requested
                if show_indices:
                    cv2.putText(annotated_image, str(i), (x + 8, y - 8),
                               self.font, 0.3, self.colors['text'], 1)

            # Draw metadata
            self._draw_metadata(annotated_image, landmarks_dict)

            logger.debug("Successfully drew landmarks on image")

        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")

        return annotated_image

    def draw_measurements(self, image: np.ndarray, landmarks_dict: Dict,
                         measurements: Dict[str, float], size_class: str) -> np.ndarray:
        """
        Draw measurement lines and size information on image.

        Parameters
        ----------
        image : np.ndarray
            Input image
        landmarks_dict : Dict
            Landmarks dictionary
        measurements : Dict[str, float]
            Hand measurements
        size_class : str
            Size classification ('small', 'medium', 'large')

        Returns
        -------
        np.ndarray
            Image with measurements and size information

        Examples
        --------
        >>> visualizer = HandVisualizer()
        >>> measurements = {'hand_length': 165.5, 'palm_width': 85.2}
        >>> annotated_image = visualizer.draw_measurements(image, landmarks, measurements, 'medium')
        """
        if not landmarks_dict or not measurements:
            logger.warning("No landmarks or measurements provided")
            return image.copy()

        annotated_image = image.copy()
        landmarks = landmarks_dict['landmarks']

        try:
            # Draw basic landmarks first
            annotated_image = self.draw_landmarks(annotated_image, landmarks_dict,
                                                show_connections=False)

            # Draw measurement lines
            annotated_image = self._draw_measurement_lines(annotated_image, landmarks, measurements)

            # Draw size classification
            self._draw_size_info(annotated_image, measurements, size_class)

            logger.debug(f"Successfully drew measurements for {size_class} hand")

        except Exception as e:
            logger.error(f"Error drawing measurements: {e}")

        return annotated_image

    def create_comparison_view(self, images: List[np.ndarray],
                              labels: List[str]) -> np.ndarray:
        """
        Create a side-by-side comparison of multiple hand images.

        Parameters
        ----------
        images : List[np.ndarray]
            List of hand images
        labels : List[str]
            Labels for each image

        Returns
        -------
        np.ndarray
            Combined comparison image

        Examples
        --------
        >>> visualizer = HandVisualizer()
        >>> comparison = visualizer.create_comparison_view([img1, img2], ['Before', 'After'])
        """
        if not images or not labels or len(images) != len(labels):
            logger.error("Images and labels must be provided and have same length")
            return np.zeros((100, 100, 3), dtype=np.uint8)

        try:
            # Resize all images to same height
            target_height = 300
            resized_images = []

            for img in images:
                height, width = img.shape[:2]
                aspect_ratio = width / height
                new_width = int(target_height * aspect_ratio)
                resized_img = cv2.resize(img, (new_width, target_height))
                resized_images.append(resized_img)

            # Calculate total width
            total_width = sum(img.shape[1] for img in resized_images)
            combined_image = np.zeros((target_height + 50, total_width, 3), dtype=np.uint8)

            # Place images side by side
            x_offset = 0
            for i, (img, label) in enumerate(zip(resized_images, labels)):
                height, width = img.shape[:2]

                # Place image
                combined_image[50:50+height, x_offset:x_offset+width] = img

                # Add label
                text_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
                text_x = x_offset + (width - text_size[0]) // 2
                cv2.putText(combined_image, label, (text_x, 30),
                           self.font, self.font_scale, self.colors['text'], self.font_thickness)

                x_offset += width

            logger.debug(f"Created comparison view with {len(images)} images")
            return combined_image

        except Exception as e:
            logger.error(f"Error creating comparison view: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def save_visualization(self, image: np.ndarray, path: str) -> bool:
        """
        Save visualization to file.

        Parameters
        ----------
        image : np.ndarray
            Image to save
        path : str
            Output file path

        Returns
        -------
        bool
            True if saved successfully, False otherwise

        Examples
        --------
        >>> visualizer = HandVisualizer()
        >>> success = visualizer.save_visualization(annotated_image, 'hand_analysis.jpg')
        """
        try:
            success = cv2.imwrite(path, image)
            if success:
                logger.info(f"Visualization saved to {path}")
            else:
                logger.error(f"Failed to save visualization to {path}")
            return success

        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            return False

    def _draw_connections(self, image: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """Draw connections between landmarks."""
        try:
            for finger_connections in self.connections:
                for start_idx, end_idx in finger_connections:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = (landmarks[start_idx]['x'], landmarks[start_idx]['y'])
                        end_point = (landmarks[end_idx]['x'], landmarks[end_idx]['y'])
                        cv2.line(image, start_point, end_point,
                                self.colors['connection'], 2)

        except Exception as e:
            logger.error(f"Error drawing connections: {e}")

        return image

    def _get_landmark_color(self, landmark_index: int) -> Tuple[int, int, int]:
        """Get color for a specific landmark based on its index."""
        if landmark_index == 0:  # Wrist
            return self.colors['wrist']
        elif 1 <= landmark_index <= 4:  # Thumb
            return self.colors['thumb']
        elif 5 <= landmark_index <= 8:  # Index finger
            return self.colors['index']
        elif 9 <= landmark_index <= 12:  # Middle finger
            return self.colors['middle']
        elif 13 <= landmark_index <= 16:  # Ring finger
            return self.colors['ring']
        else:  # Pinky
            return self.colors['pinky']

    def _draw_metadata(self, image: np.ndarray, landmarks_dict: Dict) -> None:
        """Draw metadata information (handedness, confidence) on image."""
        try:
            height, width = image.shape[:2]

            # Draw handedness
            handedness = landmarks_dict.get('handedness', 'Unknown')
            confidence = landmarks_dict.get('confidence', 0.0)

            text = f"{handedness} Hand"
            cv2.putText(image, text, (10, 30),
                       self.font, self.font_scale, self.colors['text'], self.font_thickness)

            # Draw confidence
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(image, conf_text, (10, 60),
                       self.font, self.font_scale * 0.8, self.colors['text'], self.font_thickness)

        except Exception as e:
            logger.error(f"Error drawing metadata: {e}")

    def _draw_measurement_lines(self, image: np.ndarray, landmarks: List[Dict],
                               measurements: Dict[str, float]) -> np.ndarray:
        """Draw measurement lines on the image."""
        try:
            # Hand length (wrist to middle finger tip)
            if 'hand_length' in measurements:
                wrist = landmarks[0]
                middle_tip = landmarks[12]
                cv2.line(image, (wrist['x'], wrist['y']),
                        (middle_tip['x'], middle_tip['y']),
                        self.colors['measurement'], 3)

                # Add measurement text
                mid_x = (wrist['x'] + middle_tip['x']) // 2
                mid_y = (wrist['y'] + middle_tip['y']) // 2
                text = f"Length: {measurements['hand_length']:.1f}px"
                cv2.putText(image, text, (mid_x + 10, mid_y),
                           self.font, 0.5, self.colors['measurement'], 1)

            # Palm width (thumb MCP to pinky MCP)
            if 'palm_width' in measurements:
                thumb_mcp = landmarks[2]
                pinky_mcp = landmarks[17]
                cv2.line(image, (thumb_mcp['x'], thumb_mcp['y']),
                        (pinky_mcp['x'], pinky_mcp['y']),
                        self.colors['measurement'], 3)

                # Add measurement text
                mid_x = (thumb_mcp['x'] + pinky_mcp['x']) // 2
                mid_y = (thumb_mcp['y'] + pinky_mcp['y']) // 2
                text = f"Width: {measurements['palm_width']:.1f}px"
                cv2.putText(image, text, (mid_x + 10, mid_y + 20),
                           self.font, 0.5, self.colors['measurement'], 1)

        except Exception as e:
            logger.error(f"Error drawing measurement lines: {e}")

        return image

    def _draw_size_info(self, image: np.ndarray, measurements: Dict[str, float],
                       size_class: str) -> None:
        """Draw size classification and key measurements."""
        try:
            height, width = image.shape[:2]

            # Size classification
            size_text = f"Size: {size_class.upper()}"
            text_size = cv2.getTextSize(size_text, self.font, self.font_scale + 0.2,
                                       self.font_thickness + 1)[0]

            # Draw background rectangle for better visibility
            bg_start = (width - text_size[0] - 20, 10)
            bg_end = (width - 5, text_size[1] + 30)
            cv2.rectangle(image, bg_start, bg_end, (0, 0, 0), -1)
            cv2.rectangle(image, bg_start, bg_end, self.colors['text'], 2)

            # Draw size text
            cv2.putText(image, size_text, (width - text_size[0] - 10, text_size[1] + 20),
                       self.font, self.font_scale + 0.2, self._get_size_color(size_class),
                       self.font_thickness + 1)

            # Draw key measurements
            y_offset = 60
            key_measurements = ['hand_length', 'palm_width', 'middle_finger_length']

            for measurement in key_measurements:
                if measurement in measurements:
                    value = measurements[measurement]
                    text = f"{measurement.replace('_', ' ').title()}: {value:.1f}px"
                    cv2.putText(image, text, (width - 250, y_offset),
                               self.font, 0.5, self.colors['text'], 1)
                    y_offset += 25

        except Exception as e:
            logger.error(f"Error drawing size info: {e}")

    def _get_size_color(self, size_class: str) -> Tuple[int, int, int]:
        """Get color based on size classification."""
        size_colors = {
            'small': (0, 255, 255),    # Yellow
            'medium': (0, 165, 255),   # Orange
            'large': (0, 0, 255),      # Red
            'unknown': (128, 128, 128) # Gray
        }
        return size_colors.get(size_class.lower(), (255, 255, 255))

    def create_analysis_dashboard(self, image: np.ndarray, landmarks_dict: Dict,
                                 measurements: Dict[str, float], size_class: str,
                                 confidence: float = 1.0) -> np.ndarray:
        """
        Create a comprehensive analysis dashboard.

        Parameters
        ----------
        image : np.ndarray
            Original hand image
        landmarks_dict : Dict
            Landmarks dictionary
        measurements : Dict[str, float]
            Hand measurements
        size_class : str
            Size classification
        confidence : float, default=1.0
            Classification confidence

        Returns
        -------
        np.ndarray
            Dashboard image with all analysis information
        """
        try:
            # Create a larger canvas
            original_height, original_width = image.shape[:2]
            dashboard_width = max(800, original_width + 400)
            dashboard_height = max(600, original_height + 100)

            dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)

            # Place original image with landmarks
            annotated_image = self.draw_measurements(image, landmarks_dict, measurements, size_class)
            dashboard[50:50+original_height, 50:50+original_width] = annotated_image

            # Add title
            title = "Hand Size Analysis Dashboard"
            cv2.putText(dashboard, title, (50, 30),
                       self.font, 1.0, self.colors['text'], 2)

            # Add analysis panel
            panel_x = original_width + 100
            panel_y = 80

            # Size classification with confidence
            size_text = f"Classification: {size_class.upper()}"
            cv2.putText(dashboard, size_text, (panel_x, panel_y),
                       self.font, self.font_scale, self._get_size_color(size_class), 2)

            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(dashboard, conf_text, (panel_x, panel_y + 30),
                       self.font, self.font_scale, self.colors['text'], 1)

            # Measurements table
            table_y = panel_y + 80
            cv2.putText(dashboard, "Measurements:", (panel_x, table_y),
                       self.font, self.font_scale, self.colors['text'], 2)

            table_y += 40
            for measurement, value in measurements.items():
                if isinstance(value, (int, float)):
                    text = f"{measurement.replace('_', ' ').title()}: {value:.1f}px"
                    cv2.putText(dashboard, text, (panel_x, table_y),
                               self.font, 0.5, self.colors['text'], 1)
                    table_y += 25

            # Add handedness info
            if landmarks_dict:
                handedness = landmarks_dict.get('handedness', 'Unknown')
                hand_conf = landmarks_dict.get('confidence', 0.0)

                hand_info_y = table_y + 40
                cv2.putText(dashboard, "Hand Detection:", (panel_x, hand_info_y),
                           self.font, self.font_scale, self.colors['text'], 2)

                cv2.putText(dashboard, f"Hand: {handedness}", (panel_x, hand_info_y + 30),
                           self.font, 0.5, self.colors['text'], 1)
                cv2.putText(dashboard, f"Detection Conf: {hand_conf:.2f}", (panel_x, hand_info_y + 55),
                           self.font, 0.5, self.colors['text'], 1)

            logger.info("Created comprehensive analysis dashboard")
            return dashboard

        except Exception as e:
            logger.error(f"Error creating analysis dashboard: {e}")
            return image.copy()