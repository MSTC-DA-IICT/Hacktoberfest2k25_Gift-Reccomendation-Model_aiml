"""
Input validation utilities.

This module provides validation functions for various inputs
used throughout the application.
"""

import os
import re
from typing import Union, List, Optional
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def validate_sentiment_score(score: Union[float, int]) -> bool:
    """
    Validate sentiment score is between 0 and 1.

    Parameters
    ----------
    score : Union[float, int]
        Sentiment score to validate

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> validate_sentiment_score(0.5)
    True
    >>> validate_sentiment_score(1.5)
    False
    """
    try:
        score = float(score)
        is_valid = 0.0 <= score <= 1.0
        if not is_valid:
            logger.warning(
                f"Invalid sentiment score: {score} (must be between 0 and 1)"
            )
        return is_valid
    except (ValueError, TypeError):
        logger.error(f"Cannot convert sentiment score to float: {score}")
        return False


def validate_hand_size(size: str) -> bool:
    """
    Validate hand size is a valid category.

    Parameters
    ----------
    size : str
        Hand size to validate

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> validate_hand_size('small')
    True
    >>> validate_hand_size('extra-large')
    False
    """
    if not isinstance(size, str):
        logger.error(f"Hand size must be string, got {type(size)}")
        return False

    valid_sizes = {"small", "medium", "large"}
    is_valid = size.lower() in valid_sizes

    if not is_valid:
        logger.warning(f"Invalid hand size: {size} (must be one of {valid_sizes})")

    return is_valid


def validate_image_path(path: str) -> bool:
    """
    Validate image file exists and is in a valid format.

    Parameters
    ----------
    path : str
        Image file path

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> validate_image_path('hand.jpg')
    True  # If file exists and is valid
    >>> validate_image_path('nonexistent.txt')
    False
    """
    try:
        if not isinstance(path, str):
            logger.error("Image path must be a string")
            return False

        if not path.strip():
            logger.error("Image path cannot be empty")
            return False

        # Check if file exists
        if not os.path.isfile(path):
            logger.error(f"Image file not found: {path}")
            return False

        # Check file extension
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        file_extension = os.path.splitext(path)[1].lower()

        if file_extension not in valid_extensions:
            logger.error(f"Invalid image format: {file_extension}")
            return False

        # Try to open with PIL to verify it's a valid image
        try:
            with Image.open(path) as img:
                img.verify()  # Verify it's a valid image
            return True
        except Exception as e:
            logger.error(f"Invalid image file {path}: {e}")
            return False

    except Exception as e:
        logger.error(f"Error validating image path: {e}")
        return False


def validate_text_input(text: str, min_length: int = 1, max_length: int = 1000) -> bool:
    """
    Validate text input for processing.

    Parameters
    ----------
    text : str
        Text to validate
    min_length : int, default=1
        Minimum text length
    max_length : int, default=1000
        Maximum text length

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> validate_text_input("This is a valid tweet!")
    True
    >>> validate_text_input("")
    False
    """
    try:
        if not isinstance(text, str):
            logger.error("Text input must be a string")
            return False

        text_length = len(text.strip())

        if text_length < min_length:
            logger.error(f"Text too short: {text_length} < {min_length}")
            return False

        if text_length > max_length:
            logger.error(f"Text too long: {text_length} > {max_length}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating text input: {e}")
        return False


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Parameters
    ----------
    email : str
        Email address to validate

    Returns
    -------
    bool
        True if valid format, False otherwise

    Examples
    --------
    >>> validate_email("user@example.com")
    True
    >>> validate_email("invalid-email")
    False
    """
    try:
        if not isinstance(email, str):
            return False

        # Basic email regex pattern
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        is_valid = re.match(pattern, email.strip()) is not None

        if not is_valid:
            logger.warning(f"Invalid email format: {email}")

        return is_valid

    except Exception as e:
        logger.error(f"Error validating email: {e}")
        return False


def validate_numpy_array(
    array: np.ndarray,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[np.dtype] = None,
) -> bool:
    """
    Validate NumPy array properties.

    Parameters
    ----------
    array : np.ndarray
        Array to validate
    expected_shape : Optional[tuple]
        Expected array shape
    expected_dtype : Optional[np.dtype]
        Expected data type

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> validate_numpy_array(arr, expected_shape=(2, 2))
    True
    """
    try:
        if not isinstance(array, np.ndarray):
            logger.error(f"Expected numpy array, got {type(array)}")
            return False

        if array.size == 0:
            logger.error("Array is empty")
            return False

        if expected_shape is not None and array.shape != expected_shape:
            logger.error(
                f"Shape mismatch: expected {expected_shape}, got {array.shape}"
            )
            return False

        if expected_dtype is not None and array.dtype != expected_dtype:
            logger.error(
                f"Dtype mismatch: expected {expected_dtype}, got {array.dtype}"
            )
            return False

        # Check for NaN or infinite values
        if np.isnan(array).any():
            logger.error("Array contains NaN values")
            return False

        if np.isinf(array).any():
            logger.error("Array contains infinite values")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating numpy array: {e}")
        return False


def validate_landmarks_dict(landmarks_dict: dict) -> bool:
    """
    Validate hand landmarks dictionary structure.

    Parameters
    ----------
    landmarks_dict : dict
        Landmarks dictionary from hand detection

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> landmarks = {'landmarks': [{'x': 100, 'y': 200, 'z': 0.1}], 'handedness': 'Right'}
    >>> validate_landmarks_dict(landmarks)
    True
    """
    try:
        if not isinstance(landmarks_dict, dict):
            logger.error("Landmarks must be a dictionary")
            return False

        # Check required keys
        required_keys = ["landmarks"]
        for key in required_keys:
            if key not in landmarks_dict:
                logger.error(f"Missing required key in landmarks: {key}")
                return False

        landmarks = landmarks_dict["landmarks"]

        if not isinstance(landmarks, list):
            logger.error("Landmarks must be a list")
            return False

        if len(landmarks) != 21:  # MediaPipe hands has 21 landmarks
            logger.error(f"Expected 21 landmarks, got {len(landmarks)}")
            return False

        # Validate each landmark
        for i, landmark in enumerate(landmarks):
            if not isinstance(landmark, dict):
                logger.error(f"Landmark {i} must be a dictionary")
                return False

            required_coords = ["x", "y", "z"]
            for coord in required_coords:
                if coord not in landmark:
                    logger.error(f"Landmark {i} missing coordinate: {coord}")
                    return False

                try:
                    float(landmark[coord])
                except (ValueError, TypeError):
                    logger.error(f"Landmark {i} coordinate {coord} must be numeric")
                    return False

        return True

    except Exception as e:
        logger.error(f"Error validating landmarks dict: {e}")
        return False


def validate_gift_dict(gift: dict) -> bool:
    """
    Validate gift dictionary structure.

    Parameters
    ----------
    gift : dict
        Gift dictionary

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> gift = {
    ...     'id': 1,
    ...     'name': 'Toy Car',
    ...     'category': 'toy',
    ...     'hand_size': 'small',
    ...     'sentiment_min': 0.0,
    ...     'sentiment_max': 1.0
    ... }
    >>> validate_gift_dict(gift)
    True
    """
    try:
        if not isinstance(gift, dict):
            logger.error("Gift must be a dictionary")
            return False

        # Required fields
        required_fields = [
            "id",
            "name",
            "category",
            "hand_size",
            "sentiment_min",
            "sentiment_max",
        ]

        for field in required_fields:
            if field not in gift:
                logger.error(f"Gift missing required field: {field}")
                return False

        # Validate ID
        try:
            int(gift["id"])
        except (ValueError, TypeError):
            logger.error(f"Gift ID must be an integer: {gift['id']}")
            return False

        # Validate name
        if not isinstance(gift["name"], str) or not gift["name"].strip():
            logger.error("Gift name must be a non-empty string")
            return False

        # Validate category
        if not isinstance(gift["category"], str) or not gift["category"].strip():
            logger.error("Gift category must be a non-empty string")
            return False

        # Validate hand size
        if not validate_hand_size(gift["hand_size"]):
            return False

        # Validate sentiment range
        try:
            sent_min = float(gift["sentiment_min"])
            sent_max = float(gift["sentiment_max"])

            if not (0 <= sent_min <= sent_max <= 1):
                logger.error(f"Invalid sentiment range: [{sent_min}, {sent_max}]")
                return False

        except (ValueError, TypeError):
            logger.error("Sentiment min/max must be numeric")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating gift dict: {e}")
        return False


def validate_config_file(config_path: str) -> bool:
    """
    Validate configuration file exists and is readable.

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> validate_config_file('config/config.yaml')
    True
    """
    try:
        if not isinstance(config_path, str):
            logger.error("Config path must be a string")
            return False

        if not os.path.isfile(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False

        # Check if file is readable
        try:
            with open(config_path, "r") as f:
                f.read(1)  # Try to read first character
            return True
        except PermissionError:
            logger.error(f"No permission to read config file: {config_path}")
            return False
        except Exception as e:
            logger.error(f"Cannot read config file {config_path}: {e}")
            return False

    except Exception as e:
        logger.error(f"Error validating config file: {e}")
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be safe for filesystem.

    Parameters
    ----------
    filename : str
        Original filename

    Returns
    -------
    str
        Sanitized filename

    Examples
    --------
    >>> sanitize_filename("my file?.txt")
    'my_file_.txt'
    """
    try:
        if not isinstance(filename, str):
            return "unnamed_file"

        # Replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")

        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed_file"

        # Limit length
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:190] + ext

        return sanitized

    except Exception as e:
        logger.error(f"Error sanitizing filename: {e}")
        return "unnamed_file"
