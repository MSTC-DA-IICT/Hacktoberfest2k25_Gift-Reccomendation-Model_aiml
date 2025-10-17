"""
Utility functions for computer vision module.

This module provides helper functions for image processing,
validation, and common CV operations.
"""

import os
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_image(
    image_path: str, target_size: Optional[Tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """
    Load image from file path with optional resizing.

    Parameters
    ----------
    image_path : str
        Path to image file
    target_size : Optional[Tuple[int, int]]
        Target size as (width, height) for resizing

    Returns
    -------
    Optional[np.ndarray]
        Loaded image or None if failed

    Examples
    --------
    >>> image = load_image('hand.jpg', target_size=(640, 480))
    >>> if image is not None:
    ...     print(f"Loaded image with shape: {image.shape}")
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None

        # Load image
        image = cv2.imread(image_path)

        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Resize if target size specified
        if target_size is not None:
            width, height = target_size
            image = cv2.resize(image, (width, height))
            logger.debug(f"Resized image to {target_size}")

        logger.info(f"Successfully loaded image from {image_path}")
        return image

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """
    Save image to file.

    Parameters
    ----------
    image : np.ndarray
        Image to save
    output_path : str
        Output file path
    quality : int, default=95
        JPEG quality (0-100)

    Returns
    -------
    bool
        True if saved successfully, False otherwise

    Examples
    --------
    >>> success = save_image(processed_image, 'output.jpg', quality=90)
    >>> print(f"Save successful: {success}")
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Set compression parameters
        if output_path.lower().endswith(".jpg") or output_path.lower().endswith(
            ".jpeg"
        ):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:
            encode_params = []

        success = cv2.imwrite(output_path, image, encode_params)

        if success:
            logger.info(f"Image saved to {output_path}")
        else:
            logger.error(f"Failed to save image to {output_path}")

        return success

    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False


def validate_image_format(image_path: str) -> bool:
    """
    Validate if image file has supported format.

    Parameters
    ----------
    image_path : str
        Path to image file

    Returns
    -------
    bool
        True if format is supported, False otherwise

    Examples
    --------
    >>> is_valid = validate_image_format('hand.jpg')
    >>> print(f"Valid format: {is_valid}")
    """
    try:
        if not os.path.exists(image_path):
            return False

        # Check file extension
        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
        }
        file_extension = os.path.splitext(image_path)[1].lower()

        if file_extension not in supported_extensions:
            logger.warning(f"Unsupported image format: {file_extension}")
            return False

        # Try to load image to verify it's valid
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot read image file: {image_path}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating image format: {e}")
        return False


def preprocess_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better hand detection.

    Parameters
    ----------
    image : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Preprocessed image

    Examples
    --------
    >>> preprocessed = preprocess_image_for_detection(raw_image)
    >>> # Use preprocessed image for detection
    """
    try:
        processed = image.copy()

        # Apply basic enhancements for better detection
        # Adjust brightness and contrast
        processed = cv2.convertScaleAbs(processed, alpha=1.1, beta=10)

        # Apply slight Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        logger.debug("Applied preprocessing for hand detection")
        return processed

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image


def crop_hand_region(
    image: np.ndarray, landmarks_dict: dict, padding: int = 50
) -> Optional[np.ndarray]:
    """
    Crop image to focus on hand region.

    Parameters
    ----------
    image : np.ndarray
        Input image
    landmarks_dict : dict
        Landmarks dictionary with hand coordinates
    padding : int, default=50
        Padding around hand region

    Returns
    -------
    Optional[np.ndarray]
        Cropped hand region or None if failed

    Examples
    --------
    >>> cropped_hand = crop_hand_region(image, landmarks, padding=30)
    >>> if cropped_hand is not None:
    ...     cv2.imshow('Cropped Hand', cropped_hand)
    """
    if not landmarks_dict or "landmarks" not in landmarks_dict:
        logger.error("No landmarks provided for cropping")
        return None

    try:
        landmarks = landmarks_dict["landmarks"]
        height, width = image.shape[:2]

        # Find bounding box of all landmarks
        x_coords = [lm["x"] for lm in landmarks]
        y_coords = [lm["y"] for lm in landmarks]

        min_x = max(0, min(x_coords) - padding)
        max_x = min(width, max(x_coords) + padding)
        min_y = max(0, min(y_coords) - padding)
        max_y = min(height, max(y_coords) + padding)

        # Crop the image
        cropped = image[min_y:max_y, min_x:max_x]

        logger.debug(f"Cropped hand region: {cropped.shape}")
        return cropped

    except Exception as e:
        logger.error(f"Error cropping hand region: {e}")
        return None


def batch_process_images(
    image_paths: List[str], output_dir: str, processor_func, **kwargs
) -> List[str]:
    """
    Process multiple images in batch.

    Parameters
    ----------
    image_paths : List[str]
        List of input image paths
    output_dir : str
        Output directory for processed images
    processor_func : callable
        Function to process each image
    **kwargs
        Additional arguments for processor function

    Returns
    -------
    List[str]
        List of output file paths

    Examples
    --------
    >>> def my_processor(img):
    ...     return cv2.GaussianBlur(img, (5, 5), 0)
    >>> output_paths = batch_process_images(input_paths, 'output/', my_processor)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")

            # Load image
            image = load_image(image_path)
            if image is None:
                logger.error(f"Failed to load {image_path}, skipping")
                continue

            # Process image
            try:
                processed_image = processor_func(image, **kwargs)
                if processed_image is None:
                    logger.error(f"Processor returned None for {image_path}")
                    continue
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

            # Save processed image
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_processed{ext}"
            output_path = os.path.join(output_dir, output_filename)

            if save_image(processed_image, output_path):
                output_paths.append(output_path)

        logger.info(
            f"Batch processing completed: {len(output_paths)}/{len(image_paths)} successful"
        )
        return output_paths

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return []


def calculate_image_quality_score(image: np.ndarray) -> float:
    """
    Calculate a quality score for the image.

    Parameters
    ----------
    image : np.ndarray
        Input image

    Returns
    -------
    float
        Quality score between 0 and 1

    Examples
    --------
    >>> quality = calculate_image_quality_score(image)
    >>> print(f"Image quality: {quality:.2f}")
    """
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate various quality metrics

        # 1. Sharpness (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)  # Normalize

        # 2. Brightness (avoid too dark or too bright images)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0

        # 3. Contrast
        contrast = gray.std()
        contrast_score = min(1.0, contrast / 64.0)  # Normalize

        # 4. Overall quality (weighted average)
        quality_score = (
            0.5 * sharpness_score + 0.3 * brightness_score + 0.2 * contrast_score
        )

        logger.debug(
            f"Image quality components: sharpness={sharpness_score:.3f}, "
            f"brightness={brightness_score:.3f}, contrast={contrast_score:.3f}"
        )

        return max(0.0, min(1.0, quality_score))

    except Exception as e:
        logger.error(f"Error calculating image quality: {e}")
        return 0.0


def resize_image_maintain_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio, padding with fill color.

    Parameters
    ----------
    image : np.ndarray
        Input image
    target_size : Tuple[int, int]
        Target size as (width, height)
    fill_color : Tuple[int, int, int], default=(0, 0, 0)
        Fill color for padding

    Returns
    -------
    np.ndarray
        Resized image with padding

    Examples
    --------
    >>> resized = resize_image_maintain_aspect(image, (640, 480), (128, 128, 128))
    """
    try:
        target_width, target_height = target_size
        height, width = image.shape[:2]

        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)

        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height))

        # Create output image with target size
        if len(image.shape) == 3:
            output = np.full(
                (target_height, target_width, 3), fill_color, dtype=np.uint8
            )
        else:
            output = np.full(
                (target_height, target_width), fill_color[0], dtype=np.uint8
            )

        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Place resized image in the center
        output[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
            resized
        )

        logger.debug(
            f"Resized image from {width}x{height} to {target_width}x{target_height}"
        )
        return output

    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image


def extract_color_statistics(
    image: np.ndarray, landmarks_dict: Optional[dict] = None
) -> dict:
    """
    Extract color statistics from image or hand region.

    Parameters
    ----------
    image : np.ndarray
        Input image
    landmarks_dict : Optional[dict]
        If provided, analyze only hand region

    Returns
    -------
    dict
        Color statistics including mean, std for each channel

    Examples
    --------
    >>> stats = extract_color_statistics(image, landmarks)
    >>> print(f"Average skin tone (BGR): {stats['mean']}")
    """
    try:
        # Use hand region if landmarks provided
        if landmarks_dict:
            cropped = crop_hand_region(image, landmarks_dict)
            if cropped is not None:
                image = cropped

        # Calculate statistics
        if len(image.shape) == 3:
            # Color image
            mean_color = np.mean(image, axis=(0, 1))
            std_color = np.std(image, axis=(0, 1))

            stats = {
                "mean": mean_color.tolist(),
                "std": std_color.tolist(),
                "channels": ["blue", "green", "red"],
            }
        else:
            # Grayscale image
            mean_color = np.mean(image)
            std_color = np.std(image)

            stats = {"mean": [mean_color], "std": [std_color], "channels": ["gray"]}

        # Add HSV statistics for color images
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_mean = np.mean(hsv, axis=(0, 1))
            hsv_std = np.std(hsv, axis=(0, 1))

            stats["hsv_mean"] = hsv_mean.tolist()
            stats["hsv_std"] = hsv_std.tolist()
            stats["hsv_channels"] = ["hue", "saturation", "value"]

        logger.debug(f"Extracted color statistics: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error extracting color statistics: {e}")
        return {}


def create_image_grid(
    images: List[np.ndarray],
    grid_size: Tuple[int, int],
    image_size: Tuple[int, int] = (200, 200),
) -> np.ndarray:
    """
    Create a grid layout of multiple images.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images to arrange in grid
    grid_size : Tuple[int, int]
        Grid dimensions as (rows, cols)
    image_size : Tuple[int, int], default=(200, 200)
        Size to resize each image to

    Returns
    -------
    np.ndarray
        Grid image

    Examples
    --------
    >>> grid = create_image_grid(hand_images, (2, 3), (150, 150))
    >>> cv2.imshow('Hand Grid', grid)
    """
    try:
        rows, cols = grid_size
        img_height, img_width = image_size

        # Create empty grid
        grid_height = rows * img_height
        grid_width = cols * img_width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Place images in grid
        for i, image in enumerate(images[: rows * cols]):
            row = i // cols
            col = i % cols

            # Resize image
            if image.shape[:2] != image_size:
                resized_img = resize_image_maintain_aspect(
                    image, (img_width, img_height), (50, 50, 50)
                )
            else:
                resized_img = image

            # Ensure image is color
            if len(resized_img.shape) == 2:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

            # Place in grid
            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width

            grid[y_start:y_end, x_start:x_end] = resized_img

        logger.info(f"Created {rows}x{cols} image grid with {len(images)} images")
        return grid

    except Exception as e:
        logger.error(f"Error creating image grid: {e}")
        return np.zeros((200, 200, 3), dtype=np.uint8)
