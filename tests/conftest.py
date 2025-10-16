"""
Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return [
        "I love this amazing product!",
        "This is terrible and disappointing",
        "It's okay, nothing special",
        "Absolutely fantastic experience!",
        "Worst purchase ever made",
        "Great quality and fast shipping",
        "Not satisfied with the service",
        "Excellent customer support!",
        "Poor build quality",
        "Highly recommended!",
    ]


@pytest.fixture
def sample_sentiment_labels():
    """Sample sentiment labels corresponding to text data."""
    return [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
    ]


@pytest.fixture
def sample_image():
    """Sample image for testing (simple numpy array)."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


@pytest.fixture
def sample_landmarks():
    """Sample hand landmarks for testing."""
    landmarks = []
    for i in range(21):  # MediaPipe has 21 hand landmarks
        landmarks.append({"x": 100 + i * 10, "y": 200 + i * 5, "z": 0.1 * i})

    return {
        "landmarks": landmarks,
        "handedness": "Right",
        "confidence": 0.95,
        "image_shape": (480, 640),
    }


@pytest.fixture
def sample_measurements():
    """Sample hand measurements for testing."""
    return {
        "hand_length": 165.5,
        "palm_width": 85.2,
        "hand_span": 180.3,
        "middle_finger_length": 78.2,
        "thumb_length": 45.6,
        "index_length": 72.1,
        "ring_length": 68.9,
        "pinky_length": 58.4,
    }


@pytest.fixture
def sample_gift_data():
    """Sample gift data for testing."""
    return [
        {
            "id": 1,
            "name": "Toy Car",
            "category": "toy",
            "hand_size": "small",
            "sentiment_min": 0.5,
            "sentiment_max": 1.0,
            "price_range": "$10-$20",
            "description": "Colorful toy car for kids",
        },
        {
            "id": 2,
            "name": "Story Book",
            "category": "book",
            "hand_size": "small",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$8-$15",
            "description": "Engaging storybook for young readers",
        },
        {
            "id": 3,
            "name": "Stylish Watch",
            "category": "accessory",
            "hand_size": "medium",
            "sentiment_min": 0.6,
            "sentiment_max": 1.0,
            "price_range": "$50-$100",
            "description": "Elegant watch for daily wear",
        },
        {
            "id": 4,
            "name": "Comfort Pillow",
            "category": "home",
            "hand_size": "medium",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$25-$40",
            "description": "Soft pillow for relaxation",
        },
        {
            "id": 5,
            "name": "Gaming Mouse",
            "category": "tech",
            "hand_size": "large",
            "sentiment_min": 0.5,
            "sentiment_max": 1.0,
            "price_range": "$45-$90",
            "description": "High-precision gaming mouse",
        },
    ]


@pytest.fixture
def temp_gift_config(sample_gift_data):
    """Temporary gift configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"gifts": sample_gift_data}, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_csv_file(sample_text_data, sample_sentiment_labels):
    """Temporary CSV file with text and sentiment data."""
    import pandas as pd

    df = pd.DataFrame({"text": sample_text_data, "sentiment": sample_sentiment_labels})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_image_file(sample_image):
    """Temporary image file."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name

    cv2.imwrite(temp_path, sample_image)

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_feature_matrix():
    """Sample feature matrix for ML testing."""
    np.random.seed(42)  # For reproducibility
    return np.random.randn(100, 50)  # 100 samples, 50 features


@pytest.fixture
def sample_labels():
    """Sample binary labels for ML testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)  # 100 binary labels


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "nlp": {
            "word2vec": {
                "vector_size": 100,
                "window": 5,
                "min_count": 2,
                "workers": 4,
                "epochs": 10,
            },
            "sentiment": {"learning_rate": 0.01, "epochs": 100, "batch_size": 32},
        },
        "cv": {
            "hand_detection": {
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            "size_thresholds": {"small_max": 150, "medium_max": 200},
        },
        "recommendation": {"min_confidence": 0.6, "max_results": 5},
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory path."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def mock_word_vectors():
    """Mock word vectors for embedding tests."""
    np.random.seed(42)
    vocab_size = 1000
    vector_size = 100

    # Generate random word vectors
    vectors = np.random.randn(vocab_size, vector_size).astype(np.float32)

    # Normalize vectors (common in word embeddings)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-8)

    return vectors


@pytest.fixture
def sample_tokenized_texts():
    """Sample tokenized texts for embedding training."""
    return [
        ["love", "amazing", "product"],
        ["terrible", "disappointing"],
        ["okay", "nothing", "special"],
        ["fantastic", "experience"],
        ["worst", "purchase"],
        ["great", "quality", "fast", "shipping"],
        ["not", "satisfied", "service"],
        ["excellent", "customer", "support"],
        ["poor", "build", "quality"],
        ["highly", "recommended"],
    ]


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "nlp: NLP module tests")
    config.addinivalue_line("markers", "cv: Computer vision tests")
    config.addinivalue_line("markers", "recommendation: Recommendation engine tests")
