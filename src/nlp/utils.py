"""
Utility functions for NLP module.

This module provides helper functions for data loading, preprocessing,
and evaluation of NLP models.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_tweet_data(
    path: str, text_column: str = "text", label_column: str = "sentiment"
) -> pd.DataFrame:
    """
    Load tweet data from CSV file.

    Parameters
    ----------
    path : str
        Path to CSV file
    text_column : str, default='text'
        Name of column containing tweet text
    label_column : str, default='sentiment'
        Name of column containing sentiment labels

    Returns
    -------
    pd.DataFrame
        DataFrame with tweet data

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If required columns are missing

    Examples
    --------
    >>> df = load_tweet_data('tweets.csv')
    >>> print(df.head())
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} tweets from {path}")

        # Check required columns exist
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")

        # Remove rows with missing text or labels
        initial_len = len(df)
        df = df.dropna(subset=[text_column, label_column])
        final_len = len(df)

        if final_len < initial_len:
            logger.warning(
                f"Dropped {initial_len - final_len} rows with missing values"
            )

        return df

    except Exception as e:
        logger.error(f"Error loading tweet data: {e}")
        raise


def split_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> X_train, X_test, y_train, y_test = split_data(X, y)
    >>> print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    Train size: 80, Test size: 20
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # Set random seed
    np.random.seed(random_state)

    # Create shuffled indices
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    # Calculate split point
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    logger.info(f"Split data: {n_train} training samples, {n_test} test samples")

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels

    Returns
    -------
    Dict[str, float]
        Dictionary containing accuracy, precision, recall, and F1-score

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 0, 0])
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(metrics)
    {'accuracy': 0.8, 'precision': 1.0, 'recall': 0.667, 'f1': 0.8}
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Handle division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    return metrics


def encode_sentiment_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode string sentiment labels to binary integers.

    Parameters
    ----------
    labels : List[str]
        List of sentiment labels (e.g., ['positive', 'negative', 'positive'])

    Returns
    -------
    Tuple[np.ndarray, Dict[str, int]]
        Encoded labels and label mapping

    Examples
    --------
    >>> labels = ['positive', 'negative', 'positive', 'negative']
    >>> encoded, mapping = encode_sentiment_labels(labels)
    >>> print(encoded)
    [1 0 1 0]
    >>> print(mapping)
    {'negative': 0, 'positive': 1}
    """
    # Create label mapping
    unique_labels = sorted(list(set(labels)))

    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected 2 unique labels, got {len(unique_labels)}: {unique_labels}"
        )

    # Map to 0 (negative) and 1 (positive)
    # Assume alphabetical order: negative comes before positive
    label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}

    # Encode labels
    encoded_labels = np.array([label_mapping[label] for label in labels])

    logger.info(f"Encoded {len(labels)} labels with mapping: {label_mapping}")

    return encoded_labels, label_mapping


def create_feature_matrix(
    tokenized_texts: List[List[str]], embeddings_model
) -> np.ndarray:
    """
    Create feature matrix from tokenized texts using embeddings.

    Parameters
    ----------
    tokenized_texts : List[List[str]]
        List of tokenized texts
    embeddings_model : Word2VecEmbeddings
        Trained embeddings model

    Returns
    -------
    np.ndarray
        Feature matrix with shape (n_texts, embedding_dim)

    Examples
    --------
    >>> texts = [['happy', 'day'], ['sad', 'news']]
    >>> # Assuming embeddings_model is trained
    >>> X = create_feature_matrix(texts, embeddings_model)
    >>> print(X.shape)
    (2, 100)
    """
    if not embeddings_model.is_trained:
        raise ValueError("Embeddings model must be trained first")

    feature_vectors = []
    for tokens in tokenized_texts:
        sentence_vector = embeddings_model.get_sentence_vector(tokens)
        feature_vectors.append(sentence_vector)

    X = np.array(feature_vectors)
    logger.info(f"Created feature matrix with shape {X.shape}")

    return X


def preprocess_texts_for_training(texts: List[str], preprocessor) -> List[List[str]]:
    """
    Preprocess texts for model training.

    Parameters
    ----------
    texts : List[str]
        Raw text data
    preprocessor : TextPreprocessor
        Text preprocessor instance

    Returns
    -------
    List[List[str]]
        List of tokenized and preprocessed texts

    Examples
    --------
    >>> texts = ["I love this!", "This is terrible :("]
    >>> preprocessor = TextPreprocessor()
    >>> tokenized = preprocess_texts_for_training(texts, preprocessor)
    >>> print(tokenized)
    [['love'], ['terrible']]
    """
    tokenized_texts = []
    skipped_count = 0

    for text in texts:
        tokens = preprocessor.preprocess(text)

        # Skip empty tokenized texts
        if not tokens:
            skipped_count += 1
            tokens = ["<empty>"]  # Placeholder for empty texts

        tokenized_texts.append(tokens)

    if skipped_count > 0:
        logger.warning(f"Found {skipped_count} texts that resulted in empty tokens")

    logger.info(f"Preprocessed {len(texts)} texts")
    return tokenized_texts


def balance_dataset(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset by undersampling the majority class.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    random_state : int, default=42
        Random seed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Balanced X and y arrays
    """
    np.random.seed(random_state)

    unique_labels, counts = np.unique(y, return_counts=True)
    min_count = min(counts)

    balanced_indices = []

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        selected_indices = np.random.choice(label_indices, min_count, replace=False)
        balanced_indices.extend(selected_indices)

    # Shuffle the final indices
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    logger.info(f"Balanced dataset from {len(X)} to {len(X_balanced)} samples")

    return X_balanced, y_balanced
