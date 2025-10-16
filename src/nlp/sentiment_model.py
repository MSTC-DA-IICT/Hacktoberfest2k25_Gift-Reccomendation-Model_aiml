"""
Logistic Regression implementation for sentiment analysis.

This module implements logistic regression from scratch using only NumPy,
designed specifically for binary sentiment classification.
"""

import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LogisticRegression:
    """
    Logistic Regression implemented from scratch using NumPy.

    This class provides a complete implementation of logistic regression
    with gradient descent optimization for binary sentiment classification.
    """

    def __init__(self):
        """Initialize the logistic regression model."""
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.history: Dict[str, List[float]] = {"loss": [], "accuracy": []}
        self.is_fitted = False

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters
        ----------
        z : np.ndarray
            Input values

        Returns
        -------
        np.ndarray
            Sigmoid output values between 0 and 1

        Examples
        --------
        >>> model = LogisticRegression()
        >>> model.sigmoid(np.array([0, 1, -1]))
        array([0.5       , 0.73105858, 0.26894142])
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def initialize_weights(self, n_features: int) -> None:
        """
        Initialize weights and bias.

        Parameters
        ----------
        n_features : int
            Number of features in the input data
        """
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0

    def compute_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0 or 1)
        y_pred_proba : np.ndarray
            Predicted probabilities

        Returns
        -------
        float
            Binary cross-entropy loss

        Examples
        --------
        >>> model = LogisticRegression()
        >>> y_true = np.array([1, 0, 1])
        >>> y_pred = np.array([0.8, 0.2, 0.9])
        >>> loss = model.compute_loss(y_true, y_pred)
        >>> print(f"Loss: {loss:.4f}")
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

        # Binary cross-entropy loss
        loss = -np.mean(
            y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba)
        )
        return loss

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy score.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_pred : np.ndarray
            Predicted binary labels

        Returns
        -------
        float
            Accuracy score between 0 and 1
        """
        return np.mean(y_true == y_pred)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Train the logistic regression model using gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training features with shape (n_samples, n_features)
        y : np.ndarray
            Training labels with shape (n_samples,)
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        epochs : int, default=100
            Number of training epochs
        verbose : bool, default=True
            Whether to print training progress

        Raises
        ------
        ValueError
            If X and y have incompatible shapes

        Examples
        --------
        >>> model = LogisticRegression()
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> model.train(X, y, learning_rate=0.01, epochs=50)
        """
        # Validate input
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}"
            )

        n_samples, n_features = X.shape

        # Initialize weights
        self.initialize_weights(n_features)

        # Reset history
        self.history = {"loss": [], "accuracy": []}

        if verbose:
            logger.info(
                f"Starting training with {n_samples} samples, {n_features} features"
            )

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            y_pred_proba = self.sigmoid(z)

            # Compute loss
            loss = self.compute_loss(y, y_pred_proba)

            # Compute predictions for accuracy
            y_pred = (y_pred_proba >= 0.5).astype(int)
            accuracy = self.compute_accuracy(y, y_pred)

            # Store metrics
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)

            # Backward pass - compute gradients
            dw = (1 / n_samples) * X.T.dot(y_pred_proba - y)
            db = (1 / n_samples) * np.sum(y_pred_proba - y)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                )

        self.is_fitted = True

        if verbose:
            final_loss = self.history["loss"][-1]
            final_accuracy = self.history["accuracy"][-1]
            logger.info(
                f"Training completed - Final Loss: {final_loss:.4f}, Final Accuracy: {final_accuracy:.4f}"
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted probabilities with shape (n_samples,)

        Raises
        ------
        ValueError
            If model hasn't been trained yet

        Examples
        --------
        >>> model = LogisticRegression()
        >>> # After training...
        >>> X_test = np.random.randn(10, 5)
        >>> probabilities = model.predict_proba(X_test)
        >>> print(probabilities.shape)
        (10,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        z = X.dot(self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1) with shape (n_samples,)

        Examples
        --------
        >>> model = LogisticRegression()
        >>> # After training...
        >>> X_test = np.random.randn(10, 5)
        >>> predictions = model.predict(X_test)
        >>> print(predictions)
        [1 0 1 1 0 0 1 0 1 0]
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance based on absolute weight values.

        Returns
        -------
        Optional[np.ndarray]
            Feature importance scores or None if model not trained
        """
        if not self.is_fitted or self.weights is None:
            return None

        return np.abs(self.weights)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str
            Path to save the model

        Raises
        ------
        ValueError
            If model hasn't been trained yet
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "weights": self.weights,
            "bias": self.bias,
            "history": self.history,
            "is_fitted": self.is_fitted,
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model

        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        """
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.weights = model_data["weights"]
            self.bias = model_data["bias"]
            self.history = model_data["history"]
            self.is_fitted = model_data["is_fitted"]

            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary containing 'loss' and 'accuracy' histories
        """
        return self.history.copy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True test labels

        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        metrics = {
            "accuracy": self.compute_accuracy(y, y_pred),
            "loss": self.compute_loss(y, y_pred_proba),
        }

        return metrics
