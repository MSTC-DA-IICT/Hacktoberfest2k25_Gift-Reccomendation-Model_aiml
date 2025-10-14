"""
Performance metrics and evaluation utilities.

This module provides functions to calculate various metrics for
model evaluation and system performance monitoring.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

logger = logging.getLogger(__name__)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

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

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 0, 0])
    >>> accuracy = calculate_accuracy(y_true, y_pred)
    >>> print(f"Accuracy: {accuracy:.2f}")
    0.80
    """
    try:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        if len(y_true) == 0:
            return 0.0

        correct = np.sum(y_true == y_pred)
        accuracy = correct / len(y_true)

        logger.debug(f"Calculated accuracy: {accuracy:.4f} ({correct}/{len(y_true)})")
        return accuracy

    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return 0.0


def calculate_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1-score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels

    Returns
    -------
    Dict[str, float]
        Dictionary containing precision, recall, and f1 scores

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 0, 0])
    >>> metrics = calculate_precision_recall_f1(y_true, y_pred)
    >>> print(f"F1: {metrics['f1']:.2f}")
    """
    try:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives

        # Calculate metrics with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        metrics = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'specificity': round(specificity, 4),
            'accuracy': round(accuracy, 4),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

        logger.debug(f"Calculated metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating precision/recall/F1: {e}")
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'specificity': 0.0, 'accuracy': 0.0,
            'true_positives': 0, 'true_negatives': 0,
            'false_positives': 0, 'false_negatives': 0
        }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         normalize: bool = False) -> Optional[plt.Figure]:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : Optional[List[str]]
        Class labels for display
    save_path : Optional[str]
        Path to save the plot
    normalize : bool, default=False
        Whether to normalize the confusion matrix

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure object or None if error

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 0, 0])
    >>> fig = plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'])
    """
    try:
        # Calculate confusion matrix
        cm = sklearn_confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=labels or ['Class 0', 'Class 1'],
                   yticklabels=labels or ['Class 0', 'Class 1'],
                   ax=ax)

        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        logger.debug("Generated confusion matrix plot")
        return fig

    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        return None


def calculate_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate ROC AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Prediction scores/probabilities

    Returns
    -------
    float
        ROC AUC score

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> auc = calculate_roc_auc(y_true, y_scores)
    >>> print(f"AUC: {auc:.2f}")
    """
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_scores)
        logger.debug(f"Calculated ROC AUC: {auc:.4f}")
        return auc

    except ImportError:
        logger.error("sklearn not available for ROC AUC calculation")
        return 0.5  # Random classifier baseline
    except Exception as e:
        logger.error(f"Error calculating ROC AUC: {e}")
        return 0.5


def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot training history (loss and accuracy curves).

    Parameters
    ----------
    history : Dict[str, List[float]]
        Training history dictionary with 'loss' and 'accuracy' keys
    save_path : Optional[str]
        Path to save the plot

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure object or None if error

    Examples
    --------
    >>> history = {'loss': [0.8, 0.6, 0.4], 'accuracy': [0.6, 0.7, 0.8]}
    >>> fig = plot_training_history(history, 'training_curves.png')
    """
    try:
        if not isinstance(history, dict):
            raise ValueError("History must be a dictionary")

        required_keys = ['loss', 'accuracy']
        for key in required_keys:
            if key not in history:
                raise ValueError(f"History missing required key: {key}")

        epochs = range(1, len(history['loss']) + 1)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(epochs, history['accuracy'], 'r-', label='Training Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        logger.debug("Generated training history plot")
        return fig

    except Exception as e:
        logger.error(f"Error plotting training history: {e}")
        return None


def calculate_recommendation_metrics(recommendations: List[Dict],
                                   ground_truth: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Calculate metrics for recommendation system performance.

    Parameters
    ----------
    recommendations : List[Dict]
        List of recommendation dictionaries with 'confidence' scores
    ground_truth : Optional[List[int]]
        Ground truth relevance scores (1 for relevant, 0 for not)

    Returns
    -------
    Dict[str, float]
        Recommendation metrics

    Examples
    --------
    >>> recommendations = [{'confidence': 0.9}, {'confidence': 0.7}]
    >>> metrics = calculate_recommendation_metrics(recommendations)
    >>> print(f"Average confidence: {metrics['avg_confidence']:.2f}")
    """
    try:
        if not recommendations:
            return {'count': 0, 'avg_confidence': 0.0}

        # Basic metrics
        confidences = [rec.get('confidence', 0.0) for rec in recommendations]

        metrics = {
            'count': len(recommendations),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'std_confidence': np.std(confidences),
            'high_confidence_ratio': sum(1 for c in confidences if c >= 0.7) / len(confidences)
        }

        # If ground truth is provided, calculate precision@k
        if ground_truth is not None:
            if len(ground_truth) != len(recommendations):
                logger.warning("Ground truth length doesn't match recommendations")
            else:
                # Calculate precision at different k values
                for k in [1, 3, 5]:
                    if k <= len(recommendations):
                        relevant_at_k = sum(ground_truth[:k])
                        precision_at_k = relevant_at_k / k
                        metrics[f'precision_at_{k}'] = precision_at_k

                # Calculate NDCG@5 (simplified)
                dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ground_truth[:5]))
                ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(5, sum(ground_truth))))
                ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
                metrics['ndcg_at_5'] = ndcg

        logger.debug(f"Calculated recommendation metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating recommendation metrics: {e}")
        return {'count': 0, 'avg_confidence': 0.0}


def calculate_hand_detection_metrics(detected_landmarks: List[Optional[Dict]],
                                   ground_truth_landmarks: List[Optional[Dict]]) -> Dict[str, float]:
    """
    Calculate metrics for hand detection performance.

    Parameters
    ----------
    detected_landmarks : List[Optional[Dict]]
        List of detected landmark dictionaries
    ground_truth_landmarks : List[Optional[Dict]]
        List of ground truth landmark dictionaries

    Returns
    -------
    Dict[str, float]
        Hand detection metrics

    Examples
    --------
    >>> detected = [{'landmarks': [...]}, None, {'landmarks': [...]}]
    >>> ground_truth = [{'landmarks': [...]}, {'landmarks': [...]}, None]
    >>> metrics = calculate_hand_detection_metrics(detected, ground_truth)
    >>> print(f"Detection rate: {metrics['detection_rate']:.2f}")
    """
    try:
        if len(detected_landmarks) != len(ground_truth_landmarks):
            raise ValueError("Detected and ground truth lists must have same length")

        total_samples = len(detected_landmarks)
        if total_samples == 0:
            return {}

        # Detection metrics
        true_positives = 0  # Hand present and detected
        false_positives = 0  # No hand but detected
        true_negatives = 0  # No hand and not detected
        false_negatives = 0  # Hand present but not detected

        landmark_errors = []  # For samples where both have detections

        for detected, ground_truth in zip(detected_landmarks, ground_truth_landmarks):
            has_gt = ground_truth is not None
            has_detection = detected is not None

            if has_gt and has_detection:
                true_positives += 1
                # Calculate landmark accuracy if both have landmarks
                if 'landmarks' in detected and 'landmarks' in ground_truth:
                    error = calculate_landmark_error(detected['landmarks'], ground_truth['landmarks'])
                    if error is not None:
                        landmark_errors.append(error)

            elif not has_gt and has_detection:
                false_positives += 1
            elif not has_gt and not has_detection:
                true_negatives += 1
            elif has_gt and not has_detection:
                false_negatives += 1

        # Calculate metrics
        detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        accuracy = (true_positives + true_negatives) / total_samples

        metrics = {
            'detection_rate': detection_rate,
            'precision': precision,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'total_samples': total_samples
        }

        # Add landmark accuracy if available
        if landmark_errors:
            metrics['avg_landmark_error'] = np.mean(landmark_errors)
            metrics['landmark_error_std'] = np.std(landmark_errors)

        logger.debug(f"Calculated hand detection metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating hand detection metrics: {e}")
        return {}


def calculate_landmark_error(detected_landmarks: List[Dict], ground_truth_landmarks: List[Dict]) -> Optional[float]:
    """
    Calculate average landmark position error.

    Parameters
    ----------
    detected_landmarks : List[Dict]
        Detected landmarks with 'x', 'y' coordinates
    ground_truth_landmarks : List[Dict]
        Ground truth landmarks with 'x', 'y' coordinates

    Returns
    -------
    Optional[float]
        Average Euclidean distance error or None if invalid
    """
    try:
        if len(detected_landmarks) != len(ground_truth_landmarks):
            return None

        errors = []
        for detected, gt in zip(detected_landmarks, ground_truth_landmarks):
            if 'x' in detected and 'y' in detected and 'x' in gt and 'y' in gt:
                dx = detected['x'] - gt['x']
                dy = detected['y'] - gt['y']
                error = np.sqrt(dx*dx + dy*dy)
                errors.append(error)

        return np.mean(errors) if errors else None

    except Exception as e:
        logger.error(f"Error calculating landmark error: {e}")
        return None


def generate_performance_report(metrics: Dict[str, Union[float, int]],
                              model_name: str = "Model") -> str:
    """
    Generate a formatted performance report.

    Parameters
    ----------
    metrics : Dict[str, Union[float, int]]
        Metrics dictionary
    model_name : str, default="Model"
        Name of the model being evaluated

    Returns
    -------
    str
        Formatted performance report

    Examples
    --------
    >>> metrics = {'accuracy': 0.85, 'f1': 0.82, 'precision': 0.80}
    >>> report = generate_performance_report(metrics, "Sentiment Model")
    >>> print(report)
    """
    try:
        report_lines = [
            f"Performance Report for {model_name}",
            "=" * (25 + len(model_name)),
            ""
        ]

        # Group metrics by category
        accuracy_metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        detection_metrics = ['detection_rate', 'avg_confidence', 'ndcg_at_5']

        # Add accuracy metrics
        if any(metric in metrics for metric in accuracy_metrics):
            report_lines.append("Classification Metrics:")
            for metric in accuracy_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, float):
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
                    else:
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {value}")
            report_lines.append("")

        # Add detection metrics
        if any(metric in metrics for metric in detection_metrics):
            report_lines.append("Detection/Recommendation Metrics:")
            for metric in detection_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, float):
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
                    else:
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {value}")
            report_lines.append("")

        # Add other metrics
        other_metrics = [k for k in metrics.keys()
                        if k not in accuracy_metrics and k not in detection_metrics]
        if other_metrics:
            report_lines.append("Other Metrics:")
            for metric in sorted(other_metrics):
                value = metrics[metric]
                if isinstance(value, float):
                    report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
                else:
                    report_lines.append(f"  {metric.replace('_', ' ').title()}: {value}")

        return "\n".join(report_lines)

    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return f"Error generating report for {model_name}"