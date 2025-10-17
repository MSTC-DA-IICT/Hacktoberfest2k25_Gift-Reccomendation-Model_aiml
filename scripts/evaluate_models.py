#!/usr/bin/env python3
"""
Evaluate trained models on test data.

This script evaluates both NLP and CV models on test datasets
and generates comprehensive performance reports.

Usage:
    python scripts/evaluate_models.py --nlp-data data/test/tweets.csv \
                                     --cv-data data/test/hand_images \
                                     --models-dir data/models \
                                     --output reports/evaluation_report.txt
"""

import argparse
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp.preprocessor import TextPreprocessor
from nlp.embeddings import Word2VecEmbeddings
from nlp.sentiment_model import LogisticRegression
from nlp.utils import calculate_metrics, encode_sentiment_labels
from cv.hand_detector import HandDetector
from cv.size_estimator import HandSizeEstimator
from cv.utils import load_image, batch_process_images
from recommendation.gift_database import GiftDatabase
from recommendation.engine import RecommendationEngine
from utils.logger import setup_logger
from utils.metrics import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    calculate_recommendation_metrics,
    calculate_hand_detection_metrics,
    generate_performance_report,
)


def evaluate_nlp_model(test_data_path: str, models_dir: str, logger) -> Dict:
    """Evaluate NLP sentiment model."""
    try:
        logger.info("Evaluating NLP sentiment model...")

        # Load test data
        df = pd.read_csv(test_data_path)
        texts = df["text"].tolist()
        true_labels = df["sentiment"].tolist()

        # Encode labels
        y_true, label_mapping = encode_sentiment_labels(true_labels)

        # Load models
        embeddings_path = os.path.join(models_dir, "word2vec.model")
        sentiment_path = os.path.join(models_dir, "sentiment_model.pkl")

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings model not found: {embeddings_path}")

        if not os.path.exists(sentiment_path):
            raise FileNotFoundError(f"Sentiment model not found: {sentiment_path}")

        # Load embeddings
        embeddings = Word2VecEmbeddings()
        embeddings.load_model(embeddings_path)

        # Load sentiment model
        sentiment_model = LogisticRegression()
        sentiment_model.load_model(sentiment_path)

        # Initialize preprocessor
        preprocessor = TextPreprocessor()

        # Make predictions
        y_pred = []
        y_scores = []

        logger.info(f"Processing {len(texts)} test samples...")

        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(texts)} samples")

            # Preprocess text
            tokens = preprocessor.preprocess(text)

            # Get sentence vector
            vector = embeddings.get_sentence_vector(tokens)
            vector = vector.reshape(1, -1)

            # Predict
            pred = sentiment_model.predict(vector)[0]
            prob = sentiment_model.predict_proba(vector)[0]

            y_pred.append(pred)
            y_scores.append(prob)

        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        # Calculate metrics
        metrics = calculate_precision_recall_f1(y_true, y_pred)

        # Add ROC AUC if available
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_true, y_scores)
            metrics["roc_auc"] = auc
        except ImportError:
            logger.warning("sklearn not available for ROC AUC calculation")

        # Add sample predictions for analysis
        sample_predictions = []
        for i in range(min(10, len(texts))):
            sample_predictions.append(
                {
                    "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                    "true_label": true_labels[i],
                    "predicted_label": "positive" if y_pred[i] == 1 else "negative",
                    "confidence": y_scores[i] if y_pred[i] == 1 else (1 - y_scores[i]),
                }
            )

        results = {
            "model_type": "NLP Sentiment Analysis",
            "test_samples": len(texts),
            "metrics": metrics,
            "sample_predictions": sample_predictions,
            "label_mapping": label_mapping,
        }

        logger.info("NLP model evaluation completed")
        return results

    except Exception as e:
        logger.error(f"Error evaluating NLP model: {e}")
        return {"model_type": "NLP Sentiment Analysis", "error": str(e)}


def evaluate_cv_model(test_images_dir: str, logger) -> Dict:
    """Evaluate computer vision hand detection model."""
    try:
        logger.info("Evaluating CV hand detection model...")

        # Find test images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = []

        for ext in image_extensions:
            pattern = f"*{ext}"
            images = Path(test_images_dir).glob(pattern)
            image_paths.extend(list(images))

        if not image_paths:
            raise FileNotFoundError(f"No images found in {test_images_dir}")

        logger.info(f"Found {len(image_paths)} test images")

        # Initialize models
        detector = HandDetector()
        estimator = HandSizeEstimator()

        # Process images
        detection_results = []
        size_estimations = []

        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(image_paths)} images")

            try:
                # Load image
                image = load_image(str(image_path))
                if image is None:
                    continue

                # Detect hand
                landmarks = detector.get_landmarks(image)

                if landmarks:
                    # Estimate size
                    measurements = estimator.calculate_all_measurements(landmarks)
                    size_class = estimator.classify_size(measurements)

                    detection_results.append(
                        {
                            "image_path": str(image_path),
                            "detected": True,
                            "handedness": landmarks.get("handedness", "Unknown"),
                            "confidence": landmarks.get("confidence", 0.0),
                            "measurements": measurements,
                            "size_class": size_class,
                        }
                    )

                    size_estimations.append(size_class)
                else:
                    detection_results.append(
                        {"image_path": str(image_path), "detected": False}
                    )

            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
                detection_results.append(
                    {"image_path": str(image_path), "error": str(e)}
                )

        # Calculate metrics
        total_images = len(image_paths)
        successful_detections = sum(
            1 for r in detection_results if r.get("detected", False)
        )
        detection_rate = successful_detections / total_images if total_images > 0 else 0

        # Size distribution
        size_distribution = {}
        for size in size_estimations:
            size_distribution[size] = size_distribution.get(size, 0) + 1

        # Average confidence
        confidences = [
            r.get("confidence", 0) for r in detection_results if r.get("detected")
        ]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Sample results
        sample_results = detection_results[:10]  # First 10 results

        metrics = {
            "detection_rate": detection_rate,
            "average_confidence": avg_confidence,
            "successful_detections": successful_detections,
            "total_images": total_images,
        }

        results = {
            "model_type": "CV Hand Detection",
            "test_samples": total_images,
            "metrics": metrics,
            "size_distribution": size_distribution,
            "sample_results": sample_results,
        }

        logger.info("CV model evaluation completed")
        return results

    except Exception as e:
        logger.error(f"Error evaluating CV model: {e}")
        return {"model_type": "CV Hand Detection", "error": str(e)}


def evaluate_recommendation_system(models_dir: str, config_dir: str, logger) -> Dict:
    """Evaluate recommendation system."""
    try:
        logger.info("Evaluating recommendation system...")

        # Load gift database
        gift_config_path = os.path.join(config_dir, "gift_mapping.json")
        if not os.path.exists(gift_config_path):
            raise FileNotFoundError(f"Gift config not found: {gift_config_path}")

        gift_db = GiftDatabase(gift_config_path)
        engine = RecommendationEngine(gift_db)

        # Test scenarios
        test_scenarios = [
            {
                "sentiment": 0.9,
                "hand_size": "small",
                "description": "Very positive, small hands",
            },
            {
                "sentiment": 0.8,
                "hand_size": "medium",
                "description": "Positive, medium hands",
            },
            {
                "sentiment": 0.7,
                "hand_size": "large",
                "description": "Positive, large hands",
            },
            {
                "sentiment": 0.3,
                "hand_size": "small",
                "description": "Negative, small hands",
            },
            {
                "sentiment": 0.2,
                "hand_size": "medium",
                "description": "Negative, medium hands",
            },
            {
                "sentiment": 0.1,
                "hand_size": "large",
                "description": "Very negative, large hands",
            },
            {
                "sentiment": 0.5,
                "hand_size": "medium",
                "description": "Neutral, medium hands",
            },
        ]

        recommendation_results = []
        all_confidences = []

        for scenario in test_scenarios:
            recommendations = engine.recommend(
                sentiment=scenario["sentiment"],
                hand_size=scenario["hand_size"],
                max_results=3,
            )

            confidences = [rec["confidence"] for rec in recommendations]
            all_confidences.extend(confidences)

            scenario_result = {
                "scenario": scenario["description"],
                "sentiment": scenario["sentiment"],
                "hand_size": scenario["hand_size"],
                "recommendation_count": len(recommendations),
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "recommendations": [
                    {
                        "name": rec.get("name", "Unknown"),
                        "category": rec.get("category", "Unknown"),
                        "confidence": rec.get("confidence", 0.0),
                    }
                    for rec in recommendations[:3]  # Top 3
                ],
            }

            recommendation_results.append(scenario_result)

        # Overall metrics
        metrics = {
            "scenarios_tested": len(test_scenarios),
            "avg_recommendations_per_scenario": np.mean(
                [r["recommendation_count"] for r in recommendation_results]
            ),
            "overall_avg_confidence": (
                np.mean(all_confidences) if all_confidences else 0.0
            ),
            "min_confidence": np.min(all_confidences) if all_confidences else 0.0,
            "max_confidence": np.max(all_confidences) if all_confidences else 0.0,
        }

        # Database statistics
        db_stats = gift_db.get_statistics()

        results = {
            "model_type": "Recommendation System",
            "test_scenarios": len(test_scenarios),
            "metrics": metrics,
            "database_stats": db_stats,
            "scenario_results": recommendation_results,
        }

        logger.info("Recommendation system evaluation completed")
        return results

    except Exception as e:
        logger.error(f"Error evaluating recommendation system: {e}")
        return {"model_type": "Recommendation System", "error": str(e)}


def generate_evaluation_report(results: List[Dict], output_path: str, logger) -> None:
    """Generate comprehensive evaluation report."""
    try:
        logger.info(f"Generating evaluation report: {output_path}")

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("Gift Recommendation Platform - Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            # Summary
            f.write("EVALUATION SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Models Evaluated: {len(results)}\n\n")

            # Individual model results
            for i, result in enumerate(results, 1):
                model_type = result.get("model_type", "Unknown Model")
                f.write(f"{i}. {model_type.upper()}\n")
                f.write("-" * (len(model_type) + 4) + "\n")

                if "error" in result:
                    f.write(f"Error: {result['error']}\n\n")
                    continue

                # Write metrics
                if "metrics" in result:
                    f.write("Metrics:\n")
                    for metric, value in result["metrics"].items():
                        if isinstance(value, float):
                            f.write(f"  {metric}: {value:.4f}\n")
                        else:
                            f.write(f"  {metric}: {value}\n")
                    f.write("\n")

                # Model-specific details
                if (
                    model_type == "NLP Sentiment Analysis"
                    and "sample_predictions" in result
                ):
                    f.write("Sample Predictions:\n")
                    for pred in result["sample_predictions"][:5]:
                        f.write(f"  Text: {pred['text']}\n")
                        f.write(
                            f"  True: {pred['true_label']}, Predicted: {pred['predicted_label']}\n"
                        )
                        f.write(f"  Confidence: {pred['confidence']:.3f}\n\n")

                elif (
                    model_type == "CV Hand Detection" and "size_distribution" in result
                ):
                    f.write("Hand Size Distribution:\n")
                    for size, count in result["size_distribution"].items():
                        f.write(f"  {size}: {count}\n")
                    f.write("\n")

                elif (
                    model_type == "Recommendation System"
                    and "scenario_results" in result
                ):
                    f.write("Sample Scenarios:\n")
                    for scenario in result["scenario_results"][:3]:
                        f.write(f"  Scenario: {scenario['scenario']}\n")
                        f.write(
                            f"  Recommendations: {scenario['recommendation_count']}\n"
                        )
                        f.write(f"  Avg Confidence: {scenario['avg_confidence']:.3f}\n")
                        if scenario["recommendations"]:
                            top_rec = scenario["recommendations"][0]
                            f.write(
                                f"  Top Recommendation: {top_rec['name']} ({top_rec['confidence']:.3f})\n"
                            )
                        f.write("\n")

                f.write("\n")

            # Overall assessment
            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 18 + "\n")

            successful_models = [r for r in results if "error" not in r]
            failed_models = [r for r in results if "error" in r]

            f.write(
                f"Successful Evaluations: {len(successful_models)}/{len(results)}\n"
            )

            if failed_models:
                f.write("Failed Evaluations:\n")
                for failed in failed_models:
                    f.write(
                        f"  - {failed.get('model_type', 'Unknown')}: {failed.get('error', 'Unknown error')}\n"
                    )

            f.write("\n")

            # Recommendations for improvements
            f.write("RECOMMENDATIONS FOR IMPROVEMENT\n")
            f.write("-" * 32 + "\n")

            for result in successful_models:
                model_type = result.get("model_type", "Unknown")
                metrics = result.get("metrics", {})

                if model_type == "NLP Sentiment Analysis":
                    accuracy = metrics.get("accuracy", 0)
                    if accuracy < 0.8:
                        f.write(
                            f"- NLP Model: Consider increasing training data or tuning hyperparameters (current accuracy: {accuracy:.3f})\n"
                        )

                elif model_type == "CV Hand Detection":
                    detection_rate = metrics.get("detection_rate", 0)
                    if detection_rate < 0.85:
                        f.write(
                            f"- CV Model: Consider improving hand detection threshold or adding data augmentation (current detection rate: {detection_rate:.3f})\n"
                        )

                elif model_type == "Recommendation System":
                    avg_confidence = metrics.get("overall_avg_confidence", 0)
                    if avg_confidence < 0.6:
                        f.write(
                            f"- Recommendation System: Consider expanding gift database or refining recommendation rules (current avg confidence: {avg_confidence:.3f})\n"
                        )

        logger.info(f"Evaluation report saved to {output_path}")

    except Exception as e:
        logger.error(f"Error generating report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test data")
    parser.add_argument(
        "--nlp-data", help="Path to NLP test data (CSV with text and sentiment columns)"
    )
    parser.add_argument("--cv-data", help="Path to CV test data directory (images)")
    parser.add_argument(
        "--models-dir",
        default="data/models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="reports/evaluation_report.txt",
        help="Path to save evaluation report",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(__name__)
    if args.verbose:
        logger.setLevel("DEBUG")

    logger.info("Starting model evaluation script")

    try:
        # Validate inputs
        if not os.path.exists(args.models_dir):
            os.makedirs(args.models_dir, exist_ok=True)
            logger.warning(f"Models directory created: {args.models_dir}")

        if not os.path.exists(args.config_dir):
            raise FileNotFoundError(f"Config directory not found: {args.config_dir}")

        results = []

        # Evaluate NLP model
        if args.nlp_data:
            if os.path.exists(args.nlp_data):
                nlp_results = evaluate_nlp_model(args.nlp_data, args.models_dir, logger)
                results.append(nlp_results)
            else:
                logger.warning(f"NLP test data not found: {args.nlp_data}")

        # Evaluate CV model
        if args.cv_data:
            if os.path.exists(args.cv_data):
                cv_results = evaluate_cv_model(args.cv_data, logger)
                results.append(cv_results)
            else:
                logger.warning(f"CV test data not found: {args.cv_data}")

        # Always evaluate recommendation system (uses synthetic data)
        rec_results = evaluate_recommendation_system(
            args.models_dir, args.config_dir, logger
        )
        results.append(rec_results)

        if not results:
            logger.warning("No models were evaluated")
            return 1

        # Generate report
        generate_evaluation_report(results, args.output, logger)

        logger.info("Model evaluation completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
