#!/usr/bin/env python3
"""
Train sentiment classification model.

This script trains a logistic regression model for sentiment analysis
using preprocessed tweets and Word2Vec embeddings.

Usage:
    python scripts/train_sentiment_model.py --tweets data/raw/tweets.csv \
                                           --embeddings data/models/word2vec.model \
                                           --output data/models/sentiment_model.pkl \
                                           --config config/model_config.yaml
"""

import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp.preprocessor import TextPreprocessor
from nlp.embeddings import Word2VecEmbeddings
from nlp.sentiment_model import LogisticRegression
from nlp.utils import split_data, calculate_metrics, encode_sentiment_labels
from utils.logger import setup_logger
from utils.validators import validate_config_file
from utils.metrics import plot_training_history, plot_confusion_matrix
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def load_labeled_tweets(
    file_path: str, text_column: str = "text", label_column: str = "sentiment"
) -> pd.DataFrame:
    """Load labeled tweet data."""
    try:
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in data")

        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in data")

        # Remove rows with missing data
        df = df.dropna(subset=[text_column, label_column])

        return df
    except Exception as e:
        raise RuntimeError(f"Error loading tweet data: {e}")


def create_feature_matrix(
    texts: list, embeddings: Word2VecEmbeddings, preprocessor: TextPreprocessor
) -> np.ndarray:
    """Create feature matrix from texts using embeddings."""
    feature_vectors = []

    for text in texts:
        tokens = preprocessor.preprocess(text)
        vector = embeddings.get_sentence_vector(tokens)
        feature_vectors.append(vector)

    return np.array(feature_vectors)


def main():
    parser = argparse.ArgumentParser(description="Train sentiment classification model")
    parser.add_argument(
        "--tweets", "-t", required=True, help="Path to labeled tweets CSV file"
    )
    parser.add_argument(
        "--embeddings", "-e", required=True, help="Path to trained Word2Vec embeddings"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to save trained sentiment model"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--text-column", default="text", help="Name of column containing tweet text"
    )
    parser.add_argument(
        "--label-column",
        default="sentiment",
        help="Name of column containing sentiment labels",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save training plots")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(__name__)
    if args.verbose:
        logger.setLevel("DEBUG")

    logger.info("Starting sentiment model training script")

    try:
        # Validate inputs
        if not os.path.exists(args.tweets):
            raise FileNotFoundError(f"Tweets file not found: {args.tweets}")

        if not os.path.exists(args.embeddings):
            raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")

        if not validate_config_file(args.config):
            raise FileNotFoundError(f"Config file not found or invalid: {args.config}")

        # Load configuration
        config = load_config(args.config)
        sentiment_config = config.get("nlp", {}).get("sentiment", {})

        logger.info(f"Loaded configuration from {args.config}")

        # Load data
        logger.info(f"Loading labeled tweets from {args.tweets}")
        df = load_labeled_tweets(args.tweets, args.text_column, args.label_column)
        logger.info(f"Loaded {len(df)} labeled tweets")

        # Display label distribution
        label_counts = df[args.label_column].value_counts()
        logger.info("Label distribution:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} ({count/len(df):.1%})")

        # Load embeddings
        logger.info(f"Loading Word2Vec embeddings from {args.embeddings}")
        embeddings = Word2VecEmbeddings()
        embeddings.load_model(args.embeddings)
        logger.info(
            f"Loaded embeddings with vocabulary size: {embeddings.get_vocabulary_size()}"
        )

        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        logger.info("Initialized text preprocessor")

        # Prepare data
        logger.info("Creating feature matrix...")
        texts = df[args.text_column].tolist()
        labels = df[args.label_column].tolist()

        # Encode labels
        y, label_mapping = encode_sentiment_labels(labels)
        logger.info(f"Label mapping: {label_mapping}")

        # Create feature matrix
        X = create_feature_matrix(texts, embeddings, preprocessor)
        logger.info(f"Created feature matrix with shape: {X.shape}")

        # Check for zero variance features (optional)
        feature_std = np.std(X, axis=0)
        zero_var_features = np.sum(feature_std == 0)
        if zero_var_features > 0:
            logger.warning(f"Found {zero_var_features} zero-variance features")

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
        logger.info(
            f"Split data: {len(X_train)} training, {len(X_test)} testing samples"
        )

        # Initialize model
        model = LogisticRegression()
        logger.info("Initialized logistic regression model")

        # Train model
        learning_rate = sentiment_config.get("learning_rate", 0.01)
        epochs = sentiment_config.get("epochs", 100)

        logger.info(f"Training model with lr={learning_rate}, epochs={epochs}")

        model.train(
            X_train,
            y_train,
            learning_rate=learning_rate,
            epochs=epochs,
            verbose=args.verbose,
        )

        # Evaluate model
        logger.info("Evaluating model on test set...")

        # Test set evaluation
        test_metrics = model.evaluate(X_test, y_test)
        logger.info("Test Set Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Detailed metrics
        y_pred = model.predict(X_test)
        detailed_metrics = calculate_metrics(y_test, y_pred)

        logger.info("Detailed Test Metrics:")
        for metric, value in detailed_metrics.items():
            logger.info(f"  {metric}: {value}")

        # Create output directory
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save model
        logger.info(f"Saving model to {args.output}")
        model.save_model(args.output)

        # Save training metadata
        metadata = {
            "tweets_file": args.tweets,
            "embeddings_file": args.embeddings,
            "total_samples": len(df),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_dim": X.shape[1],
            "label_mapping": label_mapping,
            "training_config": {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "test_size": args.test_size,
            },
            "test_metrics": test_metrics,
            "detailed_metrics": detailed_metrics,
        }

        metadata_path = args.output.replace(".pkl", "_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")

        # Save plots if requested
        if args.save_plots:
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Training history plot
            history = model.get_training_history()
            if history["loss"] and history["accuracy"]:
                history_path = os.path.join(plots_dir, "training_history.png")
                plot_training_history(history, history_path)
                logger.info(f"Saved training history plot to {history_path}")

            # Confusion matrix
            cm_path = os.path.join(plots_dir, "confusion_matrix.png")
            labels = list(label_mapping.keys())
            plot_confusion_matrix(y_test, y_pred, labels=labels, save_path=cm_path)
            logger.info(f"Saved confusion matrix to {cm_path}")

        # Test with example sentences
        logger.info("Testing model with example sentences:")
        test_sentences = [
            "I love this amazing product!",
            "This is terrible and disappointing",
            "It's okay, nothing special",
            "Absolutely fantastic experience!",
            "Worst purchase ever made",
        ]

        for sentence in test_sentences:
            tokens = preprocessor.preprocess(sentence)
            vector = embeddings.get_sentence_vector(tokens)
            vector = vector.reshape(1, -1)

            prob = model.predict_proba(vector)[0]
            pred = model.predict(vector)[0]

            sentiment_label = "positive" if pred == 1 else "negative"
            confidence = prob if pred == 1 else (1 - prob)

            logger.info(
                f"  '{sentence}' -> {sentiment_label} (confidence: {confidence:.3f})"
            )

        # Training summary
        logger.info("Training Summary:")
        logger.info(f"  Model: Logistic Regression from scratch")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Feature dimension: {X.shape[1]}")
        logger.info(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test loss: {test_metrics['loss']:.4f}")

        logger.info("Sentiment model training completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
