#!/usr/bin/env python3
"""
Train Word2Vec embeddings on tweet corpus.

This script trains Word2Vec embeddings on a corpus of preprocessed tweets
for use in sentiment analysis.

Usage:
    python scripts/train_embeddings.py --input data/raw/tweets.csv \
                                       --output data/models/word2vec.model \
                                       --config config/model_config.yaml
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp.preprocessor import TextPreprocessor
from nlp.embeddings import Word2VecEmbeddings
from utils.logger import setup_logger
from utils.validators import validate_config_file
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


def load_tweet_data(file_path: str, text_column: str = "text") -> pd.DataFrame:
    """Load tweet data from CSV file."""
    try:
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in data")

        # Remove rows with missing text
        df = df.dropna(subset=[text_column])

        return df
    except Exception as e:
        raise RuntimeError(f"Error loading tweet data: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec embeddings on tweet corpus"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input CSV file with tweets"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to save trained Word2Vec model"
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
        "--min-samples",
        type=int,
        default=100,
        help="Minimum number of samples required for training",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(__name__)
    if args.verbose:
        logger.setLevel("DEBUG")

    logger.info("Starting Word2Vec training script")

    try:
        # Validate inputs
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        if not validate_config_file(args.config):
            raise FileNotFoundError(f"Config file not found or invalid: {args.config}")

        # Load configuration
        config = load_config(args.config)
        word2vec_config = config.get("nlp", {}).get("word2vec", {})

        logger.info(f"Loaded configuration from {args.config}")

        # Load tweet data
        logger.info(f"Loading tweet data from {args.input}")
        df = load_tweet_data(args.input, args.text_column)

        if len(df) < args.min_samples:
            raise ValueError(f"Not enough samples: {len(df)} < {args.min_samples}")

        logger.info(f"Loaded {len(df)} tweets")

        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        logger.info("Initialized text preprocessor")

        # Preprocess tweets
        logger.info("Preprocessing tweets...")
        texts = df[args.text_column].tolist()
        tokenized_texts = []

        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(texts)} texts")

            tokens = preprocessor.preprocess(text)
            if tokens:  # Only add non-empty token lists
                tokenized_texts.append(tokens)

        logger.info(
            f"Preprocessed {len(tokenized_texts)} texts (removed {len(texts) - len(tokenized_texts)} empty)"
        )

        if len(tokenized_texts) < args.min_samples:
            raise ValueError(
                f"Not enough valid samples after preprocessing: {len(tokenized_texts)} < {args.min_samples}"
            )

        # Initialize Word2Vec model
        embeddings = Word2VecEmbeddings(
            vector_size=word2vec_config.get("vector_size", 100),
            window=word2vec_config.get("window", 5),
            min_count=word2vec_config.get("min_count", 2),
            workers=word2vec_config.get("workers", 4),
            epochs=word2vec_config.get("epochs", 10),
        )

        logger.info("Initialized Word2Vec model with config:")
        logger.info(f"  Vector size: {embeddings.vector_size}")
        logger.info(f"  Window: {embeddings.window}")
        logger.info(f"  Min count: {embeddings.min_count}")
        logger.info(f"  Workers: {embeddings.workers}")
        logger.info(f"  Epochs: {embeddings.epochs}")

        # Train embeddings
        logger.info("Training Word2Vec embeddings...")
        embeddings.train(tokenized_texts)

        logger.info(
            f"Training completed. Vocabulary size: {embeddings.get_vocabulary_size()}"
        )

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save model
        logger.info(f"Saving model to {args.output}")
        embeddings.save_model(args.output)

        # Test the model with a few example words
        logger.info("Testing trained embeddings:")
        test_words = ["happy", "sad", "good", "bad", "love", "hate"]

        for word in test_words:
            vector = embeddings.get_word_vector(word)
            if vector is not None:
                similar_words = embeddings.get_similar_words(word, topn=3)
                similar_str = ", ".join([f"{w}({s:.2f})" for w, s in similar_words[:3]])
                logger.info(f"  '{word}': similar words = {similar_str}")
            else:
                logger.info(f"  '{word}': not in vocabulary")

        # Generate training summary
        summary = {
            "input_file": args.input,
            "output_file": args.output,
            "total_tweets": len(df),
            "processed_tweets": len(tokenized_texts),
            "vocabulary_size": embeddings.get_vocabulary_size(),
            "vector_size": embeddings.vector_size,
            "training_epochs": embeddings.epochs,
        }

        logger.info("Training Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        logger.info("Word2Vec training completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
