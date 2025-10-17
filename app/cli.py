#!/usr/bin/env python3
"""
Command-line interface for gift recommendation.

This module provides a CLI for batch processing and single predictions
for the gift recommendation platform.

Usage:
    python app/cli.py --text "Feeling great!" --hand-image hand.jpg
    python app/cli.py --tweets tweets.csv --hand-image hand.jpg --output results.json
    python app/cli.py --interactive
"""

import click
import json
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp.preprocessor import TextPreprocessor
from nlp.embeddings import Word2VecEmbeddings
from nlp.sentiment_model import LogisticRegression
from cv.hand_detector import HandDetector
from cv.size_estimator import HandSizeEstimator
from cv.visualizer import HandVisualizer
from cv.utils import load_image, save_image
from recommendation.gift_database import GiftDatabase
from recommendation.engine import RecommendationEngine
from utils.logger import setup_logger
from utils.validators import validate_text_input, validate_image_path

logger = setup_logger(__name__)


class GiftRecommendationCLI:
    """Main CLI class for gift recommendations."""

    def __init__(self):
        """Initialize CLI with models and components."""
        self.models_loaded = False
        self.models = {}
        self.load_models()

    def load_models(self) -> None:
        """Load all required models and components."""
        try:
            logger.info("Loading models and components...")

            # Initialize base components
            self.models["preprocessor"] = TextPreprocessor()
            self.models["detector"] = HandDetector()
            self.models["estimator"] = HandSizeEstimator()
            self.models["visualizer"] = HandVisualizer()

            # Load trained models
            self._load_nlp_models()
            self._load_recommendation_system()

            self.models_loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False

    def _load_nlp_models(self) -> None:
        """Load NLP models (embeddings and sentiment)."""
        # Word2Vec embeddings
        embeddings_path = "data/models/word2vec.model"
        if os.path.exists(embeddings_path):
            embeddings = Word2VecEmbeddings()
            embeddings.load_model(embeddings_path)
            self.models["embeddings"] = embeddings
            logger.info("Loaded Word2Vec embeddings")
        else:
            logger.warning(f"Word2Vec model not found: {embeddings_path}")

        # Sentiment model
        sentiment_path = "data/models/sentiment_model.pkl"
        if os.path.exists(sentiment_path):
            sentiment_model = LogisticRegression()
            sentiment_model.load_model(sentiment_path)
            self.models["sentiment_model"] = sentiment_model
            logger.info("Loaded sentiment model")
        else:
            logger.warning(f"Sentiment model not found: {sentiment_path}")

    def _load_recommendation_system(self) -> None:
        """Load recommendation system components."""
        gift_config_path = "config/gift_mapping.json"
        if os.path.exists(gift_config_path):
            gift_db = GiftDatabase(gift_config_path)
            recommendation_engine = RecommendationEngine(gift_db)
            self.models["gift_db"] = gift_db
            self.models["recommendation_engine"] = recommendation_engine
            logger.info("Loaded recommendation system")
        else:
            logger.error(f"Gift database not found: {gift_config_path}")

    def analyze_sentiment(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Analyze sentiment of input text."""
        try:
            if not self.models.get("embeddings") or not self.models.get(
                "sentiment_model"
            ):
                logger.error("NLP models not loaded")
                return None, None

            preprocessor = self.models["preprocessor"]
            embeddings = self.models["embeddings"]
            sentiment_model = self.models["sentiment_model"]

            # Preprocess text
            tokens = preprocessor.preprocess(text)
            if not tokens:
                logger.warning("Text preprocessing resulted in empty tokens")
                return None, None

            # Get sentence vector
            vector = embeddings.get_sentence_vector(tokens)
            vector = vector.reshape(1, -1)

            # Predict sentiment
            prob = sentiment_model.predict_proba(vector)[0]
            pred = sentiment_model.predict(vector)[0]

            sentiment_label = "positive" if pred == 1 else "negative"
            return prob, sentiment_label

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None, None

    def detect_hand_size(
        self, image_path: str, save_visualization: bool = False
    ) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        """Detect hand size from image."""
        try:
            # Load image
            image = load_image(image_path)
            if image is None:
                return None, None, "Failed to load image"

            detector = self.models["detector"]
            estimator = self.models["estimator"]

            # Detect hand landmarks
            landmarks = detector.get_landmarks(image)
            if not landmarks:
                return None, None, "No hand detected in image"

            # Calculate measurements
            measurements = estimator.calculate_all_measurements(landmarks)
            if not measurements:
                return None, None, "Could not calculate hand measurements"

            # Classify size
            size_class = estimator.classify_size(measurements)

            # Save visualization if requested
            visualization_path = None
            if save_visualization:
                visualizer = self.models["visualizer"]
                visualized_image = visualizer.draw_measurements(
                    image, landmarks, measurements, size_class
                )

                # Generate output path
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                visualization_path = f"{base_name}_analyzed.jpg"
                save_image(visualized_image, visualization_path)
                logger.info(f"Saved visualization: {visualization_path}")

            return size_class, measurements, visualization_path

        except Exception as e:
            error_msg = f"Error detecting hand size: {e}"
            logger.error(error_msg)
            return None, None, error_msg

    def get_recommendations(
        self, sentiment_score: float, hand_size: str, max_results: int = 5
    ) -> List[Dict]:
        """Get gift recommendations."""
        try:
            recommendation_engine = self.models.get("recommendation_engine")
            if not recommendation_engine:
                logger.error("Recommendation engine not loaded")
                return []

            recommendations = recommendation_engine.recommend(
                sentiment=sentiment_score,
                hand_size=hand_size.lower(),
                max_results=max_results,
                include_explanations=True,
            )

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    def process_single_request(
        self, text: str, image_path: str, save_visualization: bool = False
    ) -> Dict:
        """Process a single recommendation request."""
        result = {
            "input": {"text": text, "image_path": image_path},
            "sentiment": {},
            "hand_detection": {},
            "recommendations": [],
            "success": False,
            "error": None,
        }

        try:
            # Analyze sentiment
            sentiment_score, sentiment_label = self.analyze_sentiment(text)
            if sentiment_score is not None:
                result["sentiment"] = {
                    "score": sentiment_score,
                    "label": sentiment_label,
                    "confidence": (
                        sentiment_score
                        if sentiment_label == "positive"
                        else (1 - sentiment_score)
                    ),
                }
            else:
                result["error"] = "Failed to analyze sentiment"
                return result

            # Detect hand size
            hand_size, measurements, visualization_path = self.detect_hand_size(
                image_path, save_visualization
            )
            if hand_size is not None:
                result["hand_detection"] = {
                    "size": hand_size,
                    "measurements": measurements,
                    "visualization_path": visualization_path,
                }
            else:
                result["error"] = "Failed to detect hand size"
                return result

            # Get recommendations
            recommendations = self.get_recommendations(sentiment_score, hand_size)
            if recommendations:
                result["recommendations"] = recommendations
                result["success"] = True
            else:
                result["error"] = "No recommendations found"

            return result

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing request: {e}")
            return result

    def process_batch_requests(
        self, tweets_file: str, image_path: str, max_results: int = 5
    ) -> List[Dict]:
        """Process batch of tweets with single hand image."""
        try:
            # Load tweets
            df = pd.read_csv(tweets_file)
            if "text" not in df.columns:
                raise ValueError("CSV must contain 'text' column")

            logger.info(f"Processing {len(df)} tweets from {tweets_file}")

            # Detect hand size once (reuse for all tweets)
            hand_size, measurements, _ = self.detect_hand_size(image_path)
            if hand_size is None:
                raise ValueError("Failed to detect hand size from provided image")

            results = []

            for idx, row in df.iterrows():
                text = row["text"]
                logger.info(f"Processing tweet {idx+1}/{len(df)}")

                # Analyze sentiment
                sentiment_score, sentiment_label = self.analyze_sentiment(text)
                if sentiment_score is None:
                    continue

                # Get recommendations
                recommendations = self.get_recommendations(
                    sentiment_score, hand_size, max_results
                )

                result = {
                    "index": idx,
                    "text": text,
                    "sentiment": {"score": sentiment_score, "label": sentiment_label},
                    "hand_size": hand_size,
                    "recommendations": recommendations,
                }
                results.append(result)

            logger.info(f"Processed {len(results)} tweets successfully")
            return results

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []


@click.group()
def cli():
    """Gift Recommendation Platform CLI."""
    pass


@cli.command()
@click.option("--text", "-t", help="Text to analyze for sentiment")
@click.option("--tweets", help="CSV file with tweets to process")
@click.option("--hand-image", "-i", required=True, help="Path to hand image")
@click.option("--output", "-o", help="Output JSON file path")
@click.option("--max-results", default=5, help="Maximum number of recommendations")
@click.option("--save-viz", is_flag=True, help="Save hand detection visualization")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def recommend(text, tweets, hand_image, output, max_results, save_viz, verbose):
    """Generate gift recommendations."""

    if verbose:
        logger.setLevel("DEBUG")

    # Validate inputs
    if not text and not tweets:
        click.echo("Error: Either --text or --tweets must be provided", err=True)
        return

    if text and tweets:
        click.echo("Error: Provide either --text or --tweets, not both", err=True)
        return

    if not validate_image_path(hand_image):
        click.echo(f"Error: Invalid image path: {hand_image}", err=True)
        return

    # Initialize CLI
    cli_app = GiftRecommendationCLI()
    if not cli_app.models_loaded:
        click.echo("Error: Failed to load required models", err=True)
        return

    try:
        if text:
            # Single text processing
            if not validate_text_input(text):
                click.echo("Error: Invalid text input", err=True)
                return

            click.echo("🔍 Processing single recommendation request...")

            result = cli_app.process_single_request(text, hand_image, save_viz)

            if result["success"]:
                click.echo("✅ Recommendation completed successfully!")

                # Display results
                sentiment = result["sentiment"]
                hand_detection = result["hand_detection"]
                recommendations = result["recommendations"]

                click.echo(f"\n📊 Analysis Results:")
                click.echo(
                    f"  Sentiment: {sentiment['label']} ({sentiment['score']:.3f})"
                )
                click.echo(f"  Hand Size: {hand_detection['size']}")

                if hand_detection.get("visualization_path"):
                    click.echo(
                        f"  Visualization saved: {hand_detection['visualization_path']}"
                    )

                click.echo(f"\n🎁 Top {len(recommendations)} Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    click.echo(f"  {i}. {rec['name']} ({rec['confidence']:.1%} match)")
                    click.echo(
                        f"     Category: {rec['category']}, Price: {rec['price_range']}"
                    )

            else:
                click.echo(f"❌ Error: {result['error']}", err=True)

            # Save results if output specified
            if output:
                os.makedirs(os.path.dirname(output), exist_ok=True)
                with open(output, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                click.echo(f"\n💾 Results saved to: {output}")

        else:
            # Batch processing
            if not os.path.exists(tweets):
                click.echo(f"Error: Tweets file not found: {tweets}", err=True)
                return

            click.echo("📝 Processing batch of tweets...")

            results = cli_app.process_batch_requests(tweets, hand_image, max_results)

            if results:
                click.echo(f"✅ Processed {len(results)} tweets successfully!")

                # Display summary
                avg_sentiment = np.mean([r["sentiment"]["score"] for r in results])
                positive_count = sum(
                    1 for r in results if r["sentiment"]["label"] == "positive"
                )

                click.echo(f"\n📊 Batch Summary:")
                click.echo(f"  Total processed: {len(results)}")
                click.echo(f"  Average sentiment: {avg_sentiment:.3f}")
                click.echo(
                    f"  Positive sentiment: {positive_count}/{len(results)} ({positive_count/len(results):.1%})"
                )
                click.echo(f"  Hand size: {results[0]['hand_size']}")

                # Save results
                output_path = output or "batch_recommendations.json"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                click.echo(f"\n💾 Results saved to: {output_path}")

            else:
                click.echo("❌ No results generated", err=True)

    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)


@cli.command()
def interactive():
    """Run interactive CLI mode."""
    click.echo("🎁 Welcome to the Interactive Gift Recommendation Platform!")
    click.echo("=" * 60)

    # Initialize CLI
    cli_app = GiftRecommendationCLI()
    if not cli_app.models_loaded:
        click.echo("❌ Error: Failed to load required models")
        return

    click.echo("✅ Models loaded successfully!")

    while True:
        click.echo("\n" + "─" * 40)
        click.echo("Choose an option:")
        click.echo("1. Single recommendation")
        click.echo("2. Batch processing")
        click.echo("3. Model status")
        click.echo("4. Exit")

        choice = click.prompt("Enter your choice (1-4)", type=int)

        if choice == 1:
            # Single recommendation
            click.echo("\n🎯 Single Recommendation Mode")

            text = click.prompt("Enter your text/thoughts")
            if not validate_text_input(text):
                click.echo("Invalid text input. Please try again.")
                continue

            image_path = click.prompt("Enter path to hand image")
            if not validate_image_path(image_path):
                click.echo("Invalid image path. Please try again.")
                continue

            save_viz = click.confirm("Save visualization?", default=False)

            click.echo("Processing...")

            result = cli_app.process_single_request(text, image_path, save_viz)

            if result["success"]:
                sentiment = result["sentiment"]
                hand_detection = result["hand_detection"]
                recommendations = result["recommendations"]

                click.echo(f"\n📊 Results:")
                click.echo(
                    f"Sentiment: {sentiment['label']} ({sentiment['score']:.3f})"
                )
                click.echo(f"Hand Size: {hand_detection['size']}")

                click.echo(f"\n🎁 Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    click.echo(f"{i}. {rec['name']} ({rec['confidence']:.1%})")
                    click.echo(f"   {rec['description']}")

                if click.confirm("Save detailed results to file?", default=False):
                    output_file = click.prompt(
                        "Output filename", default="recommendation_result.json"
                    )
                    with open(output_file, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    click.echo(f"Results saved to {output_file}")

            else:
                click.echo(f"❌ Error: {result['error']}")

        elif choice == 2:
            # Batch processing
            click.echo("\n📝 Batch Processing Mode")

            tweets_file = click.prompt("Enter path to tweets CSV file")
            if not os.path.exists(tweets_file):
                click.echo("File not found. Please try again.")
                continue

            image_path = click.prompt("Enter path to hand image")
            if not validate_image_path(image_path):
                click.echo("Invalid image path. Please try again.")
                continue

            max_results = click.prompt(
                "Maximum recommendations per tweet", default=3, type=int
            )

            click.echo("Processing batch...")

            results = cli_app.process_batch_requests(
                tweets_file, image_path, max_results
            )

            if results:
                click.echo(f"✅ Processed {len(results)} tweets")

                output_file = click.prompt(
                    "Output filename", default="batch_results.json"
                )
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                click.echo(f"Results saved to {output_file}")
            else:
                click.echo("❌ Batch processing failed")

        elif choice == 3:
            # Model status
            click.echo("\n🔧 Model Status:")
            models = cli_app.models
            status_items = [
                ("Text Preprocessor", models.get("preprocessor") is not None),
                ("Word2Vec Embeddings", models.get("embeddings") is not None),
                ("Sentiment Model", models.get("sentiment_model") is not None),
                ("Hand Detector", models.get("detector") is not None),
                ("Size Estimator", models.get("estimator") is not None),
                ("Gift Database", models.get("gift_db") is not None),
                (
                    "Recommendation Engine",
                    models.get("recommendation_engine") is not None,
                ),
            ]

            for model_name, is_loaded in status_items:
                status = "✅ Loaded" if is_loaded else "❌ Not Available"
                click.echo(f"  {model_name}: {status}")

            if models.get("gift_db"):
                stats = models["gift_db"].get_statistics()
                click.echo(f"\nDatabase Stats:")
                click.echo(f"  Total gifts: {stats['total_gifts']}")
                click.echo(f"  Categories: {len(stats['categories'])}")

        elif choice == 4:
            click.echo("👋 Goodbye!")
            break

        else:
            click.echo("Invalid choice. Please try again.")


@cli.command()
def status():
    """Check model and system status."""
    click.echo("🔧 Gift Recommendation Platform Status")
    click.echo("=" * 40)

    # Initialize CLI
    cli_app = GiftRecommendationCLI()

    # Model status
    click.echo("Model Status:")
    models = cli_app.models
    status_items = [
        ("Text Preprocessor", models.get("preprocessor") is not None),
        ("Word2Vec Embeddings", models.get("embeddings") is not None),
        ("Sentiment Model", models.get("sentiment_model") is not None),
        ("Hand Detector", models.get("detector") is not None),
        ("Size Estimator", models.get("estimator") is not None),
        ("Gift Database", models.get("gift_db") is not None),
        ("Recommendation Engine", models.get("recommendation_engine") is not None),
    ]

    for model_name, is_loaded in status_items:
        status = "✅" if is_loaded else "❌"
        click.echo(f"  {status} {model_name}")

    # System info
    click.echo("\nSystem Info:")
    click.echo(f"  Python: {sys.version.split()[0]}")
    click.echo(f"  Platform: {sys.platform}")
    click.echo(f"  Working Directory: {os.getcwd()}")

    # File system check
    important_paths = [
        "config/config.yaml",
        "config/gift_mapping.json",
        "data/models/",
        "src/nlp/",
        "src/cv/",
    ]

    click.echo("\nFile System Check:")
    for path in important_paths:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        click.echo(f"  {status} {path}")


if __name__ == "__main__":
    cli()
