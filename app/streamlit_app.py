"""
Streamlit web application for Gift Recommendation Platform.

This module provides an interactive web interface for users to get
personalized gift recommendations based on sentiment analysis and hand size detection.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import sys
from pathlib import Path
from PIL import Image
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp.preprocessor import TextPreprocessor
from nlp.embeddings import Word2VecEmbeddings
from nlp.sentiment_model import LogisticRegression
from cv.hand_detector import HandDetector
from cv.size_estimator import HandSizeEstimator
from cv.visualizer import HandVisualizer
from recommendation.gift_database import GiftDatabase
from recommendation.engine import RecommendationEngine
from utils.logger import setup_logger
from utils.validators import validate_sentiment_score, validate_text_input

# Configure page
st.set_page_config(
    page_title="🎁 Gift Recommendation Platform",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logger
logger = setup_logger(__name__)


@st.cache_resource
def load_models():
    """Load all models and components."""
    try:
        # Initialize components
        preprocessor = TextPreprocessor()
        detector = HandDetector()
        estimator = HandSizeEstimator()
        visualizer = HandVisualizer()

        # Load models if they exist
        models = {
            "preprocessor": preprocessor,
            "detector": detector,
            "estimator": estimator,
            "visualizer": visualizer,
            "embeddings": None,
            "sentiment_model": None,
            "gift_db": None,
            "recommendation_engine": None,
        }

        # Try to load Word2Vec embeddings
        embeddings_path = "data/models/word2vec.model"
        if os.path.exists(embeddings_path):
            embeddings = Word2VecEmbeddings()
            embeddings.load_model(embeddings_path)
            models["embeddings"] = embeddings
            logger.info("Loaded Word2Vec embeddings")
        else:
            st.warning("Word2Vec embeddings not found. Please train the model first.")

        # Try to load sentiment model
        sentiment_path = "data/models/sentiment_model.pkl"
        if os.path.exists(sentiment_path):
            sentiment_model = LogisticRegression()
            sentiment_model.load_model(sentiment_path)
            models["sentiment_model"] = sentiment_model
            logger.info("Loaded sentiment model")
        else:
            st.warning("Sentiment model not found. Please train the model first.")

        # Load gift database
        gift_config_path = "config/gift_mapping.json"
        if os.path.exists(gift_config_path):
            gift_db = GiftDatabase(gift_config_path)
            recommendation_engine = RecommendationEngine(gift_db)
            models["gift_db"] = gift_db
            models["recommendation_engine"] = recommendation_engine
            logger.info("Loaded gift database and recommendation engine")
        else:
            st.error("Gift database not found. Please create the database first.")

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        logger.error(f"Error loading models: {e}")
        return None


def analyze_sentiment(text: str, models: dict) -> tuple:
    """Analyze sentiment of input text."""
    try:
        preprocessor = models["preprocessor"]
        embeddings = models["embeddings"]
        sentiment_model = models["sentiment_model"]

        if not embeddings or not sentiment_model:
            st.error("Sentiment analysis models not loaded")
            return None, None

        # Preprocess text
        tokens = preprocessor.preprocess(text)

        # Get sentence vector
        vector = embeddings.get_sentence_vector(tokens)
        vector = vector.reshape(1, -1)

        # Predict sentiment
        prob = sentiment_model.predict_proba(vector)[0]
        pred = sentiment_model.predict(vector)[0]

        sentiment_label = "Positive" if pred == 1 else "Negative"
        confidence = prob if pred == 1 else (1 - prob)

        return prob, sentiment_label

    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        logger.error(f"Error analyzing sentiment: {e}")
        return None, None


def detect_hand_size(image: np.ndarray, models: dict) -> tuple:
    """Detect hand size from image."""
    try:
        detector = models["detector"]
        estimator = models["estimator"]
        visualizer = models["visualizer"]

        # Detect hand landmarks
        landmarks = detector.get_landmarks(image)

        if not landmarks:
            return None, None, None, "No hand detected in the image"

        # Calculate measurements
        measurements = estimator.calculate_all_measurements(landmarks)

        if not measurements:
            return None, None, None, "Could not calculate hand measurements"

        # Classify size
        size_class = estimator.classify_size(measurements)

        # Create visualization
        visualized_image = visualizer.draw_measurements(
            image, landmarks, measurements, size_class
        )

        return landmarks, measurements, size_class, visualized_image

    except Exception as e:
        error_msg = f"Error detecting hand size: {e}"
        logger.error(error_msg)
        return None, None, None, error_msg


def get_recommendations(sentiment_score: float, hand_size: str, models: dict) -> list:
    """Get gift recommendations."""
    try:
        recommendation_engine = models["recommendation_engine"]

        if not recommendation_engine:
            st.error("Recommendation engine not loaded")
            return []

        recommendations = recommendation_engine.recommend(
            sentiment=sentiment_score,
            hand_size=hand_size.lower(),
            max_results=5,
            include_explanations=True,
        )

        return recommendations

    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        logger.error(f"Error getting recommendations: {e}")
        return []


def main():
    """Main Streamlit application."""

    # Title and header
    st.title("🎁 Personalized Gift Recommendation Platform")
    st.markdown(
        """
    Get personalized gift recommendations based on your sentiment and hand size!
    This platform combines NLP sentiment analysis with computer vision hand detection
    to suggest the perfect gifts for you.
    """
    )

    # Load models
    models = load_models()
    if not models:
        st.error("Failed to load required models. Please check your setup.")
        st.stop()

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    # Demo mode toggle
    demo_mode = st.sidebar.checkbox(
        "Demo Mode", value=False, help="Use pre-filled examples for demonstration"
    )

    if demo_mode:
        st.sidebar.info("Demo mode enabled - using sample data")

    # Main interface
    tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 Analytics", "ℹ️ About"])

    with tab1:
        # Create two columns for inputs
        col1, col2 = st.columns(2)

        with col1:
            st.header("1. 💭 Share Your Thoughts")

            if demo_mode:
                default_text = "I'm having such an amazing day! Everything is going perfectly and I feel so happy!"
            else:
                default_text = ""

            tweet_text = st.text_area(
                "What's on your mind? Share your current thoughts or feelings:",
                value=default_text,
                height=150,
                placeholder="e.g., Having a great day! Feeling excited about new opportunities...",
            )

            analyze_button = st.button("🔍 Analyze Sentiment", type="primary")

            # Sentiment analysis results
            sentiment_score = None
            sentiment_label = None

            if analyze_button and tweet_text:
                if validate_text_input(tweet_text, min_length=10):
                    with st.spinner("Analyzing sentiment..."):
                        sentiment_score, sentiment_label = analyze_sentiment(
                            tweet_text, models
                        )

                    if sentiment_score is not None:
                        st.success(f"✅ Sentiment Analysis Complete!")

                        # Display results
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Sentiment",
                                sentiment_label,
                                f"{sentiment_score:.1%} confidence",
                            )
                        with col_b:
                            # Progress bar for sentiment score
                            st.metric("Score", f"{sentiment_score:.3f}")
                            st.progress(sentiment_score)

                        # Store in session state
                        st.session_state.sentiment_score = sentiment_score
                        st.session_state.sentiment_label = sentiment_label
                else:
                    st.error("Please enter at least 10 characters of meaningful text.")
            elif analyze_button:
                st.error("Please enter some text to analyze.")

        with col2:
            st.header("2. 🤚 Hand Size Detection")

            # Image upload options
            upload_option = st.radio(
                "Choose how to provide your hand image:",
                (
                    ["Upload Image", "Use Webcam", "Demo Image"]
                    if demo_mode
                    else ["Upload Image", "Use Webcam"]
                ),
                horizontal=True,
            )

            hand_image = None

            if upload_option == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload a clear photo of your hand:",
                    type=["jpg", "jpeg", "png"],
                    help="For best results, ensure good lighting and your hand is clearly visible",
                )

                if uploaded_file is not None:
                    # Convert uploaded file to image
                    image = Image.open(uploaded_file)
                    hand_image = np.array(image)

                    st.image(image, caption="Uploaded Image", use_column_width=True)

            elif upload_option == "Use Webcam":
                st.info(
                    "Webcam capture functionality would be implemented here. For now, please upload an image."
                )
                # Note: Streamlit webcam integration requires additional setup
                # picture = st.camera_input("Take a picture of your hand")
                # if picture is not None:
                #     hand_image = np.array(Image.open(picture))

            elif upload_option == "Demo Image" and demo_mode:
                # Create a demo image (placeholder)
                demo_image = np.ones((300, 300, 3), dtype=np.uint8) * 128
                cv2.putText(
                    demo_image,
                    "Demo Hand Image",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                hand_image = demo_image
                st.image(demo_image, caption="Demo Image", use_column_width=True)

            # Hand size detection
            detect_button = st.button("📏 Detect Hand Size", type="primary")

            hand_size = None
            measurements = None

            if detect_button and hand_image is not None:
                with st.spinner("Detecting hand size..."):
                    landmarks, measurements, hand_size, result = detect_hand_size(
                        hand_image, models
                    )

                if isinstance(result, np.ndarray):  # Successful detection
                    st.success("✅ Hand Size Detection Complete!")

                    # Display results
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Hand Size", hand_size.title())

                        if measurements:
                            # Show key measurements
                            if "hand_length" in measurements:
                                st.metric(
                                    "Hand Length",
                                    f"{measurements['hand_length']:.1f}px",
                                )
                            if "palm_width" in measurements:
                                st.metric(
                                    "Palm Width", f"{measurements['palm_width']:.1f}px"
                                )

                    with col_b:
                        st.image(result, caption="Hand Analysis", use_column_width=True)

                    # Store in session state
                    st.session_state.hand_size = hand_size
                    st.session_state.measurements = measurements

                else:  # Error message
                    st.error(result)
            elif detect_button:
                st.error("Please provide a hand image first.")

        # Recommendations section
        st.header("3. 🎁 Your Personalized Recommendations")

        # Check if we have both inputs
        sentiment_score = st.session_state.get("sentiment_score", None)
        hand_size = st.session_state.get("hand_size", None)

        if demo_mode and not sentiment_score:
            sentiment_score = 0.85
            st.session_state.sentiment_score = sentiment_score

        if demo_mode and not hand_size:
            hand_size = "medium"
            st.session_state.hand_size = hand_size

        if sentiment_score is not None and hand_size is not None:
            get_recs_button = st.button(
                "🎯 Get My Recommendations", type="primary", size="large"
            )

            if get_recs_button:
                with st.spinner("Finding the perfect gifts for you..."):
                    recommendations = get_recommendations(
                        sentiment_score, hand_size, models
                    )

                if recommendations:
                    st.success(
                        f"✅ Found {len(recommendations)} perfect recommendations for you!"
                    )

                    # Display recommendations as cards
                    for i, rec in enumerate(recommendations):
                        with st.expander(
                            f"🎁 {rec.get('name', 'Gift')} - {rec.get('confidence', 0):.1%} match",
                            expanded=i == 0,
                        ):

                            col_info, col_details = st.columns([2, 1])

                            with col_info:
                                st.write(
                                    f"**Category:** {rec.get('category', 'N/A').title()}"
                                )
                                st.write(
                                    f"**Price Range:** {rec.get('price_range', 'N/A')}"
                                )
                                st.write(
                                    f"**Description:** {rec.get('description', 'No description available')}"
                                )

                                if "explanation" in rec:
                                    st.write(
                                        f"**Why this is perfect for you:** {rec['explanation']}"
                                    )

                            with col_details:
                                # Confidence meter
                                confidence = rec.get("confidence", 0)
                                st.metric("Match Score", f"{confidence:.1%}")
                                st.progress(confidence)

                                # Gift details
                                st.write(
                                    f"**Hand Size:** {rec.get('hand_size', 'N/A').title()}"
                                )
                                st.write(
                                    f"**Sentiment Range:** {rec.get('sentiment_min', 0):.1f} - {rec.get('sentiment_max', 1):.1f}"
                                )

                else:
                    st.warning("No recommendations found. Try adjusting your inputs.")
        else:
            # Show what's missing
            missing = []
            if sentiment_score is None:
                missing.append("sentiment analysis")
            if hand_size is None:
                missing.append("hand size detection")

            st.info(
                f"Complete the {' and '.join(missing)} step(s) above to get your personalized recommendations!"
            )

    with tab2:
        st.header("📊 Platform Analytics")

        if models["gift_db"]:
            # Database statistics
            stats = models["gift_db"].get_statistics()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Gifts", stats["total_gifts"])

            with col2:
                st.metric("Categories", len(stats["categories"]))

            with col3:
                st.metric("Hand Sizes", len(stats["size_distribution"]))

            with col4:
                sentiment_ranges = stats.get("sentiment_ranges", {})
                st.metric(
                    "Avg Range Size", f"{sentiment_ranges.get('avg_range_size', 0):.2f}"
                )

            # Category distribution
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Gift Categories")
                category_data = pd.DataFrame(
                    [
                        {"Category": cat, "Count": count}
                        for cat, count in stats["categories"].items()
                    ]
                )
                st.bar_chart(category_data.set_index("Category"))

            with col2:
                st.subheader("Hand Size Distribution")
                size_data = pd.DataFrame(
                    [
                        {"Size": size, "Count": count}
                        for size, count in stats["size_distribution"].items()
                    ]
                )
                st.bar_chart(size_data.set_index("Size"))

            # All gifts table
            st.subheader("Gift Database")
            all_gifts = models["gift_db"].get_all_gifts()

            if all_gifts:
                df = pd.DataFrame(all_gifts)
                # Select relevant columns for display
                display_cols = [
                    "name",
                    "category",
                    "hand_size",
                    "sentiment_min",
                    "sentiment_max",
                    "price_range",
                ]
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(df[available_cols], use_container_width=True)
        else:
            st.error("Gift database not loaded")

    with tab3:
        st.header("ℹ️ About the Platform")

        st.markdown(
            """
        ### 🎯 How It Works

        This platform uses cutting-edge AI to provide personalized gift recommendations:

        1. **Sentiment Analysis (NLP)**:
           - Analyzes your text input using a custom-trained logistic regression model
           - Uses Word2Vec embeddings for semantic understanding
           - Determines your emotional state and preferences

        2. **Hand Size Detection (Computer Vision)**:
           - Uses MediaPipe for accurate hand landmark detection
           - Calculates multiple hand measurements (length, width, span)
           - Classifies hand size to understand age group and preferences

        3. **Recommendation Engine**:
           - Combines sentiment scores with hand size data
           - Applies intelligent filtering and ranking algorithms
           - Provides confidence scores and explanations

        ### 🛠️ Technology Stack

        - **Backend**: Python with NumPy, Pandas
        - **NLP**: Custom Logistic Regression, Gensim Word2Vec, NLTK
        - **Computer Vision**: OpenCV, MediaPipe
        - **Web Framework**: Streamlit
        - **Machine Learning**: Custom implementations from scratch

        ### 🎨 Features

        - ✅ Real-time sentiment analysis
        - ✅ Accurate hand size detection
        - ✅ Personalized recommendations with confidence scores
        - ✅ Interactive web interface
        - ✅ Detailed explanations for each recommendation
        - ✅ Analytics and insights

        ### 🤝 Contributing

        This is an open-source project perfect for Hacktoberfest!

        **Ways to contribute:**
        - 🐛 Bug fixes and improvements
        - ✨ New features and enhancements
        - 📖 Documentation improvements
        - 🧪 Additional test coverage
        - 🎨 UI/UX improvements

        ### 📄 License

        This project is licensed under the MIT License.
        """
        )

        # Model status
        st.subheader("🔧 Model Status")

        model_status = {
            "Text Preprocessor": (
                "✅ Loaded" if models["preprocessor"] else "❌ Not Available"
            ),
            "Word2Vec Embeddings": (
                "✅ Loaded" if models["embeddings"] else "❌ Not Available"
            ),
            "Sentiment Model": (
                "✅ Loaded" if models["sentiment_model"] else "❌ Not Available"
            ),
            "Hand Detector": "✅ Loaded" if models["detector"] else "❌ Not Available",
            "Size Estimator": (
                "✅ Loaded" if models["estimator"] else "❌ Not Available"
            ),
            "Gift Database": "✅ Loaded" if models["gift_db"] else "❌ Not Available",
            "Recommendation Engine": (
                "✅ Loaded" if models["recommendation_engine"] else "❌ Not Available"
            ),
        }

        for model, status in model_status.items():
            st.write(f"**{model}**: {status}")


if __name__ == "__main__":
    main()
