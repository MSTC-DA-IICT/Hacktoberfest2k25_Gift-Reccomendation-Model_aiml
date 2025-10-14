# 🎁 Personalized Gift Recommendation Platform

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/gift-recommendation-platform)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hacktoberfest](https://img.shields.io/badge/Hacktoberfest-2025-orange)](https://hacktoberfest.digitalocean.com/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive platform that combines **Natural Language Processing** (sentiment analysis) and **Computer Vision** (hand size detection) to provide personalized gift recommendations. Built for Hacktoberfest 2025 with a focus on modularity, testing, and contributor-friendliness.

## 🌟 Features

### 🧠 NLP Module
- **Custom Sentiment Analysis**: Logistic Regression implemented from scratch
- **Word2Vec Embeddings**: Semantic understanding of text
- **Tweet Processing**: Specialized preprocessing for social media content

### 👁️ Computer Vision Module
- **Hand Detection**: MediaPipe-powered hand landmark detection
- **Size Estimation**: Intelligent hand size classification
- **Visual Feedback**: Real-time visualization of measurements

### 🎯 Recommendation Engine
- **Multi-Signal Processing**: Combines sentiment + hand size
- **Smart Matching**: Rule-based gift category mapping
- **Confidence Scoring**: Transparent recommendation reasoning

### 🖥️ Multiple Interfaces
- **Streamlit Web App**: Interactive web interface
- **CLI Tool**: Command-line interface for batch processing
- **REST API**: RESTful endpoints for integration

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/gift-recommendation-platform.git
cd gift-recommendation-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Usage Examples

#### Web App
```bash
streamlit run app/streamlit_app.py
```

#### CLI
```bash
# Single prediction
gift-recommend --tweet "Having a great day!" --hand-image hand.jpg

# Batch processing
gift-recommend --tweets tweets.csv --hand-image hand.jpg --output results.json
```

#### Python API
```python
from src.nlp.sentiment_model import LogisticRegression
from src.cv.hand_detector import HandDetector
from src.recommendation.engine import RecommendationEngine

# Initialize components
detector = HandDetector()
engine = RecommendationEngine(gift_db)

# Get recommendations
sentiment_score = 0.85  # From NLP pipeline
hand_size = "small"     # From CV pipeline
recommendations = engine.recommend(sentiment_score, hand_size)
```

## 🏗️ Architecture

The platform follows a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐
│   NLP Module    │    │   CV Module     │
│                 │    │                 │
│ • Preprocessor  │    │ • Hand Detector │
│ • Embeddings    │    │ • Size Estimator│
│ • Sentiment     │    │ • Visualizer    │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
          ┌─────────────────┐
          │ Recommendation  │
          │     Engine      │
          │                 │
          │ • Gift Database │
          │ • Scoring Rules │
          │ • Explainer     │
          └─────────────────┘
```

For detailed architecture information, see [docs/architecture.md](docs/architecture.md).

## 📊 Dataset & Models

### Training Data
- **Sentiment Data**: 10K+ labeled tweets
- **Hand Images**: 100+ images across size categories
- **Gift Database**: 50+ categorized gift items

### Model Performance
- **Sentiment Analysis**: >85% accuracy
- **Hand Size Detection**: >90% accuracy
- **End-to-End**: >80% user satisfaction

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/ --cov-report=html

# Run specific module tests
pytest tests/test_nlp/
pytest tests/test_cv/
pytest tests/test_recommendation/
```

## 📚 Documentation

- **[Architecture Overview](docs/architecture.md)**: System design and components
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[User Guide](docs/user_guide.md)**: Installation, usage, and troubleshooting
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute to the project

## 🤝 Contributing

We welcome contributions! This project is designed to be beginner-friendly and perfect for Hacktoberfest participation.

### Ways to Contribute
- 🐛 **Bug fixes**: Find and fix issues
- ✨ **New features**: Enhance existing modules
- 📖 **Documentation**: Improve guides and examples
- 🧪 **Testing**: Add test coverage
- 🎨 **UI/UX**: Improve the interface

### Getting Started
1. Check out our [Contributing Guide](CONTRIBUTING.md)
2. Look for issues labeled `good first issue`
3. Join discussions in GitHub Issues
4. Submit your first PR!

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, NumPy, Pandas
- **NLP**: NLTK, Gensim (Word2Vec)
- **Computer Vision**: OpenCV, MediaPipe
- **Web Framework**: Streamlit, FastAPI
- **Testing**: Pytest, Coverage
- **Code Quality**: Black, Flake8

## 📈 Roadmap

- [ ] **Multi-language Support**: Support for additional languages
- [ ] **Advanced ML Models**: Deep learning improvements
- [ ] **Mobile App**: React Native interface
- [ ] **Real-time Updates**: Live recommendation updates
- [ ] **Social Integration**: Direct social media integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **MediaPipe Team**: For excellent hand detection capabilities
- **Gensim Contributors**: For Word2Vec implementation
- **Streamlit Team**: For the amazing web framework
- **Hacktoberfest**: For promoting open source collaboration

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/gift-recommendation-platform/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/gift-recommendation-platform/discussions)
- 📧 **Email**: your.email@example.com

---

**Made with ❤️ for Hacktoberfest 2025**

*Help us make gift-giving more personal and meaningful!*