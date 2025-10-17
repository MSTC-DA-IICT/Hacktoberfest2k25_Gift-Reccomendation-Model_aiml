"""
Test cases for recommendation engine.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from recommendation.engine import RecommendationEngine
from recommendation.gift_database import GiftDatabase


class TestRecommendationEngine:
    """Test cases for RecommendationEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock gift database with sample data
        self.mock_gifts = [
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
        ]

        # Create mock database
        self.mock_gift_db = Mock(spec=GiftDatabase)
        self.mock_gift_db.get_all_gifts.return_value = self.mock_gifts
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = []
        self.mock_gift_db.filter_by_size.return_value = []
        self.mock_gift_db.filter_by_sentiment.return_value = []

        self.engine = RecommendationEngine(self.mock_gift_db)

    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.gift_db == self.mock_gift_db
        assert self.engine.min_confidence == 0.3
        assert self.engine.recommendation_weights is not None

    def test_engine_with_custom_confidence(self):
        """Test engine with custom minimum confidence."""
        engine = RecommendationEngine(self.mock_gift_db, min_confidence=0.5)
        assert engine.min_confidence == 0.5

    def test_recommend_invalid_sentiment(self):
        """Test recommendation with invalid sentiment."""
        recommendations = self.engine.recommend(1.5, "small")  # Invalid sentiment
        assert isinstance(recommendations, list)
        assert len(recommendations) == 0

    def test_recommend_invalid_hand_size(self):
        """Test recommendation with invalid hand size."""
        recommendations = self.engine.recommend(0.8, "extra-large")  # Invalid size
        assert isinstance(recommendations, list)
        assert len(recommendations) == 0

    def test_recommend_no_candidates(self):
        """Test recommendation when no candidates are found."""
        # Mock returns empty lists
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = []
        self.mock_gift_db.filter_by_size.return_value = []
        self.mock_gift_db.filter_by_sentiment.return_value = []
        self.mock_gift_db.get_all_gifts.return_value = []

        recommendations = self.engine.recommend(0.8, "small")
        assert isinstance(recommendations, list)
        assert len(recommendations) == 0

    def test_recommend_with_candidates(self):
        """Test recommendation with valid candidates."""
        # Mock database to return candidates
        candidates = [self.mock_gifts[0]]  # Toy car for small hands
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = candidates

        recommendations = self.engine.recommend(0.8, "small")
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check recommendation structure
        rec = recommendations[0]
        assert "name" in rec
        assert "confidence" in rec
        assert "sentiment_input" in rec
        assert "hand_size_input" in rec

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        gift = self.mock_gifts[0]  # Toy car
        confidence = self.engine.calculate_confidence(gift, 0.8, "small")

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_perfect_match(self):
        """Test confidence calculation for perfect match."""
        gift = {
            "hand_size": "small",
            "sentiment_min": 0.7,
            "sentiment_max": 0.9,
            "category": "toy",
        }

        confidence = self.engine.calculate_confidence(gift, 0.8, "small")
        assert confidence > 0.5  # Should be reasonably high

    def test_calculate_confidence_wrong_size(self):
        """Test confidence calculation for wrong hand size."""
        gift = {
            "hand_size": "large",
            "sentiment_min": 0.7,
            "sentiment_max": 0.9,
            "category": "toy",
        }

        confidence = self.engine.calculate_confidence(gift, 0.8, "small")
        # Should be very low due to size mismatch
        assert confidence < 0.5

    def test_calculate_confidence_sentiment_mismatch(self):
        """Test confidence calculation for sentiment mismatch."""
        gift = {
            "hand_size": "small",
            "sentiment_min": 0.0,
            "sentiment_max": 0.3,
            "category": "toy",
        }

        confidence = self.engine.calculate_confidence(gift, 0.8, "small")
        # Should be low due to sentiment mismatch
        assert confidence < 0.5

    def test_explain_recommendation(self):
        """Test recommendation explanation."""
        gift = self.mock_gifts[0]
        explanation = self.engine.explain_recommendation(gift, 0.8, "small")

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "Confidence" in explanation

    def test_get_recommendation_summary(self):
        """Test recommendation summary generation."""
        # Mock some candidates
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = [
            self.mock_gifts[0]
        ]

        summary = self.engine.get_recommendation_summary(0.8, "small")

        assert isinstance(summary, dict)
        assert "total_candidates" in summary
        assert "input_analysis" in summary
        assert summary["total_candidates"] >= 0

    def test_batch_recommend(self):
        """Test batch recommendation processing."""
        requests = [
            {"sentiment": 0.8, "hand_size": "small"},
            {"sentiment": 0.3, "hand_size": "medium"},
        ]

        # Mock some candidates
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = [
            self.mock_gifts[0]
        ]

        results = self.engine.batch_recommend(requests)

        assert isinstance(results, list)
        assert len(results) == 2

        for result in results:
            assert "request_id" in result
            assert "recommendations" in result
            assert "sentiment" in result
            assert "hand_size" in result

    def test_batch_recommend_invalid_requests(self):
        """Test batch recommendation with invalid requests."""
        invalid_requests = [
            {"sentiment": 0.8},  # Missing hand_size
            {"hand_size": "small"},  # Missing sentiment
            {},  # Missing both
        ]

        results = self.engine.batch_recommend(invalid_requests)

        assert isinstance(results, list)
        assert len(results) == 3

        # All should have errors
        for result in results:
            assert "error" in result

    def test_update_recommendation_weights(self):
        """Test updating recommendation weights."""
        new_weights = {
            "category_match": 0.5,
            "sentiment_alignment": 0.3,
            "exact_size_match": 0.2,
        }

        self.engine.update_recommendation_weights(new_weights)

        # Check that weights were updated
        for key, value in new_weights.items():
            assert self.engine.recommendation_weights[key] == value

    def test_get_engine_statistics(self):
        """Test engine statistics retrieval."""
        # Mock database statistics
        mock_stats = {
            "total_gifts": 10,
            "categories": ["toy", "book"],
            "size_distribution": {"small": 3, "medium": 4, "large": 3},
        }
        self.mock_gift_db.get_statistics.return_value = mock_stats

        stats = self.engine.get_engine_statistics()

        assert isinstance(stats, dict)
        assert "total_gifts" in stats
        assert "min_confidence_threshold" in stats
        assert "recommendation_weights" in stats
        assert "database_stats" in stats

    def test_recommendation_with_explanations(self):
        """Test that recommendations include explanations when requested."""
        candidates = [self.mock_gifts[0]]
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = candidates

        recommendations = self.engine.recommend(0.8, "small", include_explanations=True)

        if recommendations:
            assert "explanation" in recommendations[0]
            assert isinstance(recommendations[0]["explanation"], str)

    def test_recommendation_without_explanations(self):
        """Test that recommendations exclude explanations when not requested."""
        candidates = [self.mock_gifts[0]]
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = candidates

        recommendations = self.engine.recommend(
            0.8, "small", include_explanations=False
        )

        if recommendations:
            assert "explanation" not in recommendations[0]

    def test_max_results_limit(self):
        """Test that max_results parameter is respected."""
        # Create more candidates than max_results
        many_candidates = self.mock_gifts * 3  # 9 candidates
        self.mock_gift_db.filter_by_size_and_sentiment.return_value = many_candidates

        recommendations = self.engine.recommend(0.8, "small", max_results=2)

        assert len(recommendations) <= 2
