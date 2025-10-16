"""
Main recommendation engine.

This module implements the core recommendation logic that combines
sentiment analysis results with hand size detection to suggest gifts.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
from .gift_database import GiftDatabase
from .rules import (
    get_category_priority,
    calculate_category_score,
    explain_recommendation_logic,
    validate_gift_compatibility,
    get_recommendation_weights,
    calculate_diversity_penalty,
)

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Main recommendation engine.

    This class combines sentiment analysis and hand size detection
    to generate personalized gift recommendations.
    """

    def __init__(self, gift_database: GiftDatabase, min_confidence: float = 0.3):
        """
        Initialize the recommendation engine.

        Parameters
        ----------
        gift_database : GiftDatabase
            Gift database instance
        min_confidence : float, default=0.3
            Minimum confidence score for recommendations
        """
        self.gift_db = gift_database
        self.min_confidence = min_confidence
        self.recommendation_weights = get_recommendation_weights()

        logger.info(
            f"RecommendationEngine initialized with {len(self.gift_db.get_all_gifts())} gifts"
        )

    def recommend(
        self,
        sentiment: float,
        hand_size: str,
        max_results: int = 5,
        include_explanations: bool = True,
    ) -> List[Dict]:
        """
        Generate gift recommendations.

        Parameters
        ----------
        sentiment : float
            Sentiment score between 0 and 1
        hand_size : str
            Hand size category ('small', 'medium', 'large')
        max_results : int, default=5
            Maximum number of recommendations to return
        include_explanations : bool, default=True
            Whether to include explanation text

        Returns
        -------
        List[Dict]
            Ranked list of gift recommendations with confidence scores

        Examples
        --------
        >>> engine = RecommendationEngine(gift_db)
        >>> recommendations = engine.recommend(0.8, 'small', max_results=3)
        >>> for rec in recommendations:
        ...     print(f"{rec['name']}: {rec['confidence']:.2f}")
        """
        try:
            logger.info(
                f"Generating recommendations for sentiment={sentiment:.2f}, size={hand_size}"
            )

            # Validate inputs
            if not self._validate_inputs(sentiment, hand_size):
                logger.error("Invalid input parameters")
                return []

            # Get candidate gifts
            candidates = self.gift_db.filter_by_size_and_sentiment(hand_size, sentiment)

            if not candidates:
                logger.warning("No candidates found, trying relaxed filtering")
                candidates = self._get_fallback_candidates(sentiment, hand_size)

            if not candidates:
                logger.error("No suitable gifts found")
                return []

            # Score and rank candidates
            scored_candidates = self._score_candidates(candidates, sentiment, hand_size)

            # Filter by minimum confidence
            qualified_candidates = [
                candidate
                for candidate in scored_candidates
                if candidate["confidence"] >= self.min_confidence
            ]

            if not qualified_candidates:
                logger.warning(
                    f"No candidates meet minimum confidence {self.min_confidence}"
                )
                # Lower threshold and try again
                qualified_candidates = [
                    candidate
                    for candidate in scored_candidates
                    if candidate["confidence"] >= self.min_confidence * 0.5
                ]

            # Apply diversity filtering
            diverse_candidates = self._apply_diversity_filtering(qualified_candidates)

            # Limit results
            final_recommendations = diverse_candidates[:max_results]

            # Add explanations if requested
            if include_explanations:
                for rec in final_recommendations:
                    rec["explanation"] = explain_recommendation_logic(
                        rec, hand_size, sentiment
                    )

            logger.info(f"Generated {len(final_recommendations)} recommendations")
            return final_recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def _validate_inputs(self, sentiment: float, hand_size: str) -> bool:
        """Validate input parameters."""
        if not (0 <= sentiment <= 1):
            logger.error(f"Invalid sentiment score: {sentiment}")
            return False

        if hand_size not in ["small", "medium", "large"]:
            logger.error(f"Invalid hand size: {hand_size}")
            return False

        return True

    def _get_fallback_candidates(self, sentiment: float, hand_size: str) -> List[Dict]:
        """Get fallback candidates when primary filtering fails."""
        try:
            # Try size-only filtering first
            candidates = self.gift_db.filter_by_size(hand_size)

            if not candidates:
                # Try sentiment-only filtering
                candidates = self.gift_db.filter_by_sentiment(sentiment)

            if not candidates:
                # Last resort: use all gifts
                logger.warning("Using all gifts as fallback candidates")
                candidates = self.gift_db.get_all_gifts()

            logger.info(f"Found {len(candidates)} fallback candidates")
            return candidates

        except Exception as e:
            logger.error(f"Error getting fallback candidates: {e}")
            return []

    def _score_candidates(
        self, candidates: List[Dict], sentiment: float, hand_size: str
    ) -> List[Dict]:
        """Score and rank candidate gifts."""
        scored_candidates = []

        try:
            for gift in candidates:
                confidence = self.calculate_confidence(gift, sentiment, hand_size)

                # Create recommendation object
                recommendation = gift.copy()
                recommendation["confidence"] = confidence
                recommendation["sentiment_input"] = sentiment
                recommendation["hand_size_input"] = hand_size

                scored_candidates.append(recommendation)

            # Sort by confidence (descending)
            scored_candidates.sort(key=lambda x: x["confidence"], reverse=True)

            logger.debug(f"Scored {len(scored_candidates)} candidates")
            return scored_candidates

        except Exception as e:
            logger.error(f"Error scoring candidates: {e}")
            return []

    def calculate_confidence(
        self, gift: Dict, sentiment: float, hand_size: str
    ) -> float:
        """
        Calculate confidence score for gift recommendation.

        Parameters
        ----------
        gift : Dict
            Gift dictionary
        sentiment : float
            Sentiment score
        hand_size : str
            Hand size category

        Returns
        -------
        float
            Confidence score between 0 and 1

        Examples
        --------
        >>> engine = RecommendationEngine(gift_db)
        >>> confidence = engine.calculate_confidence(gift, 0.7, 'medium')
        >>> print(f"Confidence: {confidence:.2f}")
        """
        try:
            confidence_components = {}

            # 1. Hand size match (exact match gets full score)
            if gift.get("hand_size") == hand_size:
                size_score = 1.0
            else:
                size_score = 0.0  # No partial credit for wrong size
            confidence_components["size_match"] = size_score

            # 2. Sentiment alignment (how well sentiment fits range)
            sent_min = gift.get("sentiment_min", 0)
            sent_max = gift.get("sentiment_max", 1)

            if sent_min <= sentiment <= sent_max:
                # Calculate how well centered the sentiment is in the range
                range_size = sent_max - sent_min
                if range_size == 0:
                    sentiment_score = 1.0
                else:
                    range_center = (sent_min + sent_max) / 2
                    distance_from_center = abs(sentiment - range_center)
                    sentiment_score = max(
                        0.0, 1.0 - (distance_from_center / (range_size / 2))
                    )
            else:
                sentiment_score = 0.0

            confidence_components["sentiment_alignment"] = sentiment_score

            # 3. Category compatibility
            category = gift.get("category", "")
            category_score = calculate_category_score(category, hand_size, sentiment)
            confidence_components["category_match"] = category_score

            # 4. Calculate weighted final score
            weights = self.recommendation_weights
            final_confidence = (
                weights["exact_size_match"] * size_score
                + weights["sentiment_alignment"] * sentiment_score
                + weights["category_match"] * category_score
            )

            # Add small popularity boost (could be enhanced with actual popularity data)
            popularity_boost = weights.get("popularity_boost", 0.0)
            final_confidence += popularity_boost

            # Ensure score is within valid range
            final_confidence = max(0.0, min(1.0, final_confidence))

            logger.debug(
                f"Confidence for {gift.get('name', 'Unknown')}: {final_confidence:.3f} "
                f"(components: {confidence_components})"
            )

            return final_confidence

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def _apply_diversity_filtering(self, candidates: List[Dict]) -> List[Dict]:
        """Apply diversity filtering to avoid too many similar recommendations."""
        try:
            diverse_candidates = []
            used_categories = []

            for candidate in candidates:
                category = candidate.get("category", "unknown")

                # Calculate diversity penalty
                penalty = calculate_diversity_penalty(used_categories, category)

                # Apply penalty to confidence
                original_confidence = candidate["confidence"]
                adjusted_confidence = max(0.0, original_confidence - penalty)
                candidate["confidence"] = adjusted_confidence
                candidate["diversity_penalty"] = penalty

                diverse_candidates.append(candidate)
                used_categories.append(category)

            # Re-sort by adjusted confidence
            diverse_candidates.sort(key=lambda x: x["confidence"], reverse=True)

            logger.debug(
                f"Applied diversity filtering to {len(diverse_candidates)} candidates"
            )
            return diverse_candidates

        except Exception as e:
            logger.error(f"Error applying diversity filtering: {e}")
            return candidates

    def explain_recommendation(
        self, gift: Dict, sentiment: float, hand_size: str
    ) -> str:
        """
        Generate detailed explanation for a recommendation.

        Parameters
        ----------
        gift : Dict
            Gift dictionary
        sentiment : float
            Sentiment score used
        hand_size : str
            Hand size used

        Returns
        -------
        str
            Detailed explanation

        Examples
        --------
        >>> engine = RecommendationEngine(gift_db)
        >>> explanation = engine.explain_recommendation(gift, 0.8, 'small')
        >>> print(explanation)
        """
        try:
            # Use the rules module for basic explanation
            basic_explanation = explain_recommendation_logic(gift, hand_size, sentiment)

            # Add confidence details
            confidence = self.calculate_confidence(gift, sentiment, hand_size)
            confidence_text = f" (Confidence: {confidence:.1%})"

            # Combine explanations
            full_explanation = basic_explanation + confidence_text

            return full_explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Recommended based on your preferences."

    def get_recommendation_summary(
        self, sentiment: float, hand_size: str
    ) -> Dict[str, Any]:
        """
        Get summary statistics about recommendations for given inputs.

        Parameters
        ----------
        sentiment : float
            Sentiment score
        hand_size : str
            Hand size category

        Returns
        -------
        Dict[str, Any]
            Summary statistics

        Examples
        --------
        >>> engine = RecommendationEngine(gift_db)
        >>> summary = engine.get_recommendation_summary(0.7, 'medium')
        >>> print(f"Available gifts: {summary['total_candidates']}")
        """
        try:
            summary = {}

            # Get all candidates
            candidates = self.gift_db.filter_by_size_and_sentiment(hand_size, sentiment)
            summary["total_candidates"] = len(candidates)

            # Category distribution
            categories = [gift.get("category", "unknown") for gift in candidates]
            category_counts = {}
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            summary["category_distribution"] = category_counts

            # Get priority categories
            priority_categories = get_category_priority(hand_size, sentiment)
            summary["priority_categories"] = priority_categories[:3]  # Top 3

            # Average confidence
            if candidates:
                confidences = [
                    self.calculate_confidence(gift, sentiment, hand_size)
                    for gift in candidates
                ]
                summary["average_confidence"] = sum(confidences) / len(confidences)
                summary["max_confidence"] = max(confidences)
                summary["min_confidence"] = min(confidences)
            else:
                summary["average_confidence"] = 0.0
                summary["max_confidence"] = 0.0
                summary["min_confidence"] = 0.0

            # Input analysis
            summary["input_analysis"] = {
                "sentiment_category": "positive" if sentiment >= 0.5 else "negative",
                "sentiment_strength": abs(sentiment - 0.5) * 2,
                "hand_size": hand_size,
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating recommendation summary: {e}")
            return {}

    def batch_recommend(self, requests: List[Dict]) -> List[Dict]:
        """
        Process multiple recommendation requests in batch.

        Parameters
        ----------
        requests : List[Dict]
            List of request dictionaries with 'sentiment' and 'hand_size' keys

        Returns
        -------
        List[Dict]
            List of recommendation results

        Examples
        --------
        >>> requests = [
        ...     {'sentiment': 0.8, 'hand_size': 'small'},
        ...     {'sentiment': 0.3, 'hand_size': 'large'}
        ... ]
        >>> results = engine.batch_recommend(requests)
        """
        try:
            results = []

            for i, request in enumerate(requests):
                logger.info(f"Processing batch request {i+1}/{len(requests)}")

                sentiment = request.get("sentiment")
                hand_size = request.get("hand_size")
                max_results = request.get("max_results", 5)

                if sentiment is None or hand_size is None:
                    logger.error(f"Invalid request {i}: missing sentiment or hand_size")
                    results.append(
                        {"error": "Missing required parameters", "recommendations": []}
                    )
                    continue

                recommendations = self.recommend(sentiment, hand_size, max_results)

                result = {
                    "request_id": i,
                    "sentiment": sentiment,
                    "hand_size": hand_size,
                    "recommendations": recommendations,
                    "count": len(recommendations),
                }
                results.append(result)

            logger.info(f"Completed batch processing of {len(requests)} requests")
            return results

        except Exception as e:
            logger.error(f"Error in batch recommendation: {e}")
            return []

    def update_recommendation_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update recommendation scoring weights.

        Parameters
        ----------
        new_weights : Dict[str, float]
            New weights dictionary

        Examples
        --------
        >>> new_weights = {'category_match': 0.5, 'sentiment_alignment': 0.5}
        >>> engine.update_recommendation_weights(new_weights)
        """
        try:
            # Validate weights
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.1:  # Allow small tolerance
                logger.warning(f"Weights don't sum to 1.0: {total_weight}")

            # Update weights
            self.recommendation_weights.update(new_weights)
            logger.info(
                f"Updated recommendation weights: {self.recommendation_weights}"
            )

        except Exception as e:
            logger.error(f"Error updating weights: {e}")

    def get_engine_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the recommendation engine.

        Returns
        -------
        Dict[str, Any]
            Engine statistics
        """
        try:
            stats = {
                "total_gifts": len(self.gift_db.get_all_gifts()),
                "min_confidence_threshold": self.min_confidence,
                "recommendation_weights": self.recommendation_weights.copy(),
                "database_stats": self.gift_db.get_statistics(),
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting engine statistics: {e}")
            return {}
