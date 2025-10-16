"""
Recommendation rules and mapping logic.

This module defines the business rules and mappings for gift recommendations
based on hand size and sentiment analysis.
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Base mapping rules for recommendations
GIFT_RULES = {
    "small": {
        "positive": ["toy", "book", "puzzle", "art_supplies", "jewelry"],
        "negative": ["book", "art_supplies", "comfort_item", "puzzle"],
    },
    "medium": {
        "positive": ["accessory", "gadget", "sports", "jewelry", "watch"],
        "negative": ["comfort_item", "book", "home_decor", "pillow"],
    },
    "large": {
        "positive": ["jewelry", "tech", "sports", "gadget", "watch"],
        "negative": ["appliance", "tool", "home_decor", "comfort_item"],
    },
}

# Sentiment thresholds for positive/negative classification
SENTIMENT_THRESHOLD = 0.5

# Category weights - higher values indicate better fit
CATEGORY_WEIGHTS = {
    "toy": {"small": 1.0, "medium": 0.3, "large": 0.1},
    "book": {"small": 0.9, "medium": 0.8, "large": 0.7},
    "puzzle": {"small": 0.8, "medium": 0.6, "large": 0.4},
    "art_supplies": {"small": 0.9, "medium": 0.7, "large": 0.5},
    "jewelry": {"small": 0.6, "medium": 0.8, "large": 1.0},
    "accessory": {"small": 0.4, "medium": 1.0, "large": 0.8},
    "gadget": {"small": 0.3, "medium": 0.9, "large": 1.0},
    "sports": {"small": 0.2, "medium": 0.8, "large": 1.0},
    "watch": {"small": 0.3, "medium": 0.9, "large": 0.8},
    "comfort_item": {"small": 0.7, "medium": 0.9, "large": 0.8},
    "home_decor": {"small": 0.3, "medium": 0.8, "large": 0.9},
    "pillow": {"small": 0.6, "medium": 0.9, "large": 0.8},
    "appliance": {"small": 0.1, "medium": 0.4, "large": 1.0},
    "tool": {"small": 0.1, "medium": 0.5, "large": 1.0},
    "tech": {"small": 0.4, "medium": 0.8, "large": 1.0},
    "home": {"small": 0.3, "medium": 0.8, "large": 0.9},
}

# Age group approximations based on hand size (for context)
AGE_APPROXIMATIONS = {
    "small": "child/teen",
    "medium": "young adult/adult",
    "large": "adult/large adult",
}

# Sentiment-based gift personality mapping
SENTIMENT_PERSONALITIES = {
    "very_positive": (0.8, 1.0, "enthusiastic, energetic"),
    "positive": (0.6, 0.8, "happy, optimistic"),
    "neutral": (0.4, 0.6, "balanced, practical"),
    "negative": (0.2, 0.4, "subdued, comfort-seeking"),
    "very_negative": (0.0, 0.2, "needs comfort, support"),
}


def get_category_priority(hand_size: str, sentiment: float) -> List[str]:
    """
    Return prioritized category list based on hand size and sentiment.

    Parameters
    ----------
    hand_size : str
        Hand size category ('small', 'medium', 'large')
    sentiment : float
        Sentiment score between 0 and 1

    Returns
    -------
    List[str]
        Ordered list of categories by priority

    Examples
    --------
    >>> priorities = get_category_priority('small', 0.8)
    >>> print(priorities[:3])  # Top 3 categories
    ['toy', 'puzzle', 'art_supplies']
    """
    if hand_size not in GIFT_RULES:
        logger.warning(f"Unknown hand size: {hand_size}")
        return []

    try:
        # Determine sentiment category
        sentiment_category = (
            "positive" if sentiment >= SENTIMENT_THRESHOLD else "negative"
        )

        # Get base categories for this combination
        base_categories = GIFT_RULES[hand_size][sentiment_category]

        # Sort by category weights for this hand size
        weighted_categories = []
        for category in base_categories:
            weight = CATEGORY_WEIGHTS.get(category, {}).get(hand_size, 0.5)
            weighted_categories.append((category, weight))

        # Sort by weight (descending)
        weighted_categories.sort(key=lambda x: x[1], reverse=True)

        # Return ordered category list
        priorities = [category for category, _ in weighted_categories]

        logger.debug(
            f"Category priorities for {hand_size}/{sentiment:.2f}: {priorities}"
        )
        return priorities

    except Exception as e:
        logger.error(f"Error getting category priority: {e}")
        return []


def calculate_category_score(category: str, hand_size: str, sentiment: float) -> float:
    """
    Calculate compatibility score for a category given hand size and sentiment.

    Parameters
    ----------
    category : str
        Gift category
    hand_size : str
        Hand size ('small', 'medium', 'large')
    sentiment : float
        Sentiment score between 0 and 1

    Returns
    -------
    float
        Compatibility score between 0 and 1

    Examples
    --------
    >>> score = calculate_category_score('toy', 'small', 0.9)
    >>> print(f"Compatibility score: {score:.2f}")
    """
    try:
        # Base score from category weights
        base_score = CATEGORY_WEIGHTS.get(category, {}).get(hand_size, 0.5)

        # Adjust based on sentiment alignment
        sentiment_category = (
            "positive" if sentiment >= SENTIMENT_THRESHOLD else "negative"
        )
        size_categories = GIFT_RULES.get(hand_size, {})

        if category in size_categories.get(sentiment_category, []):
            # Category is in the preferred list for this sentiment
            sentiment_bonus = 0.2
        elif category in size_categories.get(
            "positive" if sentiment_category == "negative" else "negative", []
        ):
            # Category is in the opposite sentiment list
            sentiment_penalty = -0.1
            base_score += sentiment_penalty
        else:
            # Category not specifically listed, neutral
            sentiment_bonus = 0.0

        # Apply sentiment bonus
        final_score = min(1.0, base_score + sentiment_bonus)

        # Additional sentiment strength modifier
        sentiment_strength = abs(sentiment - 0.5) * 2  # 0 to 1 scale
        final_score += sentiment_strength * 0.1

        # Ensure score is within valid range
        final_score = max(0.0, min(1.0, final_score))

        logger.debug(
            f"Category '{category}' score for {hand_size}/{sentiment:.2f}: {final_score:.3f}"
        )
        return final_score

    except Exception as e:
        logger.error(f"Error calculating category score: {e}")
        return 0.5  # Default neutral score


def get_sentiment_personality(sentiment: float) -> Tuple[str, str]:
    """
    Get personality description based on sentiment score.

    Parameters
    ----------
    sentiment : float
        Sentiment score between 0 and 1

    Returns
    -------
    Tuple[str, str]
        Personality category and description

    Examples
    --------
    >>> category, description = get_sentiment_personality(0.85)
    >>> print(f"{category}: {description}")
    very_positive: enthusiastic, energetic
    """
    for personality, (min_val, max_val, description) in SENTIMENT_PERSONALITIES.items():
        if min_val <= sentiment <= max_val:
            return personality, description

    # Fallback
    return "neutral", "balanced, practical"


def explain_recommendation_logic(gift: Dict, hand_size: str, sentiment: float) -> str:
    """
    Generate explanation for why a gift was recommended.

    Parameters
    ----------
    gift : Dict
        Gift dictionary
    hand_size : str
        Hand size used for recommendation
    sentiment : float
        Sentiment score used for recommendation

    Returns
    -------
    str
        Human-readable explanation

    Examples
    --------
    >>> explanation = explain_recommendation_logic(gift, 'small', 0.8)
    >>> print(explanation)
    This toy is perfect for small hands and matches your positive mood...
    """
    try:
        category = gift.get("category", "unknown")
        gift_name = gift.get("name", "Unknown Gift")

        # Get personality info
        personality, personality_desc = get_sentiment_personality(sentiment)

        # Get age approximation
        age_group = AGE_APPROXIMATIONS.get(hand_size, "unknown age group")

        # Calculate category compatibility
        category_score = calculate_category_score(category, hand_size, sentiment)

        # Build explanation
        explanations = []

        # Hand size fit
        explanations.append(f"This {category} is well-suited for {hand_size} hands")

        # Sentiment alignment
        if sentiment >= 0.7:
            explanations.append(f"matches your positive energy ({personality_desc})")
        elif sentiment >= 0.3:
            explanations.append(f"aligns with your balanced mood ({personality_desc})")
        else:
            explanations.append(
                f"provides comfort for your current mood ({personality_desc})"
            )

        # Category compatibility
        if category_score >= 0.8:
            explanations.append("and is highly recommended for your profile")
        elif category_score >= 0.6:
            explanations.append("and is a good fit for your preferences")
        else:
            explanations.append("and could be an interesting choice")

        # Price consideration
        price_range = gift.get("price_range", "Price varies")
        explanations.append(f"({price_range})")

        # Combine explanations
        main_explanation = ", ".join(explanations[:-1])
        if len(explanations) > 1:
            main_explanation += f", {explanations[-1]}"

        full_explanation = f"{gift_name}: {main_explanation}."

        logger.debug(f"Generated explanation for {gift_name}")
        return full_explanation

    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return f"Recommended based on your hand size and preferences."


def get_alternative_categories(
    hand_size: str, sentiment: float, exclude_categories: List[str] = None
) -> List[str]:
    """
    Get alternative categories when primary recommendations are not available.

    Parameters
    ----------
    hand_size : str
        Hand size category
    sentiment : float
        Sentiment score
    exclude_categories : List[str], optional
        Categories to exclude from alternatives

    Returns
    -------
    List[str]
        Alternative categories

    Examples
    --------
    >>> alternatives = get_alternative_categories('medium', 0.6, ['gadget'])
    >>> print(alternatives)
    ['accessory', 'watch', 'jewelry']
    """
    if exclude_categories is None:
        exclude_categories = []

    try:
        # Get all categories for this hand size, sorted by compatibility
        all_categories = list(CATEGORY_WEIGHTS.keys())
        scored_categories = []

        for category in all_categories:
            if category not in exclude_categories:
                score = calculate_category_score(category, hand_size, sentiment)
                scored_categories.append((category, score))

        # Sort by score (descending) and return categories
        scored_categories.sort(key=lambda x: x[1], reverse=True)
        alternatives = [
            category for category, _ in scored_categories[:5]
        ]  # Top 5 alternatives

        logger.debug(f"Found {len(alternatives)} alternative categories")
        return alternatives

    except Exception as e:
        logger.error(f"Error getting alternative categories: {e}")
        return []


def validate_gift_compatibility(
    gift: Dict, hand_size: str, sentiment: float, min_score: float = 0.3
) -> Tuple[bool, float, str]:
    """
    Validate if a gift is compatible with given criteria.

    Parameters
    ----------
    gift : Dict
        Gift dictionary
    hand_size : str
        Hand size category
    sentiment : float
        Sentiment score
    min_score : float, default=0.3
        Minimum compatibility score required

    Returns
    -------
    Tuple[bool, float, str]
        Is compatible, compatibility score, reason

    Examples
    --------
    >>> compatible, score, reason = validate_gift_compatibility(gift, 'large', 0.9)
    >>> print(f"Compatible: {compatible}, Score: {score:.2f}, Reason: {reason}")
    """
    try:
        # Check hand size match
        gift_size = gift.get("hand_size", "")
        if gift_size != hand_size:
            return (
                False,
                0.0,
                f"Hand size mismatch: gift is for {gift_size}, user has {hand_size}",
            )

        # Check sentiment range
        sent_min = gift.get("sentiment_min", 0)
        sent_max = gift.get("sentiment_max", 1)
        if not (sent_min <= sentiment <= sent_max):
            return (
                False,
                0.0,
                f"Sentiment out of range: {sentiment:.2f} not in [{sent_min}, {sent_max}]",
            )

        # Calculate compatibility score
        category = gift.get("category", "")
        score = calculate_category_score(category, hand_size, sentiment)

        # Check minimum score
        if score < min_score:
            return (
                False,
                score,
                f"Compatibility score {score:.2f} below minimum {min_score}",
            )

        # All checks passed
        return True, score, "Gift meets all compatibility criteria"

    except Exception as e:
        logger.error(f"Error validating gift compatibility: {e}")
        return False, 0.0, f"Validation error: {e}"


def get_recommendation_weights() -> Dict[str, float]:
    """
    Get weights used in recommendation scoring.

    Returns
    -------
    Dict[str, float]
        Weights for different recommendation factors

    Examples
    --------
    >>> weights = get_recommendation_weights()
    >>> print(f"Category weight: {weights['category_match']}")
    """
    return {
        "category_match": 0.4,  # How well category fits hand size
        "sentiment_alignment": 0.3,  # How well sentiment range matches
        "exact_size_match": 0.2,  # Exact hand size match bonus
        "popularity_boost": 0.1,  # General popularity of item type
    }


def calculate_diversity_penalty(
    recommended_categories: List[str], new_category: str, penalty_factor: float = 0.1
) -> float:
    """
    Calculate penalty for recommending similar categories.

    Parameters
    ----------
    recommended_categories : List[str]
        Categories already recommended
    new_category : str
        Category being considered
    penalty_factor : float, default=0.1
        Penalty factor per similar category

    Returns
    -------
    float
        Penalty value to subtract from score

    Examples
    --------
    >>> penalty = calculate_diversity_penalty(['toy', 'puzzle'], 'toy')
    >>> print(f"Diversity penalty: {penalty}")
    """
    count = recommended_categories.count(new_category)
    penalty = count * penalty_factor

    logger.debug(f"Diversity penalty for '{new_category}': {penalty:.3f}")
    return penalty
