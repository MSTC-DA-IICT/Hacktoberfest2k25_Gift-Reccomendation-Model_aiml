"""
Gift database management module.

This module provides functionality to load, query, and manage
the gift database for recommendations.
"""

import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class GiftDatabase:
    """
    Interface to gift database.

    This class provides methods to load and query gift data
    for the recommendation system.
    """

    def __init__(self, config_path: str):
        """
        Initialize gift database.

        Parameters
        ----------
        config_path : str
            Path to gift configuration JSON file
        """
        self.gifts: List[Dict] = []
        self.config_path = config_path
        self._gift_by_id: Dict[int, Dict] = {}
        self.load_gifts(config_path)

        logger.info(f"GiftDatabase initialized with {len(self.gifts)} gifts")

    def load_gifts(self, path: str) -> None:
        """
        Load gifts from JSON configuration file.

        Parameters
        ----------
        path : str
            Path to JSON file

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist
        ValueError
            If config file is invalid
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if "gifts" not in data:
                raise ValueError("Config file must contain 'gifts' key")

            self.gifts = data["gifts"]

            # Create lookup dictionary by ID
            self._gift_by_id = {gift["id"]: gift for gift in self.gifts}

            # Validate gift data
            self._validate_gifts()

            logger.info(f"Loaded {len(self.gifts)} gifts from {path}")

        except FileNotFoundError:
            logger.error(f"Gift config file not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {path}: {e}")
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading gifts: {e}")
            raise

    def _validate_gifts(self) -> None:
        """Validate gift data structure."""
        required_fields = [
            "id",
            "name",
            "category",
            "hand_size",
            "sentiment_min",
            "sentiment_max",
        ]

        for i, gift in enumerate(self.gifts):
            for field in required_fields:
                if field not in gift:
                    raise ValueError(f"Gift {i} missing required field: {field}")

            # Validate sentiment range
            if not (0 <= gift["sentiment_min"] <= gift["sentiment_max"] <= 1):
                raise ValueError(f"Gift {gift['id']} has invalid sentiment range")

            # Validate hand size
            if gift["hand_size"] not in ["small", "medium", "large"]:
                raise ValueError(
                    f"Gift {gift['id']} has invalid hand size: {gift['hand_size']}"
                )

    def get_all_gifts(self) -> List[Dict]:
        """
        Get all gifts in database.

        Returns
        -------
        List[Dict]
            List of all gift dictionaries

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> all_gifts = db.get_all_gifts()
        >>> print(f"Total gifts: {len(all_gifts)}")
        """
        return self.gifts.copy()

    def filter_by_size(self, size: str) -> List[Dict]:
        """
        Filter gifts by hand size.

        Parameters
        ----------
        size : str
            Hand size ('small', 'medium', 'large')

        Returns
        -------
        List[Dict]
            Filtered gifts matching the size

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> small_gifts = db.filter_by_size('small')
        >>> print(f"Found {len(small_gifts)} gifts for small hands")
        """
        if size not in ["small", "medium", "large"]:
            logger.warning(f"Invalid hand size: {size}")
            return []

        filtered = [gift for gift in self.gifts if gift["hand_size"] == size]
        logger.debug(f"Found {len(filtered)} gifts for size '{size}'")
        return filtered

    def filter_by_sentiment(self, sentiment: float) -> List[Dict]:
        """
        Filter gifts by sentiment score.

        Parameters
        ----------
        sentiment : float
            Sentiment score between 0 and 1

        Returns
        -------
        List[Dict]
            Filtered gifts matching the sentiment range

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> happy_gifts = db.filter_by_sentiment(0.8)
        >>> print(f"Found {len(happy_gifts)} gifts for sentiment 0.8")
        """
        if not (0 <= sentiment <= 1):
            logger.warning(f"Invalid sentiment score: {sentiment}")
            return []

        filtered = [
            gift
            for gift in self.gifts
            if gift["sentiment_min"] <= sentiment <= gift["sentiment_max"]
        ]
        logger.debug(f"Found {len(filtered)} gifts for sentiment {sentiment:.2f}")
        return filtered

    def filter_by_size_and_sentiment(self, size: str, sentiment: float) -> List[Dict]:
        """
        Filter gifts matching both size and sentiment criteria.

        Parameters
        ----------
        size : str
            Hand size ('small', 'medium', 'large')
        sentiment : float
            Sentiment score between 0 and 1

        Returns
        -------
        List[Dict]
            Filtered gifts matching both criteria

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> matches = db.filter_by_size_and_sentiment('medium', 0.7)
        >>> print(f"Found {len(matches)} matching gifts")
        """
        if size not in ["small", "medium", "large"]:
            logger.warning(f"Invalid hand size: {size}")
            return []

        if not (0 <= sentiment <= 1):
            logger.warning(f"Invalid sentiment score: {sentiment}")
            return []

        filtered = [
            gift
            for gift in self.gifts
            if (
                gift["hand_size"] == size
                and gift["sentiment_min"] <= sentiment <= gift["sentiment_max"]
            )
        ]

        logger.debug(
            f"Found {len(filtered)} gifts for size '{size}' and sentiment {sentiment:.2f}"
        )
        return filtered

    def get_gift_by_id(self, gift_id: int) -> Optional[Dict]:
        """
        Get gift by ID.

        Parameters
        ----------
        gift_id : int
            Gift ID

        Returns
        -------
        Optional[Dict]
            Gift dictionary or None if not found

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> gift = db.get_gift_by_id(1)
        >>> if gift:
        ...     print(f"Gift: {gift['name']}")
        """
        return self._gift_by_id.get(gift_id)

    def filter_by_category(self, category: str) -> List[Dict]:
        """
        Filter gifts by category.

        Parameters
        ----------
        category : str
            Gift category

        Returns
        -------
        List[Dict]
            Filtered gifts in the category

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> toys = db.filter_by_category('toy')
        >>> print(f"Found {len(toys)} toys")
        """
        filtered = [gift for gift in self.gifts if gift["category"] == category]
        logger.debug(f"Found {len(filtered)} gifts in category '{category}'")
        return filtered

    def search_gifts(self, query: str) -> List[Dict]:
        """
        Search gifts by name or description.

        Parameters
        ----------
        query : str
            Search query

        Returns
        -------
        List[Dict]
            Matching gifts

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> results = db.search_gifts('book')
        >>> print(f"Found {len(results)} gifts matching 'book'")
        """
        if not query:
            return []

        query_lower = query.lower()
        matches = []

        for gift in self.gifts:
            # Search in name and description
            if (
                query_lower in gift["name"].lower()
                or query_lower in gift.get("description", "").lower()
                or query_lower in gift["category"].lower()
            ):
                matches.append(gift)

        logger.debug(f"Found {len(matches)} gifts matching query '{query}'")
        return matches

    def add_gift(self, gift: Dict) -> None:
        """
        Add a new gift to the database.

        Parameters
        ----------
        gift : Dict
            Gift dictionary with required fields

        Raises
        ------
        ValueError
            If gift data is invalid or ID already exists

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> new_gift = {
        ...     'id': 99,
        ...     'name': 'New Gift',
        ...     'category': 'misc',
        ...     'hand_size': 'medium',
        ...     'sentiment_min': 0.0,
        ...     'sentiment_max': 1.0,
        ...     'price_range': '$10-$50',
        ...     'description': 'A new gift item'
        ... }
        >>> db.add_gift(new_gift)
        """
        # Validate required fields
        required_fields = [
            "id",
            "name",
            "category",
            "hand_size",
            "sentiment_min",
            "sentiment_max",
        ]
        for field in required_fields:
            if field not in gift:
                raise ValueError(f"Gift missing required field: {field}")

        # Check if ID already exists
        if gift["id"] in self._gift_by_id:
            raise ValueError(f"Gift with ID {gift['id']} already exists")

        # Validate sentiment range
        if not (0 <= gift["sentiment_min"] <= gift["sentiment_max"] <= 1):
            raise ValueError("Invalid sentiment range")

        # Validate hand size
        if gift["hand_size"] not in ["small", "medium", "large"]:
            raise ValueError(f"Invalid hand size: {gift['hand_size']}")

        # Add gift
        self.gifts.append(gift.copy())
        self._gift_by_id[gift["id"]] = gift.copy()

        logger.info(f"Added new gift: {gift['name']} (ID: {gift['id']})")

    def remove_gift(self, gift_id: int) -> bool:
        """
        Remove gift by ID.

        Parameters
        ----------
        gift_id : int
            Gift ID to remove

        Returns
        -------
        bool
            True if removed, False if not found

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> success = db.remove_gift(99)
        >>> print(f"Removed: {success}")
        """
        if gift_id not in self._gift_by_id:
            logger.warning(f"Gift ID {gift_id} not found")
            return False

        # Remove from main list
        self.gifts = [gift for gift in self.gifts if gift["id"] != gift_id]

        # Remove from lookup dict
        del self._gift_by_id[gift_id]

        logger.info(f"Removed gift with ID {gift_id}")
        return True

    def get_categories(self) -> List[str]:
        """
        Get all unique gift categories.

        Returns
        -------
        List[str]
            List of unique categories

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> categories = db.get_categories()
        >>> print(f"Categories: {categories}")
        """
        categories = list(set(gift["category"] for gift in self.gifts))
        return sorted(categories)

    def get_size_distribution(self) -> Dict[str, int]:
        """
        Get distribution of gifts by hand size.

        Returns
        -------
        Dict[str, int]
            Count of gifts for each hand size

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> distribution = db.get_size_distribution()
        >>> print(distribution)  # {'small': 2, 'medium': 2, 'large': 2}
        """
        distribution = {"small": 0, "medium": 0, "large": 0}

        for gift in self.gifts:
            size = gift["hand_size"]
            if size in distribution:
                distribution[size] += 1

        return distribution

    def get_price_range_stats(self) -> Dict[str, Any]:
        """
        Get statistics about price ranges.

        Returns
        -------
        Dict[str, Any]
            Price range statistics

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> stats = db.get_price_range_stats()
        >>> print(f"Price ranges: {stats['unique_ranges']}")
        """
        price_ranges = [gift.get("price_range", "N/A") for gift in self.gifts]
        unique_ranges = list(set(price_ranges))

        stats = {
            "unique_ranges": sorted(unique_ranges),
            "total_gifts": len(self.gifts),
            "gifts_without_price": price_ranges.count("N/A"),
        }

        return stats

    def save_to_file(self, output_path: str) -> None:
        """
        Save current gift database to JSON file.

        Parameters
        ----------
        output_path : str
            Output file path

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> # After modifications...
        >>> db.save_to_file('updated_gifts.json')
        """
        try:
            data = {"gifts": self.gifts}

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved gift database to {output_path}")

        except Exception as e:
            logger.error(f"Error saving gift database: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.

        Returns
        -------
        Dict[str, Any]
            Database statistics

        Examples
        --------
        >>> db = GiftDatabase('gifts.json')
        >>> stats = db.get_statistics()
        >>> print(f"Total gifts: {stats['total_gifts']}")
        """
        stats = {
            "total_gifts": len(self.gifts),
            "categories": self.get_categories(),
            "size_distribution": self.get_size_distribution(),
            "price_stats": self.get_price_range_stats(),
        }

        # Sentiment range statistics
        sentiments = [(g["sentiment_min"], g["sentiment_max"]) for g in self.gifts]
        stats["sentiment_ranges"] = {
            "min_sentiment": min(s[0] for s in sentiments) if sentiments else 0,
            "max_sentiment": max(s[1] for s in sentiments) if sentiments else 1,
            "avg_range_size": (
                sum(s[1] - s[0] for s in sentiments) / len(sentiments)
                if sentiments
                else 0
            ),
        }

        return stats
