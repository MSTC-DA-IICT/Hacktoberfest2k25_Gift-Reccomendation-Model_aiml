#!/usr/bin/env python3
"""
Create and manage gift database.

This script helps create, update, and manage the gift database
used by the recommendation system.

Usage:
    python scripts/create_gift_db.py --output config/gift_mapping.json \
                                    --mode create
    python scripts/create_gift_db.py --input config/gift_mapping.json \
                                    --mode validate
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommendation.gift_database import GiftDatabase
from utils.logger import setup_logger
from utils.validators import validate_gift_dict


def create_default_gift_database() -> List[Dict]:
    """Create default gift database with sample items."""
    gifts = [
        # Small hands - Positive sentiment
        {
            "id": 1,
            "name": "Colorful Building Blocks",
            "category": "toy",
            "hand_size": "small",
            "sentiment_min": 0.6,
            "sentiment_max": 1.0,
            "price_range": "$15-$30",
            "description": "Creative building blocks for imaginative play",
        },
        {
            "id": 2,
            "name": "Adventure Storybook",
            "category": "book",
            "hand_size": "small",
            "sentiment_min": 0.5,
            "sentiment_max": 1.0,
            "price_range": "$10-$20",
            "description": "Exciting adventure stories for young readers",
        },
        {
            "id": 3,
            "name": "Art Supply Kit",
            "category": "art_supplies",
            "hand_size": "small",
            "sentiment_min": 0.6,
            "sentiment_max": 1.0,
            "price_range": "$20-$40",
            "description": "Complete art kit with crayons, markers, and paper",
        },
        # Small hands - Negative sentiment
        {
            "id": 4,
            "name": "Comfort Teddy Bear",
            "category": "comfort_item",
            "hand_size": "small",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$15-$25",
            "description": "Soft, cuddly teddy bear for comfort",
        },
        {
            "id": 5,
            "name": "Calming Picture Book",
            "category": "book",
            "hand_size": "small",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$8-$15",
            "description": "Gentle, soothing stories for relaxation",
        },
        # Medium hands - Positive sentiment
        {
            "id": 6,
            "name": "Stylish Watch",
            "category": "accessory",
            "hand_size": "medium",
            "sentiment_min": 0.6,
            "sentiment_max": 1.0,
            "price_range": "$50-$120",
            "description": "Modern, fashionable timepiece",
        },
        {
            "id": 7,
            "name": "Wireless Earbuds",
            "category": "gadget",
            "hand_size": "medium",
            "sentiment_min": 0.5,
            "sentiment_max": 1.0,
            "price_range": "$40-$80",
            "description": "High-quality wireless audio experience",
        },
        {
            "id": 8,
            "name": "Sports Water Bottle",
            "category": "sports",
            "hand_size": "medium",
            "sentiment_min": 0.6,
            "sentiment_max": 1.0,
            "price_range": "$20-$35",
            "description": "Insulated bottle for active lifestyle",
        },
        # Medium hands - Negative sentiment
        {
            "id": 9,
            "name": "Cozy Throw Pillow",
            "category": "comfort_item",
            "hand_size": "medium",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$25-$45",
            "description": "Soft pillow for relaxation and comfort",
        },
        {
            "id": 10,
            "name": "Aromatherapy Candle",
            "category": "home_decor",
            "hand_size": "medium",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$15-$30",
            "description": "Relaxing scented candle for mood enhancement",
        },
        # Large hands - Positive sentiment
        {
            "id": 11,
            "name": "Premium Leather Wallet",
            "category": "accessory",
            "hand_size": "large",
            "sentiment_min": 0.6,
            "sentiment_max": 1.0,
            "price_range": "$80-$150",
            "description": "Elegant leather wallet with multiple compartments",
        },
        {
            "id": 12,
            "name": "Smart Fitness Tracker",
            "category": "gadget",
            "hand_size": "large",
            "sentiment_min": 0.5,
            "sentiment_max": 1.0,
            "price_range": "$100-$200",
            "description": "Advanced fitness tracking with heart rate monitor",
        },
        {
            "id": 13,
            "name": "Professional Tool Set",
            "category": "tool",
            "hand_size": "large",
            "sentiment_min": 0.6,
            "sentiment_max": 1.0,
            "price_range": "$120-$250",
            "description": "Complete tool set for DIY projects",
        },
        # Large hands - Negative sentiment
        {
            "id": 14,
            "name": "Electric Kettle",
            "category": "appliance",
            "hand_size": "large",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$40-$80",
            "description": "Quick-heating kettle for hot beverages",
        },
        {
            "id": 15,
            "name": "Memory Foam Pillow",
            "category": "comfort_item",
            "hand_size": "large",
            "sentiment_min": 0.0,
            "sentiment_max": 0.5,
            "price_range": "$35-$65",
            "description": "Ergonomic pillow for better sleep",
        },
        # Cross-category items
        {
            "id": 16,
            "name": "Bluetooth Speaker",
            "category": "gadget",
            "hand_size": "medium",
            "sentiment_min": 0.4,
            "sentiment_max": 1.0,
            "price_range": "$60-$120",
            "description": "Portable speaker with excellent sound quality",
        },
        {
            "id": 17,
            "name": "Gourmet Tea Set",
            "category": "home",
            "hand_size": "medium",
            "sentiment_min": 0.3,
            "sentiment_max": 0.8,
            "price_range": "$30-$60",
            "description": "Premium tea collection with accessories",
        },
        # Additional gifts for expanded database
        {
            "id": 18,
            "name": "Digital Photo Frame",
            "category": "tech",
            "hand_size": "medium",
            "sentiment_min": 0.4,
            "sentiment_max": 1.0,
            "price_range": "$45-$90",
            "description": "Smart digital frame for displaying memories",
        },
        {
            "id": 19,
            "name": "Handmade Pottery Mug",
            "category": "home_decor",
            "hand_size": "large",
            "sentiment_min": 0.2,
            "sentiment_max": 0.7,
            "price_range": "$25-$50",
            "description": "Unique ceramic mug crafted by local artisans",
        },
        {
            "id": 20,
            "name": "Puzzle Game Set",
            "category": "toy",
            "hand_size": "small",
            "sentiment_min": 0.5,
            "sentiment_max": 0.9,
            "price_range": "$18-$35",
            "description": "Educational puzzle collection for brain training",
        },
        {
            "id": 21,
            "name": "Yoga Mat & Accessories",
            "category": "sports",
            "hand_size": "large",
            "sentiment_min": 0.3,
            "sentiment_max": 0.8,
            "price_range": "$40-$80",
            "description": "Premium yoga mat with blocks and straps",
        },
        {
            "id": 22,
            "name": "Scented Bath Bomb Collection",
            "category": "home",
            "hand_size": "medium",
            "sentiment_min": 0.0,
            "sentiment_max": 0.6,
            "price_range": "$20-$40",
            "description": "Luxurious bath bombs for relaxation and self-care",
        },
    ]

    return gifts


def validate_database(gifts: List[Dict], logger) -> bool:
    """Validate entire gift database."""
    try:
        valid = True
        issues = []

        # Validate each gift
        for i, gift in enumerate(gifts):
            if not validate_gift_dict(gift):
                issues.append(f"Gift {i} (ID: {gift.get('id', 'unknown')}) is invalid")
                valid = False

        # Check for duplicate IDs
        ids = [gift.get("id") for gift in gifts]
        if len(ids) != len(set(ids)):
            issues.append("Duplicate gift IDs found")
            valid = False

        # Check category distribution
        categories = [gift.get("category") for gift in gifts]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        logger.info(f"Category distribution: {category_counts}")

        # Check size distribution
        sizes = [gift.get("hand_size") for gift in gifts]
        size_counts = {}
        for size in sizes:
            size_counts[size] = size_counts.get(size, 0) + 1

        logger.info(f"Size distribution: {size_counts}")

        # Log issues
        if issues:
            logger.error("Database validation issues:")
            for issue in issues:
                logger.error(f"  - {issue}")

        return valid

    except Exception as e:
        logger.error(f"Error validating database: {e}")
        return False


def expand_database(existing_gifts: List[Dict], logger) -> List[Dict]:
    """Expand existing database with additional items."""
    try:
        logger.info("Expanding gift database with additional items...")

        # Find the highest existing ID
        max_id = max([gift.get("id", 0) for gift in existing_gifts])
        next_id = max_id + 1

        additional_gifts = [
            # Books
            {
                "id": next_id,
                "name": "Cookbook for Beginners",
                "category": "book",
                "hand_size": "medium",
                "sentiment_min": 0.0,
                "sentiment_max": 0.7,
                "price_range": "$20-$35",
                "description": "Easy recipes for cooking enthusiasts",
            },
            {
                "id": next_id + 1,
                "name": "Motivational Journal",
                "category": "book",
                "hand_size": "small",
                "sentiment_min": 0.4,
                "sentiment_max": 1.0,
                "price_range": "$12-$22",
                "description": "Inspiring journal for daily reflection",
            },
            # Tech gadgets
            {
                "id": next_id + 2,
                "name": "Portable Phone Charger",
                "category": "tech",
                "hand_size": "medium",
                "sentiment_min": 0.3,
                "sentiment_max": 1.0,
                "price_range": "$25-$45",
                "description": "Compact power bank for mobile devices",
            },
            {
                "id": next_id + 3,
                "name": "LED Desk Lamp",
                "category": "tech",
                "hand_size": "large",
                "sentiment_min": 0.2,
                "sentiment_max": 0.8,
                "price_range": "$35-$70",
                "description": "Adjustable LED lamp with USB charging",
            },
            # Home items
            {
                "id": next_id + 4,
                "name": "Indoor Plant Kit",
                "category": "home",
                "hand_size": "medium",
                "sentiment_min": 0.1,
                "sentiment_max": 0.6,
                "price_range": "$20-$40",
                "description": "Complete kit for growing indoor plants",
            },
            {
                "id": next_id + 5,
                "name": "Decorative Wall Art",
                "category": "home_decor",
                "hand_size": "large",
                "sentiment_min": 0.0,
                "sentiment_max": 0.7,
                "price_range": "$30-$80",
                "description": "Beautiful artwork for home decoration",
            },
        ]

        # Validate new gifts
        for gift in additional_gifts:
            if not validate_gift_dict(gift):
                logger.error(f"Invalid additional gift: {gift}")
                continue

        expanded_gifts = existing_gifts + additional_gifts
        logger.info(f"Added {len(additional_gifts)} new gifts to database")

        return expanded_gifts

    except Exception as e:
        logger.error(f"Error expanding database: {e}")
        return existing_gifts


def analyze_database(gifts: List[Dict], logger) -> Dict:
    """Analyze gift database and provide statistics."""
    try:
        stats = {
            "total_gifts": len(gifts),
            "categories": {},
            "hand_sizes": {},
            "sentiment_ranges": {},
            "price_ranges": set(),
        }

        for gift in gifts:
            # Category count
            category = gift.get("category", "unknown")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

            # Hand size count
            size = gift.get("hand_size", "unknown")
            stats["hand_sizes"][size] = stats["hand_sizes"].get(size, 0) + 1

            # Sentiment range analysis
            sent_min = gift.get("sentiment_min", 0)
            sent_max = gift.get("sentiment_max", 1)
            range_key = f"{sent_min:.1f}-{sent_max:.1f}"
            stats["sentiment_ranges"][range_key] = (
                stats["sentiment_ranges"].get(range_key, 0) + 1
            )

            # Price ranges
            price_range = gift.get("price_range", "N/A")
            stats["price_ranges"].add(price_range)

        stats["price_ranges"] = sorted(list(stats["price_ranges"]))

        logger.info("Database Analysis:")
        logger.info(f"  Total gifts: {stats['total_gifts']}")
        logger.info(f"  Categories: {len(stats['categories'])}")
        logger.info(f"  Hand sizes covered: {list(stats['hand_sizes'].keys())}")

        return stats

    except Exception as e:
        logger.error(f"Error analyzing database: {e}")
        return {}


def get_filtered_gifts(gifts: List[Dict], category: str, logger) -> List[Dict]:
    """
    Filter a list of gifts by a specific category.

    Parameters
    ----------
    gifts : List[Dict]
        The list of gift dictionaries to filter.
    category : str
        The category to filter by (case-insensitive).
    logger : logging.Logger
        The logger instance for logging messages.

    Returns
    -------
    List[Dict]
        A new list containing only the gifts that match the category.
    """
    logger.info(f"Filtering database for category: '{category}'")

    # Use a list comprehension for efficient and readable filtering
    filtered_list = [
        gift for gift in gifts if gift.get("category", "").lower() == category.lower()
    ]

    original_count = len(gifts)
    filtered_count = len(filtered_list)
    logger.info(f"Filtered {original_count} gifts down to {filtered_count}.")

    return filtered_list


def main():
    parser = argparse.ArgumentParser(description="Create and manage gift database")
    parser.add_argument(
        "--mode",
        choices=["create", "validate", "expand", "analyze"],
        default="create",
        help="Operation mode",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input gift database file (for validate/expand/analyze modes)",
    )
    parser.add_argument("--output", "-o", help="Output file path for new database")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--filter-category", type=str, help="Filter gifts by a specific category"
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(__name__)
    if args.verbose:
        logger.setLevel("DEBUG")

    logger.info(f"Starting gift database management in {args.mode} mode")

    try:
        if args.mode == "create":
            # Create new database
            if not args.output:
                args.output = "config/gift_mapping.json"

            logger.info("Creating default gift database...")
            gifts = create_default_gift_database()

            # Validate created database
            if not validate_database(gifts, logger):
                logger.error("Created database failed validation")
                return 1

            # Create database structure
            database_data = {"gifts": gifts}

            # Create output directory
            os.makedirs(os.path.dirname(args.output), exist_ok=True)

            # Save to file
            with open(args.output, "w") as f:
                json.dump(database_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Gift database created with {len(gifts)} items: {args.output}")

            # Analyze created database
            stats = analyze_database(gifts, logger)

        elif args.mode == "validate":
            # Validate existing database
            if not args.input:
                raise ValueError("Input file required for validate mode")

            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")

            logger.info(f"Validating gift database: {args.input}")

            with open(args.input, "r") as f:
                data = json.load(f)

            gifts = data.get("gifts", [])

            if validate_database(gifts, logger):
                logger.info("Database validation passed!")
            else:
                logger.error("Database validation failed!")
                return 1

            # Analyze database
            stats = analyze_database(gifts, logger)

        elif args.mode == "expand":
            # Expand existing database
            if not args.input:
                raise ValueError("Input file required for expand mode")

            if not args.output:
                args.output = args.input.replace(".json", "_expanded.json")

            logger.info(f"Expanding gift database: {args.input}")

            with open(args.input, "r") as f:
                data = json.load(f)

            existing_gifts = data.get("gifts", [])
            expanded_gifts = expand_database(existing_gifts, logger)

            # Validate expanded database
            if not validate_database(expanded_gifts, logger):
                logger.error("Expanded database failed validation")
                return 1

            # Save expanded database
            expanded_data = {"gifts": expanded_gifts}

            with open(args.output, "w") as f:
                json.dump(expanded_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Expanded database saved: {args.output}")

            # Analyze expanded database
            stats = analyze_database(expanded_gifts, logger)

        elif args.mode == "analyze":
            # Analyze existing database
            if not args.input:
                raise ValueError("Input file required for analyze mode")

            logger.info(f"Analyzing gift database: {args.input}")

            with open(args.input, "r") as f:
                data = json.load(f)

            gifts = data.get("gifts", [])
            if args.filter_category:
                gifts = get_filtered_gifts(gifts, args.filter_category, logger)
                if not gifts:
                    logger.warning(
                        f"No gifts found in category '{args.filter_category}'"
                    )
                    return 0
            stats = analyze_database(gifts, logger)

            # Print detailed analysis
            print("\nDETAILED ANALYSIS")
            print("=" * 20)
            print(f"Total gifts: {stats['total_gifts']}")
            print("\nCategory distribution:")
            for category, count in sorted(stats["categories"].items()):
                print(f"  {category}: {count}")

            print("\nHand size distribution:")
            for size, count in sorted(stats["hand_sizes"].items()):
                print(f"  {size}: {count}")

            print("\nSentiment range distribution:")
            for range_key, count in sorted(stats["sentiment_ranges"].items()):
                print(f"  {range_key}: {count}")

            print("\nPrice ranges:")
            for price_range in stats["price_ranges"]:
                print(f"  {price_range}")

        logger.info(f"Gift database {args.mode} operation completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
