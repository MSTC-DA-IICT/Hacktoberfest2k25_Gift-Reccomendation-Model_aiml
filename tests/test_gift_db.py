import json
import pytest
from pathlib import Path

# Get the path to the directory containing this test file
current_dir = Path(__file__).parent

GIFT_DB_PATH = current_dir.parent / "config" / "gift_mapping.json"

@pytest.fixture(scope="module")
def gift_data():
    """
    A pytest fixture to load the gift database from the JSON file.
    This runs once per test module and shares the data across all tests.
    """
    try:
        with open(GIFT_DB_PATH, 'r') as f:
            data = json.load(f)
        return data["gifts"]
    except FileNotFoundError:
        pytest.fail(f"The gift database file was not found at: {GIFT_DB_PATH}")
    except json.JSONDecodeError:
        pytest.fail(f"The file at {GIFT_DB_PATH} is not a valid JSON file.")


def test_no_duplicate_ids(gift_data):
    """
    Ensures that all gift IDs are unique.
    """
    ids = [gift["id"] for gift in gift_data]
    assert len(ids) == len(set(ids)), "Found duplicate gift IDs in the database."


def test_mandatory_keys(gift_data):
    """
    Verifies that each gift entry contains all the required keys.
    """

    mandatory_keys = {"id", "name", "category", "sentiment_min", "sentiment_max"}
    
    for gift in gift_data:
        gift_keys = set(gift.keys())
        assert mandatory_keys.issubset(gift_keys), \
            f"Gift with id '{gift.get('id', 'N/A')}' is missing one or more mandatory keys."


def test_sentiment_logic(gift_data):
    """
    Checks that sentiment values are valid numbers within the range [-1.0, 1.0]
    and that min is less than or equal to max.
    """
   
    for gift in gift_data:
        gift_id = gift.get('id', 'N/A')
        min_sentiment = gift.get("sentiment_min")
        max_sentiment = gift.get("sentiment_max")

        # 1. Check that the keys exist and their values are numbers
        assert isinstance(min_sentiment, (int, float)), f"sentiment_min for gift id '{gift_id}' is not a number."
        assert isinstance(max_sentiment, (int, float)), f"sentiment_max for gift id '{gift_id}' is not a number."
        
        # 2. Check that both values are within the valid range
        assert -1.0 <= min_sentiment <= 1.0, f"sentiment_min for gift id '{gift_id}' is out of range [-1.0, 1.0]."
        assert -1.0 <= max_sentiment <= 1.0, f"sentiment_max for gift id '{gift_id}' is out of range [-1.0, 1.0]."

        # 3. Check that min is not greater than max
        assert min_sentiment <= max_sentiment, f"sentiment_min is greater than sentiment_max for gift id '{gift_id}'."