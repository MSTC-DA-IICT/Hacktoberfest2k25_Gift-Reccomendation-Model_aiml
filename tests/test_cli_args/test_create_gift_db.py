import subprocess
import sys
import pytest


SCRIPT_PATH = "scripts/create_gift_db.py"
INPUT_FILE_PATH = "config/gift_mapping.json"


class TestCreateGiftDbCLI:
    """Tests for the create_gift_db.py command-line interface."""

    def test_analyze_no_filter(self):
        """
        Tests that running in 'analyze' mode without a filter
        shows multiple categories in the output.
        """

        result = subprocess.run(
            [sys.executable, SCRIPT_PATH, "--mode", "analyze", "-i", INPUT_FILE_PATH],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout.lower()

        assert "gadget" in output
        assert "book" in output
        assert "toy" in output

    def test_analyze_with_category_filter(self):
        """
        Tests that the --filter-category argument correctly filters the output
        to show only the specified category.
        """
        category_to_filter = "gadget"

        result = subprocess.run(
            [
                sys.executable,
                SCRIPT_PATH,
                "--mode",
                "analyze",
                "-i",
                INPUT_FILE_PATH,
                "--filter-category",
                category_to_filter,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout.lower()

        assert "gadget" in output

        assert "book" not in output
        assert "toy" not in output

    def test_analyze_with_nonexistent_category(self):
        """
        Tests that filtering by a category that doesn't exist
        prints a warning and shows 0 total gifts.
        """
        category_to_filter = "nonexistent_category"

        result = subprocess.run(
            [
                sys.executable,
                SCRIPT_PATH,
                "--mode",
                "analyze",
                "-i",
                INPUT_FILE_PATH,
                "--filter-category",
                category_to_filter,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "no gifts found" in result.stderr.lower()
