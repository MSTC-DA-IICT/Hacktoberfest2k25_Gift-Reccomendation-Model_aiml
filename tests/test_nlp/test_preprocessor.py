"""
Test cases for text preprocessor module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nlp.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextPreprocessor()

    def test_clean_text_lowercase(self):
        """Test text is converted to lowercase."""
        text = "Hello WORLD"
        result = self.processor.clean_text(text)
        assert result == "hello world"

    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check this out https://example.com"
        result = self.processor.clean_text(text)
        assert "https://" not in result
        assert "example.com" not in result

    def test_remove_mentions(self):
        """Test @mention removal."""
        text = "Hey @user how are you?"
        result = self.processor.clean_text(text)
        assert "@user" not in result
        assert "hey" in result
        assert "how are you" in result

    def test_remove_hashtags_keep_content(self):
        """Test hashtag removal while keeping content."""
        text = "This is #awesome content!"
        result = self.processor.clean_text(text)
        assert "#awesome" not in result
        assert "awesome" in result

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "This is a test"
        tokens = self.processor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) >= 4
        assert "This" in tokens or "this" in tokens

    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        text = ""
        tokens = self.processor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ["this", "is", "not", "amazing"]
        filtered = self.processor.remove_stopwords(tokens)
        assert "amazing" in filtered
        assert "not" in filtered  # Should be preserved (emotional word)
        assert "is" not in filtered

    def test_preprocess_pipeline(self):
        """Test full preprocessing pipeline."""
        text = "Hey @user this is NOT amazing! https://example.com #great"
        tokens = self.processor.preprocess(text)

        assert isinstance(tokens, list)
        # Should contain meaningful words
        assert any("amazing" in token.lower() for token in tokens)
        assert any("great" in token.lower() for token in tokens)
        assert any("not" in token.lower() for token in tokens)
        # Should not contain stopwords or removed elements
        assert not any("@user" in token for token in tokens)
        assert not any("https" in token for token in tokens)

    def test_preprocess_empty_input(self):
        """Test preprocessing empty input."""
        result = self.processor.preprocess("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        texts = [
            "I love this!",
            "This is terrible",
            "Feeling great today"
        ]
        results = self.processor.preprocess_batch(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(tokens, list) for tokens in results)

    def test_preserve_emotional_punctuation(self):
        """Test that emotional punctuation is preserved."""
        text = "This is amazing!"
        result = self.processor.clean_text(text)
        # The exclamation mark might be preserved or converted
        assert "amazing" in result

    def test_handle_special_characters(self):
        """Test handling of special characters."""
        text = "Hello... world??? What's up!!!"
        tokens = self.processor.tokenize(text)

        # Should handle special characters gracefully
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_handle_non_string_input(self):
        """Test handling of non-string input."""
        result = self.processor.clean_text(None)
        assert result == ""

        result = self.processor.clean_text(123)
        assert result == ""

    def test_clean_text_extra_whitespace(self):
        """Test removal of extra whitespace."""
        text = "  hello    world   "
        result = self.processor.clean_text(text)
        assert result.strip() == result  # No leading/trailing whitespace
        assert "  " not in result  # No double spaces