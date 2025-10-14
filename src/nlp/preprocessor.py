"""
Text preprocessing module for sentiment analysis.

This module handles text cleaning, tokenization, and preprocessing
specifically designed for tweet-like social media content.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """
    Handles text preprocessing for sentiment analysis.

    This class provides methods to clean, tokenize, and preprocess text data,
    with special handling for social media content like tweets.
    """

    def __init__(self):
        """Initialize the preprocessor with NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))

        # Preserve some emotionally significant words that are typically stopwords
        emotional_words = {'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody', 'none'}
        self.stop_words = self.stop_words - emotional_words

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, mentions, hashtags, and special characters.

        Parameters
        ----------
        text : str
            Raw text input

        Returns
        -------
        str
            Cleaned text

        Examples
        --------
        >>> processor = TextPreprocessor()
        >>> processor.clean_text("Hey @user check https://example.com #awesome")
        'hey check awesome'
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove mentions (@username)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)

        # Remove hashtags but keep the content
        text = re.sub(r'#([A-Za-z0-9_]+)', r'\\1', text)

        # Remove extra whitespace and newlines
        text = re.sub(r'\\s+', ' ', text)
        text = re.sub(r'\\n', ' ', text)

        # Remove punctuation except for emotionally significant ones
        # Keep ! and ? as they can indicate sentiment
        text = re.sub(r'[^\\w\\s!?]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        List[str]
            List of tokens

        Examples
        --------
        >>> processor = TextPreprocessor()
        >>> processor.tokenize("This is a test")
        ['This', 'is', 'a', 'test']
        """
        if not text:
            return []

        try:
            tokens = word_tokenize(text)
            # Filter out empty strings and single characters (except emotionally significant ones)
            tokens = [token for token in tokens if len(token) > 1 or token in ['!', '?']]
            return tokens
        except Exception:
            # Fallback to simple split if NLTK fails
            return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove common stopwords while preserving emotionally significant words.

        Parameters
        ----------
        tokens : List[str]
            List of tokens

        Returns
        -------
        List[str]
            Filtered tokens

        Examples
        --------
        >>> processor = TextPreprocessor()
        >>> processor.remove_stopwords(['this', 'is', 'not', 'amazing'])
        ['not', 'amazing']
        """
        if not tokens:
            return []

        # Filter out stopwords but keep emotionally significant punctuation
        filtered_tokens = [
            token for token in tokens
            if token.lower() not in self.stop_words or token in ['!', '?']
        ]

        return filtered_tokens

    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline.

        Applies cleaning, tokenization, and stopword removal in sequence.

        Parameters
        ----------
        text : str
            Raw input text

        Returns
        -------
        List[str]
            Preprocessed tokens ready for feature extraction

        Examples
        --------
        >>> processor = TextPreprocessor()
        >>> processor.preprocess("Hey @user this is amazing! https://example.com")
        ['amazing', '!']
        """
        if not text:
            return []

        # Apply full preprocessing pipeline
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        filtered_tokens = self.remove_stopwords(tokens)

        return filtered_tokens

    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess a batch of texts.

        Parameters
        ----------
        texts : List[str]
            List of raw texts

        Returns
        -------
        List[List[str]]
            List of preprocessed token lists
        """
        return [self.preprocess(text) for text in texts]