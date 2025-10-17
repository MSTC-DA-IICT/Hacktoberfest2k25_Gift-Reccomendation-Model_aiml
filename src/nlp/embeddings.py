"""
Word2Vec embeddings module for text representation.

This module provides a wrapper around Gensim's Word2Vec implementation
with additional utilities for sentence-level embeddings and model management.
"""

import pickle
from typing import List, Optional, Union
import numpy as np
from gensim.models import Word2Vec
import logging

logger = logging.getLogger(__name__)


class Word2VecEmbeddings:
    """
    Word2Vec embeddings for text representation.

    This class provides methods to train Word2Vec models and convert
    text tokens into dense vector representations.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ):
        """
        Initialize Word2Vec embeddings.

        Parameters
        ----------
        vector_size : int, default=100
            Dimensionality of word vectors
        window : int, default=5
            Maximum distance between current and predicted word
        min_count : int, default=2
            Minimum word frequency threshold
        workers : int, default=4
            Number of worker threads
        epochs : int, default=10
            Number of training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model: Optional[Word2Vec] = None
        self._is_trained = False

    def train(self, sentences: List[List[str]]) -> None:
        """
        Train Word2Vec model on corpus.

        Parameters
        ----------
        sentences : List[List[str]]
            List of tokenized sentences

        Raises
        ------
        ValueError
            If sentences is empty or invalid
        """
        if not sentences or len(sentences) == 0:
            raise ValueError("Cannot train on empty sentences list")

        logger.info(f"Training Word2Vec model on {len(sentences)} sentences...")

        try:
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                epochs=self.epochs,
                sg=0,  # Use CBOW
                seed=42,  # For reproducibility
            )
            self._is_trained = True
            logger.info("Word2Vec training completed successfully")

        except Exception as e:
            logger.error(f"Error training Word2Vec model: {e}")
            raise

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get vector representation of a word.

        Parameters
        ----------
        word : str
            Word to get vector for

        Returns
        -------
        Optional[np.ndarray]
            Word vector or None if word not in vocabulary

        Examples
        --------
        >>> embeddings = Word2VecEmbeddings()
        >>> # After training...
        >>> vector = embeddings.get_word_vector("happy")
        >>> print(vector.shape)
        (100,)
        """
        if not self._is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return None

        try:
            if word in self.model.wv:
                return self.model.wv[word].copy()
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting word vector for '{word}': {e}")
            return None

    def get_sentence_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Get sentence embedding by averaging word vectors.

        Parameters
        ----------
        tokens : List[str]
            List of tokens in the sentence

        Returns
        -------
        np.ndarray
            Sentence vector (zero vector if no words found in vocabulary)

        Examples
        --------
        >>> embeddings = Word2VecEmbeddings()
        >>> # After training...
        >>> sentence_vec = embeddings.get_sentence_vector(['happy', 'day'])
        >>> print(sentence_vec.shape)
        (100,)
        """
        if not self._is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return np.zeros(self.vector_size)

        if not tokens:
            return np.zeros(self.vector_size)

        word_vectors = []
        for token in tokens:
            vector = self.get_word_vector(token)
            if vector is not None:
                word_vectors.append(vector)

        if not word_vectors:
            logger.debug(f"No words from tokens {tokens} found in vocabulary")
            return np.zeros(self.vector_size)

        # Average the word vectors
        sentence_vector = np.mean(word_vectors, axis=0)
        return sentence_vector

    def get_batch_sentence_vectors(self, batch_tokens: List[List[str]]) -> np.ndarray:
        """
        Get sentence vectors for a batch of tokenized sentences.

        Parameters
        ----------
        batch_tokens : List[List[str]]
            List of tokenized sentences

        Returns
        -------
        np.ndarray
            Array of sentence vectors with shape (n_sentences, vector_size)
        """
        vectors = []
        for tokens in batch_tokens:
            vector = self.get_sentence_vector(tokens)
            vectors.append(vector)

        return np.array(vectors)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str
            Path to save the model

        Raises
        ------
        ValueError
            If model is not trained yet
        """
        if not self._is_trained or self.model is None:
            raise ValueError("Cannot save untrained model")

        try:
            # Save the gensim model
            model_path = path if path.endswith(".model") else f"{path}.model"
            self.model.save(model_path)

            # Save metadata
            metadata = {
                "vector_size": self.vector_size,
                "window": self.window,
                "min_count": self.min_count,
                "workers": self.workers,
                "epochs": self.epochs,
                "is_trained": self._is_trained,
            }

            metadata_path = path.replace(".model", "_metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model

        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        """
        try:
            # Load the gensim model
            model_path = path if path.endswith(".model") else f"{path}.model"
            self.model = Word2Vec.load(model_path)

            # Load metadata
            metadata_path = path.replace(".model", "_metadata.pkl")
            try:
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)

                self.vector_size = metadata["vector_size"]
                self.window = metadata["window"]
                self.min_count = metadata["min_count"]
                self.workers = metadata["workers"]
                self.epochs = metadata["epochs"]
                self._is_trained = metadata["is_trained"]

            except FileNotFoundError:
                # If no metadata, infer from loaded model
                self.vector_size = self.model.wv.vector_size
                self._is_trained = True

            logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_vocabulary_size(self) -> int:
        """
        Get the size of the model's vocabulary.

        Returns
        -------
        int
            Number of words in vocabulary
        """
        if not self._is_trained or self.model is None:
            return 0

        return len(self.model.wv.key_to_index)

    def get_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        """
        Find words most similar to the given word.

        Parameters
        ----------
        word : str
            Word to find similar words for
        topn : int, default=10
            Number of similar words to return

        Returns
        -------
        List[tuple]
            List of (word, similarity_score) tuples
        """
        if not self._is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return []

        try:
            if word in self.model.wv:
                return self.model.wv.most_similar(word, topn=topn)
            else:
                logger.warning(f"Word '{word}' not in vocabulary")
                return []
        except Exception as e:
            logger.error(f"Error finding similar words for '{word}': {e}")
            return []

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained
