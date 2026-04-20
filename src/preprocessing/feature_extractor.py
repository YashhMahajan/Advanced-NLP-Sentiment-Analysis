"""
Feature extraction utilities for text representation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.exceptions import DataProcessingError
from ..utils.logger import LoggerMixin


class FeatureExtractor(LoggerMixin, BaseEstimator, TransformerMixin):
    """
    Feature extraction for text data using various vectorization methods.
    
    Supports:
    - Bag of Words (BoW)
    - TF-IDF
    - N-grams
    - Custom preprocessing pipelines
    """
    
    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        lowercase: bool = True,
        stop_words: Optional[Union[str, list]] = None,
        preprocessor: Optional[callable] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            method: Vectorization method ('bow' or 'tfidf')
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            lowercase: Whether to convert to lowercase
            stop_words: Stop words to remove
            preprocessor: Custom preprocessing function
        """
        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.preprocessor = preprocessor
        
        # Validate method
        if self.method not in ['bow', 'tfidf']:
            raise DataProcessingError(f"Method must be 'bow' or 'tfidf', got '{method}'")
        
        # Initialize vectorizer
        self.vectorizer = None
        self._initialize_vectorizer()
        
        self.logger.info(f"FeatureExtractor initialized with method={method}, max_features={max_features}")
    
    def _initialize_vectorizer(self):
        """Initialize the appropriate vectorizer."""
        vectorizer_params = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'lowercase': self.lowercase,
            'stop_words': self.stop_words,
            'preprocessor': self.preprocessor
        }
        
        if self.method == 'bow':
            self.vectorizer = CountVectorizer(**vectorizer_params)
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
        
        self.logger.debug(f"Initialized {self.method} vectorizer")
    
    def fit(self, texts: list) -> 'FeatureExtractor':
        """
        Fit the vectorizer on the provided texts.
        
        Args:
            texts: List of text documents
        
        Returns:
            Self for method chaining
        """
        try:
            if not texts or not any(texts):
                raise DataProcessingError("Cannot fit on empty or None texts")
            
            self.vectorizer.fit(texts)
            self.logger.info(f"Fitted {self.method} vectorizer on {len(texts)} documents")
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to fit vectorizer: {e}")
            raise DataProcessingError(f"Failed to fit vectorizer: {e}", "fit")
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform texts to feature matrix.
        
        Args:
            texts: List of text documents
        
        Returns:
            Feature matrix
        """
        try:
            if not texts or not any(texts):
                raise DataProcessingError("Cannot transform empty or None texts")
            
            if self.vectorizer is None:
                raise DataProcessingError("Vectorizer not fitted. Call fit() first.")
            
            features = self.vectorizer.transform(texts)
            
            # Convert to dense array if small, otherwise keep sparse
            if features.shape[0] < 1000 and features.shape[1] < 1000:
                features = features.toarray()
            
            self.logger.debug(f"Transformed {len(texts)} documents to feature matrix of shape {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to transform texts: {e}")
            raise DataProcessingError(f"Failed to transform texts: {e}", "transform")
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts: List of text documents
        
        Returns:
            Feature matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> list:
        """
        Get feature names (vocabulary).
        
        Returns:
            List of feature names
        """
        if self.vectorizer is None:
            raise DataProcessingError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_vocabulary(self) -> dict:
        """
        Get vocabulary with indices.
        
        Returns:
            Dictionary mapping features to indices
        """
        if self.vectorizer is None:
            raise DataProcessingError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.vocabulary_
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """
        Get statistics about extracted features.
        
        Returns:
            Dictionary with feature statistics
        """
        if self.vectorizer is None:
            raise DataProcessingError("Vectorizer not fitted. Call fit() first.")
        
        vocab = self.get_vocabulary()
        feature_names = self.get_feature_names()
        
        stats = {
            'method': self.method,
            'vocabulary_size': len(vocab),
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'feature_names_sample': feature_names[:10],  # First 10 features
            'total_features': len(feature_names)
        }
        
        return stats
    
    def save_vectorizer(self, filepath: str):
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the vectorizer
        """
        try:
            import pickle
            
            if self.vectorizer is None:
                raise DataProcessingError("No vectorizer to save")
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            self.logger.info(f"Vectorizer saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save vectorizer: {e}")
            raise DataProcessingError(f"Failed to save vectorizer: {e}", "save_vectorizer")
    
    def load_vectorizer(self, filepath: str):
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to load the vectorizer from
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.logger.info(f"Vectorizer loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load vectorizer: {e}")
            raise DataProcessingError(f"Failed to load vectorizer: {e}", "load_vectorizer")
    
    def get_top_features(self, n: int = 10) -> Dict[str, list]:
        """
        Get top features based on frequency or TF-IDF scores.
        
        Args:
            n: Number of top features to return
        
        Returns:
            Dictionary with top features and their scores
        """
        if self.vectorizer is None:
            raise DataProcessingError("Vectorizer not fitted. Call fit() first.")
        
        try:
            if hasattr(self.vectorizer, 'idf_'):  # TF-IDF
                # Get features with highest IDF scores (rare terms)
                idf_scores = self.vectorizer.idf_
                feature_names = self.get_feature_names()
                
                # Sort by IDF score (higher = rarer)
                top_indices = np.argsort(idf_scores)[-n:][::-1]
                top_features = [(feature_names[i], idf_scores[i]) for i in top_indices]
                
            else:  # BoW
                # Get features with highest document frequency
                feature_names = self.get_feature_names()
                doc_freq = self.vectorizer.transform([' '.join(feature_names)]).toarray()[0]
                
                top_indices = np.argsort(doc_freq)[-n:][::-1]
                top_features = [(feature_names[i], doc_freq[i]) for i in top_indices]
            
            return {
                'method': self.method,
                'top_features': top_features,
                'n_features': n
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get top features: {e}")
            raise DataProcessingError(f"Failed to get top features: {e}", "get_top_features")
