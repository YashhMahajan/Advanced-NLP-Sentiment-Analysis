"""
Text preprocessing utilities for sentiment analysis.
"""

import re
import string
from typing import List, Optional, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from ..utils.exceptions import DataProcessingError
from ..utils.logger import LoggerMixin


class TextPreprocessor(LoggerMixin):
    """
    Comprehensive text preprocessing for sentiment analysis.
    
    Features:
    - Text cleaning (URLs, special characters, etc.)
    - Tokenization
    - Stopword removal
    - Lemmatization
    - Custom filtering
    """
    
    def __init__(
        self,
        language: str = "english",
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_length: int = 2,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            language: Language for stopwords
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Add custom stopwords
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        self.logger.info(f"TextPreprocessor initialized with language={language}")
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
            for data in nltk_data:
                try:
                    nltk.data.find(f'tokenizers/{data}')
                except LookupError:
                    nltk.download(data, quiet=True)
        except Exception as e:
            self.logger.error(f"Failed to download NLTK data: {e}")
            raise DataProcessingError(f"NLTK data download failed: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, special characters, etc.
        
        Args:
            text: Input text to clean
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            raise DataProcessingError("Input must be a string", "clean_text")
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
            
            # Remove user mentions and hashtags (keep the text)
            text = re.sub(r'@\w+|#\w+', ' ', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', ' ', text)
            
            # Remove special characters and punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove numbers (optional - keep if they might be meaningful)
            text = re.sub(r'\d+', ' ', text)
            
            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Text cleaning failed: {e}")
            raise DataProcessingError(f"Text cleaning failed: {e}", "clean_text")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            raise DataProcessingError(f"Tokenization failed: {e}", "tokenize")
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Filtered token list
        """
        if not self.remove_stopwords:
            return tokens
        
        try:
            filtered_tokens = [
                token for token in tokens 
                if token not in self.stop_words
            ]
            return filtered_tokens
        except Exception as e:
            self.logger.error(f"Stopword removal failed: {e}")
            raise DataProcessingError(f"Stopword removal failed: {e}", "remove_stopwords")
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens to lemmatize
        
        Returns:
            Lemmatized token list
        """
        if not self.lemmatize:
            return tokens
        
        try:
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens
            ]
            return lemmatized_tokens
        except Exception as e:
            self.logger.error(f"Lemmatization failed: {e}")
            raise DataProcessingError(f"Lemmatization failed: {e}", "lemmatize")
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on length and other criteria.
        
        Args:
            tokens: List of tokens to filter
        
        Returns:
            Filtered token list
        """
        try:
            filtered_tokens = [
                token for token in tokens 
                if len(token) >= self.min_word_length and token.strip()
            ]
            return filtered_tokens
        except Exception as e:
            self.logger.error(f"Token filtering failed: {e}")
            raise DataProcessingError(f"Token filtering failed: {e}", "filter_tokens")
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
        
        Returns:
            Fully preprocessed text
        """
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Tokenize
            tokens = self.tokenize(cleaned_text)
            
            # Remove stopwords
            tokens = self.remove_stopwords_from_tokens(tokens)
            
            # Lemmatize
            tokens = self.lemmatize_tokens(tokens)
            
            # Filter tokens
            tokens = self.filter_tokens(tokens)
            
            # Join back to text
            processed_text = " ".join(tokens)
            
            return processed_text
            
        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {e}")
            raise DataProcessingError(f"Preprocessing pipeline failed: {e}", "preprocess")
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
        
        Returns:
            List of preprocessed texts
        """
        processed_texts = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            try:
                processed = self.preprocess(text)
                processed_texts.append(processed)
            except Exception as e:
                self.logger.warning(f"Failed to process text at index {i}: {e}")
                failed_indices.append(i)
                processed_texts.append("")  # Add empty string as placeholder
        
        if failed_indices:
            self.logger.warning(f"Failed to process {len(failed_indices)} texts at indices: {failed_indices}")
        
        return processed_texts
    
    def get_preprocessing_stats(self, original_text: str, processed_text: str) -> dict:
        """
        Get statistics about preprocessing changes.
        
        Args:
            original_text: Original input text
            processed_text: Processed output text
        
        Returns:
            Dictionary with preprocessing statistics
        """
        original_tokens = self.tokenize(original_text)
        processed_tokens = self.tokenize(processed_text)
        
        stats = {
            "original_length": len(original_text),
            "processed_length": len(processed_text),
            "original_token_count": len(original_tokens),
            "processed_token_count": len(processed_tokens),
            "reduction_ratio": 1 - (len(processed_text) / len(original_text)) if original_text else 0,
            "token_reduction_ratio": 1 - (len(processed_tokens) / len(original_tokens)) if original_tokens else 0
        }
        
        return stats
