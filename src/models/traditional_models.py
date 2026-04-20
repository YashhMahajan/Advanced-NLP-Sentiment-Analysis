"""
Traditional machine learning models for sentiment analysis.
"""

import numpy as np
from typing import Dict, Any, List, Union
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseModel
from ..utils.exceptions import ModelError


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for sentiment analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Logistic Regression model.
        
        Args:
            config: Model configuration
        """
        default_config = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs'
        }
        if config:
            default_config.update(config)
        
        super().__init__("LogisticRegression", default_config)
        self.model = self.build_model()
    
    def build_model(self) -> LogisticRegression:
        """Build and return Logistic Regression model."""
        try:
            # Filter out invalid parameters for scikit-learn
            invalid_params = ['model_type', 'max_features', 'ngram_range']
            model_config = {k: v for k, v in self.config.items() 
                          if k not in invalid_params}
            model = LogisticRegression(**model_config)
            self.logger.debug(f"Built Logistic Regression with config: {model_config}")
            return model
        except Exception as e:
            raise ModelError(f"Failed to build Logistic Regression model: {e}", self.model_name)
    
    def fit(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> 'LogisticRegressionModel':
        """Train the model on provided data."""
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            self.classes_ = self.model.classes_
            self.logger.info(f"Logistic Regression trained on {len(X)} samples")
            return self
        except Exception as e:
            raise ModelError(f"Training failed: {e}", self.model_name)
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ModelError(f"Prediction failed: {e}", self.model_name)
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            raise ModelError(f"Probability prediction failed: {e}", self.model_name)


class NaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes model for sentiment analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Naive Bayes model.
        
        Args:
            config: Model configuration
        """
        default_config = {
            'alpha': 1.0,
            'fit_prior': True
        }
        if config:
            default_config.update(config)
        
        super().__init__("NaiveBayes", default_config)
        self.model = self.build_model()
    
    def build_model(self) -> MultinomialNB:
        """Build and return Naive Bayes model."""
        try:
            # Filter out invalid parameters for scikit-learn
            invalid_params = ['model_type', 'max_features', 'ngram_range', 'random_state', 'max_iter']
            model_config = {k: v for k, v in self.config.items() 
                          if k not in invalid_params}
            model = MultinomialNB(**model_config)
            self.logger.debug(f"Built Naive Bayes with config: {model_config}")
            return model
        except Exception as e:
            raise ModelError(f"Failed to build Naive Bayes model: {e}", self.model_name)
    
    def fit(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> 'NaiveBayesModel':
        """Train the model on provided data."""
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            self.classes_ = self.model.classes_
            self.logger.info(f"Naive Bayes trained on {len(X)} samples")
            return self
        except Exception as e:
            raise ModelError(f"Training failed: {e}", self.model_name)
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ModelError(f"Prediction failed: {e}", self.model_name)
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            raise ModelError(f"Probability prediction failed: {e}", self.model_name)


class DecisionTreeModel(BaseModel):
    """Decision Tree model for sentiment analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Decision Tree model.
        
        Args:
            config: Model configuration
        """
        default_config = {
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'criterion': 'gini'
        }
        if config:
            default_config.update(config)
        
        super().__init__("DecisionTree", default_config)
        self.model = self.build_model()
    
    def build_model(self) -> DecisionTreeClassifier:
        """Build and return Decision Tree model."""
        try:
            # Filter out invalid parameters for scikit-learn
            invalid_params = ['model_type', 'max_features', 'ngram_range', 'max_iter']
            model_config = {k: v for k, v in self.config.items() 
                          if k not in invalid_params}
            model = DecisionTreeClassifier(**model_config)
            self.logger.debug(f"Built Decision Tree with config: {model_config}")
            return model
        except Exception as e:
            raise ModelError(f"Failed to build Decision Tree model: {e}", self.model_name)
    
    def fit(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> 'DecisionTreeModel':
        """Train the model on provided data."""
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            self.classes_ = self.model.classes_
            self.logger.info(f"Decision Tree trained on {len(X)} samples")
            return self
        except Exception as e:
            raise ModelError(f"Training failed: {e}", self.model_name)
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ModelError(f"Prediction failed: {e}", self.model_name)
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            raise ModelError(f"Probability prediction failed: {e}", self.model_name)
