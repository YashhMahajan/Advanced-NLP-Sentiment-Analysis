"""
Base model interface for all sentiment analysis models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from ..utils.exceptions import ModelError, PredictionError
from ..utils.logger import LoggerMixin


class BaseModel(LoggerMixin, ABC):
    """
    Abstract base class for all sentiment analysis models.
    
    Provides common interface and functionality for:
    - Training and prediction
    - Evaluation
    - Model persistence
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            config: Model configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.classes_ = None
        
        self.logger.info(f"Initialized {model_name} model")
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the underlying model.
        
        Returns:
            Configured model instance
        """
        pass
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> 'BaseModel':
        """
        Train the model on provided data.
        
        Args:
            X: Training features
            y: Training labels
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
        
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features for prediction
        
        Returns:
            Prediction probabilities
        """
        pass
    
    def evaluate(
        self, 
        X_test: Union[np.ndarray, List[str]], 
        y_test: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_names: Names for target classes
        
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_fitted:
                raise ModelError("Model must be fitted before evaluation", self.model_name)
            
            # Make predictions
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Classification report
            if target_names is None:
                target_names = [f"Class_{i}" for i in sorted(np.unique(y_test))]
            
            class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results = {
                'model_name': self.model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'target_names': target_names,
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist()
            }
            
            self.logger.info(f"Evaluation completed for {self.model_name}: F1={f1:.4f}")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for {self.model_name}: {e}")
            raise ModelError(f"Evaluation failed: {e}", self.model_name)
    
    def save_model(self, filepath: str) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        
        Returns:
            Filepath where model was saved
        """
        try:
            import pickle
            import os
            
            if not self.is_fitted:
                raise ModelError("Model must be fitted before saving", self.model_name)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'is_fitted': self.is_fitted,
                'classes_': self.classes_
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Failed to save model: {e}", self.model_name)
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        
        Returns:
            Self for method chaining
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']
            self.classes_ = model_data.get('classes_')
            
            self.logger.info(f"Model loaded from {filepath}")
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}", self.model_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'config': self.config,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None
        }
        
        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        return info
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        
        Returns:
            Self for method chaining
        """
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        # Update config
        self.config.update(params)
        
        self.logger.info(f"Updated parameters: {params}")
        return self
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Returns:
            Dictionary of feature importances or None
        """
        if (self.is_fitted and 
            hasattr(self.model, 'feature_importances_') and 
            self.model.feature_importances_ is not None):
            
            importance = self.model.feature_importances_
            return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
        
        return None
    
    def cross_validate(
        self, 
        X: Union[np.ndarray, List[str]], 
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
        
        Returns:
            Cross-validation results
        """
        try:
            from sklearn.model_selection import cross_val_score
            
            if not hasattr(self.model, 'fit'):
                raise ModelError("Model does not support fitting", self.model_name)
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
            
            cv_results = {
                'model_name': self.model_name,
                'cv_scores': cv_scores.tolist(),
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scoring_metric': scoring,
                'cv_folds': cv
            }
            
            self.logger.info(f"Cross-validation completed: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            raise ModelError(f"Cross-validation failed: {e}", self.model_name)
