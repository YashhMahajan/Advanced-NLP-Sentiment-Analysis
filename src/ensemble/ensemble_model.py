"""
Base ensemble model for combining multiple sentiment analysis models.
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional
from abc import ABC, abstractmethod

from ..models.base_model import BaseModel
from ..utils.exceptions import ModelError, PredictionError
from ..utils.logger import LoggerMixin


class EnsembleModel(LoggerMixin, ABC):
    """
    Abstract base class for ensemble models.
    
    Provides common functionality for combining multiple models:
    - Model management
    - Prediction aggregation
    - Evaluation
    """
    
    def __init__(self, name: str, models: List[BaseModel] = None):
        """
        Initialize ensemble model.
        
        Args:
            name: Name of the ensemble
            models: List of models to include in ensemble
        """
        self.name = name
        self.models = models or []
        self.is_fitted = False
        self.classes_ = None
        self.weights = None
        
        self.logger.info(f"Initialized ensemble '{name}' with {len(self.models)} models")
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Model to add
            weight: Weight for this model in ensemble
        """
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
        
        self.logger.info(f"Added model {model.model_name} to ensemble")
    
    def remove_model(self, model_name: str) -> None:
        """
        Remove a model from the ensemble.
        
        Args:
            model_name: Name of model to remove
        """
        for i, model in enumerate(self.models):
            if model.model_name == model_name:
                self.models.pop(i)
                if self.weights and i < len(self.weights):
                    self.weights.pop(i)
                self.logger.info(f"Removed model {model_name} from ensemble")
                return
        
        raise ModelError(f"Model {model_name} not found in ensemble")
    
    def fit(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> 'EnsembleModel':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Training labels
        
        Returns:
            Self for method chaining
        """
        if not self.models:
            raise ModelError("No models in ensemble to fit")
        
        try:
            self.classes_ = np.unique(y)
            
            # Fit each model
            for model in self.models:
                self.logger.info(f"Fitting model: {model.model_name}")
                model.fit(X, y)
            
            self.is_fitted = True
            self.logger.info(f"Ensemble '{self.name}' fitted with {len(self.models)} models")
            return self
            
        except Exception as e:
            self.logger.error(f"Ensemble fitting failed: {e}")
            raise ModelError(f"Ensemble fitting failed: {e}")
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features for prediction
        
        Returns:
            Ensemble predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get ensemble prediction probabilities.
        
        Args:
            X: Features for prediction
        
        Returns:
            Ensemble probabilities
        """
        pass
    
    def evaluate(
        self, 
        X_test: Union[np.ndarray, List[str]], 
        y_test: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the ensemble.
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_names: Names for target classes
        
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_fitted:
                raise ModelError("Ensemble must be fitted before evaluation")
            
            # Make predictions
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                classification_report, confusion_matrix
            )
            
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
                'ensemble_name': self.name,
                'num_models': len(self.models),
                'model_names': [model.model_name for model in self.models],
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
            
            self.logger.info(f"Ensemble evaluation completed: F1={f1:.4f}")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Ensemble evaluation failed: {e}")
            raise ModelError(f"Ensemble evaluation failed: {e}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble.
        
        Returns:
            Dictionary with ensemble information
        """
        info = {
            'ensemble_name': self.name,
            'num_models': len(self.models),
            'is_fitted': self.is_fitted,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None,
            'weights': self.weights,
            'models': [model.get_model_info() for model in self.models]
        }
        
        return info
    
    def set_weights(self, weights: List[float]) -> None:
        """
        Set weights for models in ensemble.
        
        Args:
            weights: List of weights for each model
        """
        if len(weights) != len(self.models):
            raise ModelError(f"Number of weights ({len(weights)}) must match number of models ({len(self.models)})")
        
        if not all(w >= 0 for w in weights):
            raise ModelError("All weights must be non-negative")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in weights]
        else:
            self.weights = [1.0 / len(weights)] * len(weights)
        
        self.logger.info(f"Updated ensemble weights: {self.weights}")
    
    def get_individual_predictions(self, X: Union[np.ndarray, List[str]]) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.
        
        Args:
            X: Features for prediction
        
        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before prediction")
        
        predictions = {}
        
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions[model.model_name] = pred
            except Exception as e:
                self.logger.error(f"Prediction failed for {model.model_name}: {e}")
                raise PredictionError(f"Individual prediction failed: {e}")
        
        return predictions
    
    def get_individual_probabilities(self, X: Union[np.ndarray, List[str]]) -> Dict[str, np.ndarray]:
        """
        Get prediction probabilities from each individual model.
        
        Args:
            X: Features for prediction
        
        Returns:
            Dictionary mapping model names to probabilities
        """
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before prediction")
        
        probabilities = {}
        
        for model in self.models:
            try:
                proba = model.predict_proba(X)
                probabilities[model.model_name] = proba
            except Exception as e:
                self.logger.error(f"Probability prediction failed for {model.model_name}: {e}")
                raise PredictionError(f"Individual probability prediction failed: {e}")
        
        return probabilities
