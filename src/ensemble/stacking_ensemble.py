"""
Stacking ensemble for combining models with a meta-learner.
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .ensemble_model import EnsembleModel
from ..models.base_model import BaseModel
from ..utils.exceptions import ModelError, PredictionError
from ..utils.logger import LoggerMixin


class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble that uses a meta-learner to combine base model predictions.
    
    Features:
    - K-fold cross-validation for generating meta-features
    - Multiple meta-learner options
    - Out-of-fold predictions for training meta-learner
    - Prevents data leakage
    """
    
    def __init__(
        self,
        name: str = "StackingEnsemble",
        models: List[BaseModel] = None,
        meta_learner: str = "logistic_regression",
        cv_folds: int = 5,
        meta_learner_config: Dict[str, Any] = None
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            name: Name of the ensemble
            models: List of base models
            meta_learner: Type of meta-learner
            cv_folds: Number of cross-validation folds
            meta_learner_config: Configuration for meta-learner
        """
        super().__init__(name, models)
        
        self.meta_learner_type = meta_learner
        self.cv_folds = cv_folds
        self.meta_learner_config = meta_learner_config or {}
        self.meta_learner = None
        self.oof_predictions = None  # Out-of-fold predictions
        self.is_meta_fitted = False
        
        self._initialize_meta_learner()
        
        self.logger.info(f"Initialized stacking ensemble with {meta_learner} meta-learner")
    
    def _initialize_meta_learner(self):
        """Initialize the meta-learner."""
        try:
            if self.meta_learner_type == "logistic_regression":
                default_config = {
                    'random_state': 42,
                    'max_iter': 1000,
                    'solver': 'liblinear',
                    'multi_class': 'auto'
                }
                default_config.update(self.meta_learner_config)
                self.meta_learner = LogisticRegression(**default_config)
            
            elif self.meta_learner_type == "random_forest":
                default_config = {
                    'random_state': 42,
                    'n_estimators': 100,
                    'max_depth': 10
                }
                default_config.update(self.meta_learner_config)
                self.meta_learner = RandomForestClassifier(**default_config)
            
            else:
                raise ModelError(f"Unsupported meta-learner: {self.meta_learner_type}")
            
            self.logger.info(f"Initialized meta-learner: {self.meta_learner_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize meta-learner: {e}")
            raise ModelError(f"Failed to initialize meta-learner: {e}")
    
    def fit(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Training labels
        
        Returns:
            Self for method chaining
        """
        if not self.models:
            raise ModelError("No base models in ensemble")
        
        try:
            self.classes_ = np.unique(y)
            n_samples = len(X)
            n_models = len(self.models)
            n_classes = len(self.classes_)
            
            # Step 1: Fit base models and generate out-of-fold predictions
            self.logger.info("Generating out-of-fold predictions for meta-learner...")
            oof_predictions = self._generate_oof_predictions(X, y)
            
            # Step 2: Fit base models on full training data
            self.logger.info("Fitting base models on full training data...")
            for model in self.models:
                model.fit(X, y)
            
            # Step 3: Fit meta-learner on out-of-fold predictions
            self.logger.info("Fitting meta-learner...")
            self.meta_learner.fit(oof_predictions, y)
            self.is_meta_fitted = True
            
            self.is_fitted = True
            self.logger.info(f"Stacking ensemble fitted with {n_models} base models")
            return self
            
        except Exception as e:
            self.logger.error(f"Stacking ensemble fitting failed: {e}")
            raise ModelError(f"Stacking ensemble fitting failed: {e}")
    
    def _generate_oof_predictions(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> np.ndarray:
        """
        Generate out-of-fold predictions using cross-validation.
        
        Args:
            X: Training features
            y: Training labels
        
        Returns:
            Out-of-fold predictions array
        """
        n_samples = len(X)
        n_models = len(self.models)
        n_classes = len(self.classes_)
        
        # Initialize arrays for out-of-fold predictions
        oof_probas = np.zeros((n_samples, n_models, n_classes))
        oof_preds = np.zeros((n_samples, n_models))
        
        # Create K-fold splitter
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # For each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            self.logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold = [X[i] for i in train_idx] if isinstance(X, list) else X[train_idx]
            X_val_fold = [X[i] for i in val_idx] if isinstance(X, list) else X[val_idx]
            y_train_fold = y[train_idx]
            
            # For each base model
            for model_idx, model in enumerate(self.models):
                try:
                    # Create a copy of the model for this fold
                    fold_model = self._create_model_copy(model)
                    
                    # Fit on training fold
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Predict on validation fold
                    fold_preds = fold_model.predict(X_val_fold)
                    fold_probas = fold_model.predict_proba(X_val_fold)
                    
                    # Store out-of-fold predictions
                    oof_preds[val_idx, model_idx] = fold_preds
                    oof_probas[val_idx, model_idx, :] = fold_probas
                    
                except Exception as e:
                    self.logger.error(f"Failed to train model {model.model_name} in fold {fold + 1}: {e}")
                    # Use default predictions for this model in this fold
                    default_pred = np.random.choice(self.classes_, size=len(val_idx))
                    default_proba = np.zeros((len(val_idx), n_classes))
                    for i, cls in enumerate(self.classes_):
                        default_proba[:, i] = (default_pred == cls).astype(float)
                    
                    oof_preds[val_idx, model_idx] = default_pred
                    oof_probas[val_idx, model_idx, :] = default_proba
        
        # Store out-of-fold predictions
        self.oof_predictions = oof_probas
        
        # Return probabilities for meta-learner (flatten model dimension)
        # Shape: (n_samples, n_models * n_classes)
        meta_features = oof_probas.reshape(n_samples, -1)
        
        return meta_features
    
    def _create_model_copy(self, original_model: BaseModel) -> BaseModel:
        """Create a copy of a model for cross-validation."""
        # Get model configuration
        model_info = original_model.get_model_info()
        model_type = model_info['model_name']
        config = model_info.get('config', {})
        
        # Create new instance based on type
        if model_type == "LogisticRegression":
            from ..models.traditional_models import LogisticRegressionModel
            return LogisticRegressionModel(config)
        elif model_type == "NaiveBayes":
            from ..models.traditional_models import NaiveBayesModel
            return NaiveBayesModel(config)
        elif model_type == "DecisionTree":
            from ..models.traditional_models import DecisionTreeModel
            return DecisionTreeModel(config)
        elif model_type == "BERT":
            from ..models.bert_model import BERTModel
            return BERTModel(config)
        else:
            raise ModelError(f"Unknown model type: {model_type}")
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make predictions using stacking ensemble.
        
        Args:
            X: Features for prediction
        
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted or not self.is_meta_fitted:
            raise ModelError("Stacking ensemble must be fitted before prediction")
        
        try:
            # Get meta-features from base models
            meta_features = self._get_meta_features(X)
            
            # Make predictions with meta-learner
            predictions = self.meta_learner.predict(meta_features)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Stacking prediction failed: {e}")
            raise PredictionError(f"Stacking prediction failed: {e}")
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get prediction probabilities from stacking ensemble.
        
        Args:
            X: Features for prediction
        
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted or not self.is_meta_fitted:
            raise ModelError("Stacking ensemble must be fitted before prediction")
        
        try:
            # Get meta-features from base models
            meta_features = self._get_meta_features(X)
            
            # Get probabilities from meta-learner
            probabilities = self.meta_learner.predict_proba(meta_features)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Stacking probability prediction failed: {e}")
            raise PredictionError(f"Stacking probability prediction failed: {e}")
    
    def _get_meta_features(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get meta-features from base models for prediction.
        
        Args:
            X: Features for prediction
        
        Returns:
            Meta-features array
        """
        n_samples = len(X)
        n_models = len(self.models)
        n_classes = len(self.classes_)
        
        # Get predictions from all base models
        meta_probas = np.zeros((n_samples, n_models, n_classes))
        
        for model_idx, model in enumerate(self.models):
            try:
                model_probas = model.predict_proba(X)
                meta_probas[:, model_idx, :] = model_probas
            except Exception as e:
                self.logger.error(f"Failed to get probabilities from {model.model_name}: {e}")
                # Use uniform probabilities as fallback
                uniform_proba = np.ones((n_samples, n_classes)) / n_classes
                meta_probas[:, model_idx, :] = uniform_proba
        
        # Flatten to create meta-features
        meta_features = meta_probas.reshape(n_samples, -1)
        
        return meta_features
    
    def get_stacking_info(self) -> Dict[str, Any]:
        """
        Get information about the stacking ensemble.
        
        Returns:
            Dictionary with stacking information
        """
        info = super().get_ensemble_info()
        
        info.update({
            'ensemble_type': 'stacking',
            'meta_learner_type': self.meta_learner_type,
            'cv_folds': self.cv_folds,
            'is_meta_fitted': self.is_meta_fitted,
            'meta_learner_params': self.meta_learner.get_params() if self.meta_learner else None
        })
        
        return info
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from meta-learner if available.
        
        Returns:
            Dictionary of feature importances or None
        """
        if not self.is_meta_fitted:
            return None
        
        try:
            if hasattr(self.meta_learner, 'feature_importances_'):
                importance = self.meta_learner.feature_importances_
                
                # Map to model-class combinations
                feature_names = []
                for model_idx, model in enumerate(self.models):
                    for class_idx, class_label in enumerate(self.classes_):
                        feature_names.append(f"{model.model_name}_class_{class_label}")
                
                return dict(zip(feature_names, importance))
            
            elif hasattr(self.meta_learner, 'coef_'):
                # For linear models, use coefficient magnitude
                coef = self.meta_learner.coef_
                if coef.ndim == 1:
                    importance = np.abs(coef)
                else:
                    importance = np.mean(np.abs(coef), axis=0)
                
                # Map to model-class combinations
                feature_names = []
                for model_idx, model in enumerate(self.models):
                    for class_idx, class_label in enumerate(self.classes_):
                        feature_names.append(f"{model.model_name}_class_{class_label}")
                
                return dict(zip(feature_names, importance))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return None
