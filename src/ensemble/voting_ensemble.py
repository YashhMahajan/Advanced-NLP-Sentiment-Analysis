"""
Voting ensemble for combining model predictions.
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional
from collections import Counter

from .ensemble_model import EnsembleModel
from ..models.base_model import BaseModel
from ..utils.exceptions import ModelError, PredictionError
from ..utils.logger import LoggerMixin


class VotingEnsemble(EnsembleModel):
    """
    Voting ensemble that combines predictions through voting.
    
    Supports:
    - Hard voting (majority vote on predictions)
    - Soft voting (weighted average of probabilities)
    - Custom weights for models
    """
    
    def __init__(
        self, 
        name: str = "VotingEnsemble",
        models: List[BaseModel] = None,
        voting: str = "soft",
        weights: List[float] = None
    ):
        """
        Initialize voting ensemble.
        
        Args:
            name: Name of the ensemble
            models: List of models to include
            voting: Voting method ('hard' or 'soft')
            weights: Weights for each model
        """
        super().__init__(name, models)
        
        if voting not in ['hard', 'soft']:
            raise ModelError("Voting method must be 'hard' or 'soft'")
        
        self.voting = voting
        self.weights = weights or ([1.0] * len(models) if models else [])
        
        self.logger.info(f"Initialized voting ensemble with {voting} voting")
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make predictions using voting.
        
        Args:
            X: Features for prediction
        
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before prediction")
        
        try:
            if self.voting == 'hard':
                return self._hard_voting_predict(X)
            else:
                return self._soft_voting_predict(X)
                
        except Exception as e:
            self.logger.error(f"Voting prediction failed: {e}")
            raise PredictionError(f"Voting prediction failed: {e}")
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features for prediction
        
        Returns:
            Weighted average probabilities
        """
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before prediction")
        
        if self.voting == 'hard':
            # For hard voting, return probability based on vote distribution
            return self._hard_voting_proba(X)
        else:
            return self._soft_voting_proba(X)
    
    def _hard_voting_predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Hard voting: majority vote on predictions."""
        # Get individual predictions
        individual_preds = self.get_individual_predictions(X)
        
        # Convert to array of shape (n_samples, n_models)
        pred_matrix = np.array([individual_preds[model_name] for model_name in individual_preds.keys()]).T
        
        # Apply weights if specified
        if self.weights and len(self.weights) == len(self.models):
            weighted_preds = []
            for i, sample_preds in enumerate(pred_matrix):
                weighted_counts = {}
                for j, pred in enumerate(sample_preds):
                    if pred not in weighted_counts:
                        weighted_counts[pred] = 0
                    weighted_counts[pred] += self.weights[j]
                
                # Choose class with highest weighted count
                best_pred = max(weighted_counts.keys(), key=lambda x: weighted_counts[x])
                weighted_preds.append(best_pred)
            
            return np.array(weighted_preds)
        else:
            # Simple majority vote
            predictions = []
            for sample_preds in pred_matrix:
                vote_counts = Counter(sample_preds)
                best_pred = vote_counts.most_common(1)[0][0]
                predictions.append(best_pred)
            
            return np.array(predictions)
    
    def _soft_voting_predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Soft voting: weighted average of probabilities."""
        probas = self._soft_voting_proba(X)
        return np.argmax(probas, axis=1)
    
    def _soft_voting_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Soft voting: weighted average of probabilities."""
        # Get individual probabilities
        individual_probas = self.get_individual_probabilities(X)
        
        # Stack probabilities: shape (n_models, n_samples, n_classes)
        proba_matrix = np.array([individual_probas[model_name] for model_name in individual_probas.keys()])
        
        # Apply weights and average
        if self.weights and len(self.weights) == len(self.models):
            weights_array = np.array(self.weights).reshape(-1, 1, 1)
            weighted_probas = proba_matrix * weights_array
            avg_probas = np.sum(weighted_probas, axis=0) / np.sum(weights_array)
        else:
            avg_probas = np.mean(proba_matrix, axis=0)
        
        return avg_probas
    
    def _hard_voting_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Convert hard voting to probabilities."""
        predictions = self._hard_voting_predict(X)
        n_samples = len(predictions)
        n_classes = len(self.classes_)
        
        # Create probability matrix
        probas = np.zeros((n_samples, n_classes))
        
        # For each sample, set probability based on vote distribution
        individual_preds = self.get_individual_predictions(X)
        pred_matrix = np.array([individual_preds[model_name] for model_name in individual_preds.keys()]).T
        
        for i, sample_preds in enumerate(pred_matrix):
            vote_counts = Counter(sample_preds)
            total_votes = len(sample_preds)
            
            for pred_class, count in vote_counts.items():
                class_idx = np.where(self.classes_ == pred_class)[0][0]
                probas[i, class_idx] = count / total_votes
        
        return probas
    
    def get_voting_stats(self, X: Union[np.ndarray, List[str]]) -> Dict[str, Any]:
        """
        Get voting statistics for analysis.
        
        Args:
            X: Features for prediction
        
        Returns:
            Dictionary with voting statistics
        """
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before analysis")
        
        try:
            # Get individual predictions
            individual_preds = self.get_individual_predictions(X)
            
            # Calculate agreement statistics
            pred_matrix = np.array([individual_preds[model_name] for model_name in individual_preds.keys()]).T
            
            agreement_scores = []
            disagreement_cases = []
            
            for i, sample_preds in enumerate(pred_matrix):
                unique_preds = np.unique(sample_preds)
                agreement = len(unique_preds) == 1  # All models agree
                
                if not agreement:
                    disagreement_cases.append({
                        'sample_index': i,
                        'predictions': dict(zip(individual_preds.keys(), sample_preds)),
                        'vote_distribution': dict(Counter(sample_preds))
                    })
                
                agreement_scores.append(1.0 if agreement else 0.0)
            
            stats = {
                'ensemble_name': self.name,
                'voting_method': self.voting,
                'num_samples': len(X),
                'agreement_rate': np.mean(agreement_scores),
                'disagreement_rate': 1 - np.mean(agreement_scores),
                'total_disagreements': len(disagreement_cases),
                'disagreement_cases': disagreement_cases[:10],  # First 10 cases
                'model_weights': self.weights
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get voting stats: {e}")
            raise ModelError(f"Failed to get voting stats: {e}")
    
    def get_model_contributions(self, X: Union[np.ndarray, List[str]]) -> Dict[str, float]:
        """
        Calculate contribution of each model to ensemble decisions.
        
        Args:
            X: Features for prediction
        
        Returns:
            Dictionary mapping model names to contribution scores
        """
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before analysis")
        
        try:
            # Get ensemble predictions
            ensemble_preds = self.predict(X)
            
            # Get individual predictions
            individual_preds = self.get_individual_predictions(X)
            
            # Calculate contribution scores
            contributions = {}
            
            for model_name, model_preds in individual_preds.items():
                # Calculate agreement with ensemble
                agreement = np.mean(model_preds == ensemble_preds)
                contributions[model_name] = agreement
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"Failed to calculate model contributions: {e}")
            raise ModelError(f"Failed to calculate model contributions: {e}")
