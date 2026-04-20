"""
Model manager for handling multiple models and model selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from pathlib import Path
import json

from .base_model import BaseModel
from .traditional_models import LogisticRegressionModel, NaiveBayesModel, DecisionTreeModel
from .bert_model import BERTModel
from ..preprocessing import TextPreprocessor, FeatureExtractor
from ..utils.exceptions import ModelError, ValidationError
from ..utils.logger import LoggerMixin
from ..utils.validators import validate_model_config


class ModelManager(LoggerMixin):
    """
    Manager class for handling multiple sentiment analysis models.
    
    Features:
    - Model registration and management
    - Batch training and evaluation
    - Model comparison and selection
    - Model persistence
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model manager.
        
        Args:
            config: Global configuration for model manager
        """
        self.config = config or {}
        self.models: Dict[str, BaseModel] = {}
        self.preprocessor = None
        self.feature_extractors: Dict[str, FeatureExtractor] = {}
        self.results: Dict[str, Any] = {}
        
        # Initialize default components
        self._initialize_default_components()
        
        self.logger.info("ModelManager initialized")
    
    def _initialize_default_components(self):
        """Initialize default preprocessor and feature extractors."""
        try:
            # Initialize preprocessor
            self.preprocessor = TextPreprocessor(
                language=self.config.get('language', 'english'),
                remove_stopwords=self.config.get('remove_stopwords', True),
                lemmatize=self.config.get('lemmatize', True)
            )
            
            # Initialize feature extractors
            self.feature_extractors['bow'] = FeatureExtractor(
                method='bow',
                max_features=self.config.get('max_features', 5000),
                ngram_range=self.config.get('ngram_range', (1, 2))
            )
            
            self.feature_extractors['tfidf'] = FeatureExtractor(
                method='tfidf',
                max_features=self.config.get('max_features', 5000),
                ngram_range=self.config.get('ngram_range', (1, 2))
            )
            
            self.logger.info("Default components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default components: {e}")
            raise ModelError(f"Component initialization failed: {e}")
    
    def register_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """
        Register a new model with the manager.
        
        Args:
            model_name: Name for the model
            model_config: Model configuration
        """
        try:
            # Validate configuration
            validated_config = validate_model_config(model_config)
            
            # Create model based on type
            model_type = validated_config['model_type']
            
            if model_type == 'logistic_regression':
                model = LogisticRegressionModel(validated_config)
            elif model_type == 'naive_bayes':
                model = NaiveBayesModel(validated_config)
            elif model_type == 'decision_tree':
                model = DecisionTreeModel(validated_config)
            elif model_type == 'bert':
                model = BERTModel(validated_config)
            else:
                raise ValidationError(f"Unknown model type: {model_type}")
            
            self.models[model_name] = model
            self.logger.info(f"Registered model: {model_name} ({model_type})")
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_name}: {e}")
            raise ModelError(f"Model registration failed: {e}")
    
    def register_default_models(self) -> None:
        """Register default models with standard configurations."""
        default_models = {
            'logistic_regression_bow': {
                'model_type': 'logistic_regression',
                'random_state': 42,
                'max_iter': 1000
            },
            'logistic_regression_tfidf': {
                'model_type': 'logistic_regression',
                'random_state': 42,
                'max_iter': 1000
            },
            'naive_bayes_bow': {
                'model_type': 'naive_bayes',
                'alpha': 1.0
            },
            'naive_bayes_tfidf': {
                'model_type': 'naive_bayes',
                'alpha': 1.0
            },
            'decision_tree_bow': {
                'model_type': 'decision_tree',
                'random_state': 42,
                'max_depth': 10
            },
            'decision_tree_tfidf': {
                'model_type': 'decision_tree',
                'random_state': 42,
                'max_depth': 10
            },
            'bert': {
                'model_type': 'bert',
                'model_name': 'distilbert-base-uncased',  # Lighter model, similar performance
                'num_labels': 3,
                'max_length': 128,
                'num_epochs': 2,  # Reduced for faster training
                'batch_size': 16,  # Increased batch size for better GPU utilization
                'learning_rate': 3e-5,  # Slightly higher for faster convergence
                'warmup_steps': 100,  # Reduced for shorter training
                'weight_decay': 0.01,
                'early_stopping_patience': 1,  # Early stopping to prevent overfitting
                'evaluation_strategy': 'epoch',
                'save_strategy': 'epoch',
                'load_best_model_at_end': True
            }
        }
        
        for model_name, config in default_models.items():
            try:
                self.register_model(model_name, config)
            except Exception as e:
                self.logger.warning(f"Failed to register {model_name}: {e}")
    
    def train_all_models(
        self, 
        texts: List[str], 
        labels: np.ndarray,
        feature_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Train all registered models.
        
        Args:
            texts: Training texts
            labels: Training labels
            feature_types: List of feature types to use
        
        Returns:
            Dictionary with training results
        """
        if feature_types is None:
            feature_types = ['bow', 'tfidf']
        
        try:
            # Preprocess texts
            processed_texts = self.preprocessor.preprocess_batch(texts)
            
            training_results = {}
            
            for model_name, model in self.models.items():
                self.logger.info(f"Training model: {model_name}")
                
                try:
                    # Determine feature type
                    if 'bert' in model_name.lower() or model.model_name == 'BERT':
                        # BERT handles its own tokenization
                        X_train = texts
                    else:
                        # Traditional models need feature extraction
                        feature_type = 'bow' if 'bow' in model_name else 'tfidf'
                        if feature_type not in feature_types:
                            continue
                        
                        extractor = self.feature_extractors[feature_type]
                        X_train = extractor.fit_transform(processed_texts)
                    
                    # Train model
                    model.fit(X_train, labels)
                    
                    # Store feature extractor for later use
                    if hasattr(model, 'model_name') and model.model_name != 'BERT':
                        feature_type = 'bow' if 'bow' in model_name else 'tfidf'
                        model.feature_extractor = self.feature_extractors[feature_type]
                    
                    training_results[model_name] = {
                        'status': 'success',
                        'message': f'Model trained successfully'
                    }
                    
                    self.logger.info(f"Successfully trained: {model_name}")
                    
                except Exception as e:
                    training_results[model_name] = {
                        'status': 'failed',
                        'message': str(e)
                    }
                    self.logger.error(f"Failed to train {model_name}: {e}")
            
            self.results['training'] = training_results
            return training_results
            
        except Exception as e:
            self.logger.error(f"Batch training failed: {e}")
            raise ModelError(f"Batch training failed: {e}")
    
    def evaluate_all_models(
        self, 
        texts: List[str], 
        labels: np.ndarray,
        target_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Args:
            texts: Test texts
            labels: Test labels
            target_names: Names for target classes
        
        Returns:
            Dictionary with evaluation results
        """
        if target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']
        
        try:
            # Preprocess texts
            processed_texts = self.preprocessor.preprocess_batch(texts)
            
            evaluation_results = {}
            
            for model_name, model in self.models.items():
                if not model.is_fitted:
                    self.logger.warning(f"Skipping {model_name}: not fitted")
                    continue
                
                self.logger.info(f"Evaluating model: {model_name}")
                
                try:
                    # Prepare features
                    if model.model_name == 'BERT':
                        X_test = texts
                    else:
                        # Use stored feature extractor
                        if hasattr(model, 'feature_extractor'):
                            X_test = model.feature_extractor.transform(processed_texts)
                        else:
                            # Fallback: determine feature type and extract
                            feature_type = 'bow' if 'bow' in model_name else 'tfidf'
                            extractor = self.feature_extractors[feature_type]
                            X_test = extractor.transform(processed_texts)
                    
                    # Evaluate
                    results = model.evaluate(X_test, labels, target_names)
                    evaluation_results[model_name] = results
                    
                    self.logger.info(f"Evaluated {model_name}: F1={results['f1_score']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Evaluation failed for {model_name}: {e}")
                    evaluation_results[model_name] = {
                        'status': 'failed',
                        'message': str(e)
                    }
            
            self.results['evaluation'] = evaluation_results
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            raise ModelError(f"Batch evaluation failed: {e}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, BaseModel]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        try:
            if 'evaluation' not in self.results:
                raise ModelError("No evaluation results available. Run evaluate_all_models() first.")
            
            best_model_name = None
            best_score = -float('inf')
            best_model = None
            
            for model_name, results in self.results['evaluation'].items():
                if 'status' in results and results['status'] == 'failed':
                    continue
                
                if metric in results:
                    score = results[metric]
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model = self.models[model_name]
            
            if best_model is None:
                raise ModelError(f"No valid models found for metric: {metric}")
            
            self.logger.info(f"Best model: {best_model_name} ({metric}={best_score:.4f})")
            return best_model_name, best_model
            
        except Exception as e:
            self.logger.error(f"Failed to get best model: {e}")
            raise ModelError(f"Failed to get best model: {e}")
    
    def predict_with_model(self, model_name: str, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Make predictions using a specific model.
        
        Args:
            model_name: Name of the model to use
            texts: Text(s) for prediction
        
        Returns:
            Predictions array
        """
        try:
            if model_name not in self.models:
                raise ModelError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            if not model.is_fitted:
                raise ModelError(f"Model {model_name} is not fitted")
            
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
            
            # Prepare features
            if model.model_name == 'BERT':
                X = texts
            else:
                processed_texts = self.preprocessor.preprocess_batch(texts)
                if hasattr(model, 'feature_extractor'):
                    X = model.feature_extractor.transform(processed_texts)
                else:
                    feature_type = 'bow' if 'bow' in model_name else 'tfidf'
                    extractor = self.feature_extractors[feature_type]
                    X = extractor.transform(processed_texts)
            
            # Make predictions
            predictions = model.predict(X)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_name}: {e}")
            raise ModelError(f"Prediction failed: {e}")
    
    def get_model_comparison_table(self) -> pd.DataFrame:
        """
        Get a comparison table of all models.
        
        Returns:
            DataFrame with model comparison
        """
        try:
            if 'evaluation' not in self.results:
                raise ModelError("No evaluation results available")
            
            comparison_data = []
            
            for model_name, results in self.results['evaluation'].items():
                if 'status' in results and results['status'] == 'failed':
                    continue
                
                row = {
                    'Model': model_name,
                    'Accuracy': results.get('accuracy', 0),
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1-Score': results.get('f1_score', 0)
                }
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison table: {e}")
            raise ModelError(f"Failed to create comparison table: {e}")
    
    def save_all_models(self, directory: str) -> None:
        """
        Save all trained models to disk.
        
        Args:
            directory: Directory to save models
        """
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            for model_name, model in self.models.items():
                if model.is_fitted:
                    model_path = Path(directory) / f"{model_name}.pkl"
                    model.save_model(str(model_path))
                    
                    self.logger.info(f"Saved model: {model_name}")
            
            # Save results
            results_path = Path(directory) / "results.json"
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"All models saved to {directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            raise ModelError(f"Failed to save models: {e}")
    
    def load_all_models(self, directory: str) -> None:
        """
        Load all models from disk.
        
        Args:
            directory: Directory to load models from
        """
        try:
            directory_path = Path(directory)
            
            # Load results
            results_path = directory_path / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    self.results = json.load(f)
            
            # Load models
            for model_file in directory_path.glob("*.pkl"):
                model_name = model_file.stem
                
                # Determine model type from name or config
                if 'bert' in model_name.lower():
                    model = BERTModel()
                elif 'logistic' in model_name.lower():
                    model = LogisticRegressionModel()
                elif 'naive' in model_name.lower():
                    model = NaiveBayesModel()
                elif 'decision' in model_name.lower() or 'tree' in model_name.lower():
                    model = DecisionTreeModel()
                else:
                    continue  # Skip unknown model types
                
                model.load_model(str(model_file))
                self.models[model_name] = model
                
                self.logger.info(f"Loaded model: {model_name}")
            
            self.logger.info(f"All models loaded from {directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise ModelError(f"Failed to load models: {e}")
    
    def get_manager_info(self) -> Dict[str, Any]:
        """
        Get information about the model manager.
        
        Returns:
            Dictionary with manager information
        """
        info = {
            'total_models': len(self.models),
            'fitted_models': sum(1 for model in self.models.values() if model.is_fitted),
            'available_feature_extractors': list(self.feature_extractors.keys()),
            'registered_models': list(self.models.keys()),
            'has_results': bool(self.results),
            'config': self.config
        }
        
        return info
