"""
Cross-validation framework for robust model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import time

from ..models.base_model import BaseModel
from ..preprocessing import TextPreprocessor, FeatureExtractor
from ..utils.exceptions import ValidationError, ModelError
from ..utils.logger import LoggerMixin


class CrossValidator(LoggerMixin):
    """
    Comprehensive cross-validation framework for sentiment analysis models.
    
    Features:
    - Multiple cross-validation strategies (KFold, StratifiedKFold, TimeSeriesSplit)
    - Detailed metrics and reporting
    - Support for preprocessing pipelines
    - Parallel processing support
    - Result analysis and visualization
    """
    
    def __init__(
        self,
        cv_strategy: str = "stratified_kfold",
        cv_folds: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
        n_jobs: int = 1,
        scoring_metrics: List[str] = None
    ):
        """
        Initialize cross-validator.
        
        Args:
            cv_strategy: Cross-validation strategy
            cv_folds: Number of CV folds
            random_state: Random state for reproducibility
            shuffle: Whether to shuffle data before splitting
            n_jobs: Number of parallel jobs
            scoring_metrics: List of metrics to compute
        """
        self.cv_strategy = cv_strategy
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.scoring_metrics = scoring_metrics or ['accuracy', 'precision', 'recall', 'f1']
        
        # Initialize cross-validation splitter
        self.cv_splitter = self._initialize_cv_splitter()
        
        self.logger.info(f"Initialized CrossValidator with {cv_strategy}, {cv_folds} folds")
    
    def _initialize_cv_splitter(self):
        """Initialize the cross-validation splitter."""
        try:
            if self.cv_strategy == "kfold":
                return KFold(
                    n_splits=self.cv_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
            elif self.cv_strategy == "stratified_kfold":
                return StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
            elif self.cv_strategy == "time_series":
                return TimeSeriesSplit(
                    n_splits=self.cv_folds
                )
            else:
                raise ValidationError(f"Unknown CV strategy: {self.cv_strategy}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize CV splitter: {e}")
            raise ValidationError(f"Failed to initialize CV splitter: {e}")
    
    def cross_validate_model(
        self,
        model: BaseModel,
        X: Union[np.ndarray, List[str]],
        y: np.ndarray,
        preprocessor: Optional[TextPreprocessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a single model.
        
        Args:
            model: Model to evaluate
            X: Input features
            y: Target labels
            preprocessor: Text preprocessor
            feature_extractor: Feature extractor
            return_predictions: Whether to return predictions for each fold
        
        Returns:
            Dictionary with cross-validation results
        """
        try:
            self.logger.info(f"Starting cross-validation for {model.model_name}")
            
            # Initialize results storage
            cv_results = {
                'model_name': model.model_name,
                'cv_strategy': self.cv_strategy,
                'cv_folds': self.cv_folds,
                'fold_results': [],
                'mean_scores': {},
                'std_scores': {},
                'total_time': 0
            }
            
            start_time = time.time()
            
            # Perform cross-validation
            fold_results = []
            all_predictions = []
            all_probabilities = []
            all_true_labels = []
            
            for fold, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
                self.logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
                
                # Split data
                X_train = [X[i] for i in train_idx] if isinstance(X, list) else X[train_idx]
                X_val = [X[i] for i in val_idx] if isinstance(X, list) else X[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]
                
                # Create fresh model instance for this fold
                fold_model = self._create_model_copy(model)
                
                try:
                    # Preprocess data if preprocessor provided
                    if preprocessor:
                        X_train_processed = preprocessor.preprocess_batch(X_train)
                        X_val_processed = preprocessor.preprocess_batch(X_val)
                    else:
                        X_train_processed = X_train
                        X_val_processed = X_val
                    
                    # Extract features if feature extractor provided
                    if feature_extractor and fold_model.model_name != 'BERT':
                        # Fit feature extractor on training data
                        feature_extractor_copy = self._create_feature_extractor_copy(feature_extractor)
                        X_train_features = feature_extractor_copy.fit_transform(X_train_processed)
                        X_val_features = feature_extractor_copy.transform(X_val_processed)
                    else:
                        X_train_features = X_train_processed
                        X_val_features = X_val_processed
                    
                    # Train model
                    fold_model.fit(X_train_features, y_train)
                    
                    # Make predictions
                    y_pred = fold_model.predict(X_val_features)
                    y_proba = fold_model.predict_proba(X_val_features)
                    
                    # Calculate metrics
                    fold_metrics = self._calculate_fold_metrics(y_val, y_pred, y_proba)
                    fold_metrics['fold'] = fold + 1
                    fold_metrics['train_size'] = len(X_train)
                    fold_metrics['val_size'] = len(X_val)
                    
                    fold_results.append(fold_metrics)
                    
                    if return_predictions:
                        all_predictions.extend(y_pred.tolist())
                        all_probabilities.extend(y_proba.tolist())
                        all_true_labels.extend(y_val.tolist())
                    
                    self.logger.info(f"Fold {fold + 1} completed - F1: {fold_metrics['f1']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Fold {fold + 1} failed: {e}")
                    # Add failed fold result
                    fold_results.append({
                        'fold': fold + 1,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Calculate aggregate statistics
            successful_folds = [r for r in fold_results if r.get('status') != 'failed']
            
            if successful_folds:
                for metric in self.scoring_metrics:
                    scores = [fold[metric] for fold in successful_folds if metric in fold]
                    if scores:
                        cv_results['mean_scores'][metric] = np.mean(scores)
                        cv_results['std_scores'][metric] = np.std(scores)
            
            cv_results['fold_results'] = fold_results
            cv_results['total_time'] = time.time() - start_time
            cv_results['successful_folds'] = len(successful_folds)
            cv_results['failed_folds'] = len(fold_results) - len(successful_folds)
            
            if return_predictions:
                cv_results['all_predictions'] = all_predictions
                cv_results['all_probabilities'] = all_probabilities
                cv_results['all_true_labels'] = all_true_labels
                
                # Calculate overall metrics
                overall_metrics = self._calculate_fold_metrics(
                    np.array(all_true_labels),
                    np.array(all_predictions),
                    np.array(all_probabilities)
                )
                cv_results['overall_metrics'] = overall_metrics
            
            self.logger.info(f"Cross-validation completed for {model.model_name}")
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            raise ModelError(f"Cross-validation failed: {e}")
    
    def cross_validate_multiple_models(
        self,
        models: List[BaseModel],
        X: Union[np.ndarray, List[str]],
        y: np.ndarray,
        preprocessor: Optional[TextPreprocessor] = None,
        feature_extractors: Optional[Dict[str, FeatureExtractor]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on multiple models.
        
        Args:
            models: List of models to evaluate
            X: Input features
            y: Target labels
            preprocessor: Text preprocessor
            feature_extractors: Dictionary of feature extractors
        
        Returns:
            Dictionary with results for all models
        """
        try:
            self.logger.info(f"Starting cross-validation for {len(models)} models")
            
            all_results = {}
            
            for model in models:
                try:
                    # Determine feature extractor for this model
                    feature_extractor = None
                    if feature_extractors and model.model_name != 'BERT':
                        # Try to find matching feature extractor
                        for name, extractor in feature_extractors.items():
                            if name.lower() in model.model_name.lower():
                                feature_extractor = extractor
                                break
                    
                    # Perform cross-validation
                    cv_results = self.cross_validate_model(
                        model=model,
                        X=X,
                        y=y,
                        preprocessor=preprocessor,
                        feature_extractor=feature_extractor
                    )
                    
                    all_results[model.model_name] = cv_results
                    
                except Exception as e:
                    self.logger.error(f"Cross-validation failed for {model.model_name}: {e}")
                    all_results[model.model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Create comparison table
            comparison_df = self._create_comparison_table(all_results)
            
            results_summary = {
                'models': all_results,
                'comparison_table': comparison_df,
                'best_model': self._get_best_model(all_results),
                'cv_config': {
                    'strategy': self.cv_strategy,
                    'folds': self.cv_folds,
                    'metrics': self.scoring_metrics
                }
            }
            
            self.logger.info(f"Cross-validation completed for all models")
            return results_summary
            
        except Exception as e:
            self.logger.error(f"Multiple model cross-validation failed: {e}")
            raise ModelError(f"Multiple model cross-validation failed: {e}")
    
    def _calculate_fold_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Per-class metrics
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = class_report
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate fold metrics: {e}")
            # Set default values
            for metric in self.scoring_metrics:
                metrics[metric] = 0.0
        
        return metrics
    
    def _create_model_copy(self, original_model: BaseModel) -> BaseModel:
        """Create a copy of a model for cross-validation."""
        try:
            model_info = original_model.get_model_info()
            model_type = model_info['model_name']
            config = model_info.get('config', {})
            
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
                
        except Exception as e:
            self.logger.error(f"Failed to create model copy: {e}")
            raise ModelError(f"Failed to create model copy: {e}")
    
    def _create_feature_extractor_copy(self, original_extractor: FeatureExtractor) -> FeatureExtractor:
        """Create a copy of a feature extractor."""
        try:
            config = {
                'method': original_extractor.method,
                'max_features': original_extractor.max_features,
                'ngram_range': original_extractor.ngram_range,
                'min_df': original_extractor.min_df,
                'max_df': original_extractor.max_df,
                'lowercase': original_extractor.lowercase,
                'stop_words': original_extractor.stop_words
            }
            
            return FeatureExtractor(**config)
            
        except Exception as e:
            self.logger.error(f"Failed to create feature extractor copy: {e}")
            raise ModelError(f"Failed to create feature extractor copy: {e}")
    
    def _create_comparison_table(self, all_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison table for all models."""
        try:
            comparison_data = []
            
            for model_name, results in all_results.items():
                if results.get('status') == 'failed':
                    continue
                
                row = {
                    'Model': model_name,
                    'Mean Accuracy': results.get('mean_scores', {}).get('accuracy', 0),
                    'Std Accuracy': results.get('std_scores', {}).get('accuracy', 0),
                    'Mean Precision': results.get('mean_scores', {}).get('precision', 0),
                    'Std Precision': results.get('std_scores', {}).get('precision', 0),
                    'Mean Recall': results.get('mean_scores', {}).get('recall', 0),
                    'Std Recall': results.get('std_scores', {}).get('recall', 0),
                    'Mean F1': results.get('mean_scores', {}).get('f1', 0),
                    'Std F1': results.get('std_scores', {}).get('f1', 0),
                    'Successful Folds': results.get('successful_folds', 0),
                    'Total Time (s)': results.get('total_time', 0)
                }
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            if not df.empty:
                df = df.sort_values('Mean F1', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison table: {e}")
            return pd.DataFrame()
    
    def _get_best_model(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best performing model."""
        try:
            best_model = None
            best_score = -float('inf')
            best_metric = 'f1'
            
            for model_name, results in all_results.items():
                if results.get('status') == 'failed':
                    continue
                
                mean_scores = results.get('mean_scores', {})
                if best_metric in mean_scores:
                    score = mean_scores[best_metric]
                    if score > best_score:
                        best_score = score
                        best_model = {
                            'model_name': model_name,
                            'best_metric': best_metric,
                            'best_score': score,
                            'results': results
                        }
            
            return best_model or {}
            
        except Exception as e:
            self.logger.error(f"Failed to get best model: {e}")
            return {}
    
    def generate_cv_report(self, cv_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive cross-validation report.
        
        Args:
            cv_results: Cross-validation results
            output_path: Optional path to save report
        
        Returns:
            Report as string
        """
        try:
            report = []
            report.append("# Cross-Validation Report")
            report.append(f"Strategy: {cv_results.get('cv_config', {}).get('strategy', 'Unknown')}")
            report.append(f"Folds: {cv_results.get('cv_config', {}).get('folds', 'Unknown')}")
            report.append(f"Metrics: {cv_results.get('cv_config', {}).get('metrics', [])}")
            report.append("")
            
            # Model comparison table
            if 'comparison_table' in cv_results:
                report.append("## Model Comparison")
                report.append(cv_results['comparison_table'].to_string())
                report.append("")
            
            # Best model
            if 'best_model' in cv_results and cv_results['best_model']:
                best = cv_results['best_model']
                report.append("## Best Model")
                report.append(f"Model: {best.get('model_name', 'Unknown')}")
                report.append(f"Metric: {best.get('best_metric', 'Unknown')}")
                report.append(f"Score: {best.get('best_score', 0):.4f}")
                report.append("")
            
            # Detailed results for each model
            report.append("## Detailed Results")
            for model_name, results in cv_results.get('models', {}).items():
                if results.get('status') == 'failed':
                    report.append(f"### {model_name}: FAILED")
                    report.append(f"Error: {results.get('error', 'Unknown error')}")
                else:
                    report.append(f"### {model_name}")
                    mean_scores = results.get('mean_scores', {})
                    std_scores = results.get('std_scores', {})
                    
                    for metric in self.scoring_metrics:
                        mean_val = mean_scores.get(metric, 0)
                        std_val = std_scores.get(metric, 0)
                        report.append(f"- {metric.capitalize()}: {mean_val:.4f} (+/- {std_val:.4f})")
                    
                    report.append(f"- Successful folds: {results.get('successful_folds', 0)}")
                    report.append(f"- Total time: {results.get('total_time', 0):.2f}s")
                
                report.append("")
            
            report_text = "\n".join(report)
            
            # Save report if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                self.logger.info(f"CV report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            self.logger.error(f"Failed to generate CV report: {e}")
            return f"Error generating report: {e}"
