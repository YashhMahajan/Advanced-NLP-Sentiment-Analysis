"""
Model validation utilities for comprehensive testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.base_model import BaseModel
from ..utils.exceptions import ValidationError, ModelError
from ..utils.logger import LoggerMixin


class ModelValidator(LoggerMixin):
    """
    Comprehensive model validation and testing utilities.
    
    Features:
    - Performance metrics calculation
    - Error analysis
    - Calibration analysis
    - Visualization generation
    - Statistical testing
    """
    
    def __init__(self, target_names: List[str] = None):
        """
        Initialize model validator.
        
        Args:
            target_names: Names for target classes
        """
        self.target_names = target_names or ['Negative', 'Neutral', 'Positive']
        self.logger.info("ModelValidator initialized")
    
    def validate_model_performance(
        self,
        model: BaseModel,
        X_test: Union[np.ndarray, List[str]],
        y_test: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model performance validation.
        
        Args:
            model: Trained model to validate
            X_test: Test features
            y_test: Test labels
            detailed: Whether to include detailed analysis
        
        Returns:
            Dictionary with validation results
        """
        try:
            if not model.is_fitted:
                raise ValidationError("Model must be fitted before validation")
            
            self.logger.info(f"Starting performance validation for {model.model_name}")
            
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Basic metrics
            basic_metrics = self._calculate_basic_metrics(y_test, y_pred, y_proba)
            
            validation_results = {
                'model_name': model.model_name,
                'basic_metrics': basic_metrics,
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist(),
                'true_labels': y_test.tolist(),
                'target_names': self.target_names
            }
            
            if detailed:
                # Detailed analysis
                validation_results.update({
                    'classification_report': self._get_classification_report(y_test, y_pred),
                    'confusion_matrix': self._get_confusion_matrix(y_test, y_pred),
                    'error_analysis': self._analyze_errors(y_test, y_pred, y_proba),
                    'confidence_analysis': self._analyze_confidence(y_test, y_pred, y_proba),
                    'class_wise_metrics': self._calculate_class_wise_metrics(y_test, y_pred, y_proba)
                })
            
            self.logger.info(f"Performance validation completed for {model.model_name}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            raise ValidationError(f"Performance validation failed: {e}")
    
    def _calculate_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Macro averages
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, class_name in enumerate(self.target_names):
                if i < len(precision_per_class):
                    metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
                    metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
                    metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
            
        except Exception as e:
            self.logger.error(f"Failed to calculate basic metrics: {e}")
            # Set default values
            default_metrics = ['accuracy', 'precision', 'recall', 'f1', 'precision_macro', 'recall_macro', 'f1_macro']
            for metric in default_metrics:
                metrics[metric] = 0.0
        
        return metrics
    
    def _get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Get detailed classification report."""
        try:
            return classification_report(y_true, y_pred, target_names=self.target_names, output_dict=True, zero_division=0)
        except Exception as e:
            self.logger.error(f"Failed to generate classification report: {e}")
            return {}
    
    def _get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Get confusion matrix with analysis."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate confusion matrix statistics
            cm_analysis = {
                'matrix': cm.tolist(),
                'normalized': cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].tolist(),
                'per_class_accuracy': np.diag(cm) / cm.sum(axis=1).tolist(),
                'total_correct': np.diag(cm).sum(),
                'total_incorrect': cm.sum() - np.diag(cm).sum()
            }
            
            return cm_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze confusion matrix: {e}")
            return {}
    
    def _analyze_errors(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        try:
            # Find misclassified samples
            misclassified_mask = y_true != y_pred
            misclassified_indices = np.where(misclassified_mask)[0]
            
            error_analysis = {
                'total_errors': len(misclassified_indices),
                'error_rate': len(misclassified_indices) / len(y_true),
                'misclassified_indices': misclassified_indices.tolist(),
                'error_types': {},
                'confidence_errors': {}
            }
            
            # Analyze error types
            for i in misclassified_indices:
                true_class = y_true[i]
                pred_class = y_pred[i]
                error_type = f"{true_class}_to_{pred_class}"
                
                if error_type not in error_analysis['error_types']:
                    error_analysis['error_types'][error_type] = 0
                error_analysis['error_types'][error_type] += 1
            
            # Analyze confidence in errors
            error_confidences = []
            for i in misclassified_indices:
                confidence = np.max(y_proba[i])
                error_confidences.append(confidence)
            
            if error_confidences:
                error_analysis['confidence_errors'] = {
                    'mean_confidence': np.mean(error_confidences),
                    'std_confidence': np.std(error_confidences),
                    'min_confidence': np.min(error_confidences),
                    'max_confidence': np.max(error_confidences)
                }
            
            return error_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze errors: {e}")
            return {}
    
    def _analyze_confidence(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction confidence."""
        try:
            # Calculate confidence scores
            confidence_scores = np.max(y_proba, axis=1)
            
            # Separate correct and incorrect predictions
            correct_mask = y_true == y_pred
            correct_confidence = confidence_scores[correct_mask]
            incorrect_confidence = confidence_scores[~correct_mask]
            
            confidence_analysis = {
                'overall': {
                    'mean_confidence': np.mean(confidence_scores),
                    'std_confidence': np.std(confidence_scores),
                    'min_confidence': np.min(confidence_scores),
                    'max_confidence': np.max(confidence_scores)
                },
                'correct_predictions': {
                    'mean_confidence': np.mean(correct_confidence) if len(correct_confidence) > 0 else 0,
                    'std_confidence': np.std(correct_confidence) if len(correct_confidence) > 0 else 0,
                    'count': len(correct_confidence)
                },
                'incorrect_predictions': {
                    'mean_confidence': np.mean(incorrect_confidence) if len(incorrect_confidence) > 0 else 0,
                    'std_confidence': np.std(incorrect_confidence) if len(incorrect_confidence) > 0 else 0,
                    'count': len(incorrect_confidence)
                }
            }
            
            # Confidence thresholds analysis
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            threshold_analysis = {}
            
            for threshold in thresholds:
                high_confidence_mask = confidence_scores >= threshold
                if np.sum(high_confidence_mask) > 0:
                    high_confidence_correct = np.sum(y_true[high_confidence_mask] == y_pred[high_confidence_mask])
                    high_confidence_accuracy = high_confidence_correct / np.sum(high_confidence_mask)
                    
                    threshold_analysis[f'threshold_{threshold}'] = {
                        'samples': np.sum(high_confidence_mask),
                        'accuracy': high_confidence_accuracy,
                        'percentage': np.sum(high_confidence_mask) / len(y_true)
                    }
            
            confidence_analysis['threshold_analysis'] = threshold_analysis
            
            return confidence_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze confidence: {e}")
            return {}
    
    def _calculate_class_wise_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate class-wise performance metrics."""
        try:
            class_metrics = {}
            
            for i, class_name in enumerate(self.target_names):
                # Convert to binary classification for this class
                y_true_binary = (y_true == i).astype(int)
                y_pred_binary = (y_pred == i).astype(int)
                y_proba_binary = y_proba[:, i]
                
                # Calculate binary metrics
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                # Calculate AUC if possible
                try:
                    auc = roc_auc_score(y_true_binary, y_proba_binary)
                except:
                    auc = 0.0
                
                class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'support': np.sum(y_true_binary),
                    'true_positives': np.sum((y_true_binary == 1) & (y_pred_binary == 1)),
                    'false_positives': np.sum((y_true_binary == 0) & (y_pred_binary == 1)),
                    'true_negatives': np.sum((y_true_binary == 0) & (y_pred_binary == 0)),
                    'false_negatives': np.sum((y_true_binary == 1) & (y_pred_binary == 0))
                }
            
            return class_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate class-wise metrics: {e}")
            return {}
    
    def compare_models(
        self,
        models: List[BaseModel],
        X_test: Union[np.ndarray, List[str]],
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple models.
        
        Args:
            models: List of models to compare
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info(f"Comparing {len(models)} models")
            
            comparison_results = {
                'models': {},
                'comparison_table': None,
                'best_model': None
            }
            
            # Validate each model
            for model in models:
                try:
                    validation_results = self.validate_model_performance(model, X_test, y_test, detailed=True)
                    comparison_results['models'][model.model_name] = validation_results
                except Exception as e:
                    self.logger.error(f"Validation failed for {model.model_name}: {e}")
                    comparison_results['models'][model.model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Create comparison table
            comparison_table = self._create_model_comparison_table(comparison_results['models'])
            comparison_results['comparison_table'] = comparison_table
            
            # Find best model
            best_model = self._find_best_model(comparison_results['models'])
            comparison_results['best_model'] = best_model
            
            self.logger.info("Model comparison completed")
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            raise ValidationError(f"Model comparison failed: {e}")
    
    def _create_model_comparison_table(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison table for multiple models."""
        try:
            comparison_data = []
            
            for model_name, results in model_results.items():
                if results.get('status') == 'failed':
                    continue
                
                basic_metrics = results.get('basic_metrics', {})
                
                row = {
                    'Model': model_name,
                    'Accuracy': basic_metrics.get('accuracy', 0),
                    'Precision': basic_metrics.get('precision', 0),
                    'Recall': basic_metrics.get('recall', 0),
                    'F1-Score': basic_metrics.get('f1', 0),
                    'Precision Macro': basic_metrics.get('precision_macro', 0),
                    'Recall Macro': basic_metrics.get('recall_macro', 0),
                    'F1 Macro': basic_metrics.get('f1_macro', 0)
                }
                
                # Add class-wise metrics
                for class_name in self.target_names:
                    class_metrics = results.get('class_wise_metrics', {}).get(class_name, {})
                    row[f'{class_name} F1'] = class_metrics.get('f1', 0)
                
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            if not df.empty:
                df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison table: {e}")
            return pd.DataFrame()
    
    def _find_best_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing model."""
        try:
            best_model = None
            best_score = -float('inf')
            
            for model_name, results in model_results.items():
                if results.get('status') == 'failed':
                    continue
                
                basic_metrics = results.get('basic_metrics', {})
                f1_score = basic_metrics.get('f1', 0)
                
                if f1_score > best_score:
                    best_score = f1_score
                    best_model = {
                        'model_name': model_name,
                        'f1_score': f1_score,
                        'metrics': basic_metrics
                    }
            
            return best_model or {}
            
        except Exception as e:
            self.logger.error(f"Failed to find best model: {e}")
            return {}
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: Validation results
        
        Returns:
            Report as string
        """
        try:
            report = []
            report.append("# Model Validation Report")
            report.append("")
            
            if 'model_name' in validation_results:
                report.append(f"## Model: {validation_results['model_name']}")
                report.append("")
            
            # Basic metrics
            if 'basic_metrics' in validation_results:
                report.append("### Basic Performance Metrics")
                metrics = validation_results['basic_metrics']
                
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric in metrics:
                        report.append(f"- {metric.capitalize()}: {metrics[metric]:.4f}")
                
                report.append("")
            
            # Class-wise metrics
            if 'class_wise_metrics' in validation_results:
                report.append("### Class-wise Performance")
                class_metrics = validation_results['class_wise_metrics']
                
                for class_name, metrics in class_metrics.items():
                    report.append(f"#### {class_name}")
                    report.append(f"- Precision: {metrics['precision']:.4f}")
                    report.append(f"- Recall: {metrics['recall']:.4f}")
                    report.append(f"- F1-Score: {metrics['f1']:.4f}")
                    report.append(f"- AUC: {metrics['auc']:.4f}")
                    report.append("")
            
            # Error analysis
            if 'error_analysis' in validation_results:
                error_analysis = validation_results['error_analysis']
                report.append("### Error Analysis")
                report.append(f"- Total Errors: {error_analysis.get('total_errors', 0)}")
                report.append(f"- Error Rate: {error_analysis.get('error_rate', 0):.4f}")
                
                if 'confidence_errors' in error_analysis:
                    conf_errors = error_analysis['confidence_errors']
                    report.append(f"- Mean Confidence in Errors: {conf_errors.get('mean_confidence', 0):.4f}")
                
                report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Failed to generate validation report: {e}")
            return f"Error generating report: {e}"
