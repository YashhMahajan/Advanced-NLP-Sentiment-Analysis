"""
BERT transformer model for sentiment analysis.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
import numpy as np
from typing import Dict, Any, List, Union, Optional
import logging as py_logging

from .base_model import BaseModel
from ..utils.exceptions import ModelError
from ..utils.logger import LoggerMixin

# Suppress transformers logging
py_logging.getLogger("transformers").setLevel(py_logging.ERROR)


class BERTModel(BaseModel):
    """
    BERT-based transformer model for sentiment analysis.
    
    Uses pre-trained BERT models fine-tuned for sentiment classification.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize BERT model.
        
        Args:
            config: Model configuration
        """
        default_config = {
            'model_name': 'distilbert-base-uncased',  # Lighter, faster model
            'num_labels': 3,  # Negative, Neutral, Positive
            'max_length': 128,
            'batch_size': 16,
            'learning_rate': 3e-5,  # Slightly higher for faster convergence
            'num_epochs': 2,  # Reduced epochs for efficiency
            'warmup_steps': 100,  # Reduced warmup
            'weight_decay': 0.01,
            'logging_dir': './logs',
            'output_dir': './bert_results',
            'early_stopping_patience': 1,
            'evaluation_strategy': 'epoch',
            'save_strategy': 'epoch',
            'load_best_model_at_end': True
        }
        if config:
            default_config.update(config)
        
        super().__init__("BERT", default_config)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_map = {-1: 0, 0: 1, 1: 2}  # Map to 0, 1, 2 for BERT
        self.reverse_label_map = {0: -1, 1: 0, 2: 1}
        
        self.logger.info(f"Initialized BERT model with {self.config['model_name']}")
    
    def build_model(self) -> None:
        """Build and return BERT model and tokenizer."""
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'],
                use_fast=True
            )
            
            # Initialize model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model_name'],
                num_labels=self.config['num_labels']
            )
            
            self.logger.info(f"Built BERT model: {self.config['model_name']}")
            
        except Exception as e:
            raise ModelError(f"Failed to build BERT model: {e}", self.model_name)
    
    def _preprocess_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess texts for BERT input.
        
        Args:
            texts: List of input texts
        
        Returns:
            Dictionary with tokenized inputs
        """
        if self.tokenizer is None:
            raise ModelError("Tokenizer not initialized", self.model_name)
        
        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config['max_length'],
                return_tensors='pt'
            )
            
            return inputs
            
        except Exception as e:
            raise ModelError(f"Text preprocessing failed: {e}", self.model_name)
    
    def _create_dataset(self, texts: List[str], labels: np.ndarray) -> Dataset:
        """
        Create Hugging Face dataset from texts and labels.
        
        Args:
            texts: List of input texts
            labels: Array of labels
        
        Returns:
            Hugging Face dataset
        """
        try:
            # Map labels to BERT format
            mapped_labels = [self.label_map[label] for label in labels]
            
            # Tokenize texts
            tokenized_inputs = self._preprocess_texts(texts)
            
            # Create dataset
            dataset_dict = {
                'input_ids': tokenized_inputs['input_ids'],
                'attention_mask': tokenized_inputs['attention_mask'],
                'labels': torch.tensor(mapped_labels)
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            return dataset
            
        except Exception as e:
            raise ModelError(f"Dataset creation failed: {e}", self.model_name)
    
    def fit(self, X: Union[np.ndarray, List[str]], y: np.ndarray) -> 'BERTModel':
        """Train the BERT model on provided data."""
        try:
            if self.model is None:
                self.build_model()
            
            # Convert X to list if it's numpy array
            if isinstance(X, np.ndarray):
                X = X.tolist()
            
            # Create dataset
            dataset = self._create_dataset(X, y)
            
            # Split dataset (simple train/validation split)
            train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = train_test_split['train']
            eval_dataset = train_test_split['test']
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=self.config['output_dir'],
                num_train_epochs=self.config['num_epochs'],
                per_device_train_batch_size=self.config['batch_size'],
                per_device_eval_batch_size=self.config['batch_size'],
                warmup_steps=self.config['warmup_steps'],
                weight_decay=self.config['weight_decay'],
                logging_dir=self.config['logging_dir'],
                logging_steps=50,  # More frequent logging for better monitoring
                evaluation_strategy=self.config.get('evaluation_strategy', 'epoch'),
                save_strategy=self.config.get('save_strategy', 'epoch'),
                load_best_model_at_end=self.config.get('load_best_model_at_end', True),
                metric_for_best_model="f1",
                greater_is_better=True,
                report_to=None,  # Disable wandb/tensorboard
                disable_tqdm=False,  # Enable progress bars for better UX
                dataloader_num_workers=2,  # Parallel data loading
                fp16=True,  # Use mixed precision for faster training
                gradient_accumulation_steps=1,  # No accumulation needed with batch_size=16
                learning_rate=self.config['learning_rate']
            )
            
            # Define metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                
                # Map back to original labels
                original_predictions = [self.reverse_label_map[pred] for pred in predictions]
                original_labels = [self.reverse_label_map[label] for label in labels]
                
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                
                accuracy = accuracy_score(original_labels, original_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    original_labels, original_predictions, average='weighted'
                )
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            
            # Add early stopping callback if patience is specified
            callbacks = []
            if self.config.get('early_stopping_patience'):
                from transformers import EarlyStoppingCallback
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config['early_stopping_patience']
                    )
                )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks
            )
            
            # Train model
            self.trainer.train()
            
            self.is_fitted = True
            self.classes_ = np.array([-1, 0, 1])  # Original class labels
            
            self.logger.info(f"BERT model trained on {len(X)} samples")
            return self
            
        except Exception as e:
            self.logger.error(f"BERT training failed: {e}")
            raise ModelError(f"Training failed: {e}", self.model_name)
    
    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            # Convert X to list if it's numpy array
            if isinstance(X, np.ndarray):
                X = X.tolist()
            
            # Preprocess texts
            inputs = self._preprocess_texts(X)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Map back to original labels
            original_predictions = [self.reverse_label_map[pred.item()] for pred in predictions]
            
            return np.array(original_predictions)
            
        except Exception as e:
            raise ModelError(f"Prediction failed: {e}", self.model_name)
    
    def predict_proba(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction", self.model_name)
        
        try:
            # Convert X to list if it's numpy array
            if isinstance(X, np.ndarray):
                X = X.tolist()
            
            # Preprocess texts
            inputs = self._preprocess_texts(X)
            
            # Get probabilities
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            return probabilities.numpy()
            
        except Exception as e:
            raise ModelError(f"Probability prediction failed: {e}", self.model_name)
    
    def save_model(self, filepath: str) -> str:
        """Save the trained BERT model to disk."""
        try:
            import os
            
            if not self.is_fitted:
                raise ModelError("Model must be fitted before saving", self.model_name)
            
            # Create directory if it doesn't exist
            os.makedirs(filepath, exist_ok=True)
            
            # Save model and tokenizer
            if self.model:
                self.model.save_pretrained(filepath)
            if self.tokenizer:
                self.tokenizer.save_pretrained(filepath)
            
            # Save config
            import json
            config_path = os.path.join(filepath, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"BERT model saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save BERT model: {e}")
            raise ModelError(f"Failed to save model: {e}", self.model_name)
    
    def load_model(self, filepath: str) -> 'BERTModel':
        """Load a trained BERT model from disk."""
        try:
            import os
            import json
            
            # Load config
            config_path = os.path.join(filepath, 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(filepath)
            self.model = AutoModelForSequenceClassification.from_pretrained(filepath)
            
            self.is_fitted = True
            self.classes_ = np.array([-1, 0, 1])
            
            self.logger.info(f"BERT model loaded from {filepath}")
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to load BERT model: {e}")
            raise ModelError(f"Failed to load model: {e}", self.model_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the BERT model."""
        info = super().get_model_info()
        
        if self.model:
            info.update({
                'model_type': 'transformer',
                'base_model': self.config.get('model_name'),
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'max_sequence_length': self.config.get('max_length'),
                'num_labels': self.config.get('num_labels')
            })
        
        return info
