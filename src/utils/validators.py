"""
Input validation utilities.
"""

import re
from typing import Union, List, Dict, Any
from .exceptions import ValidationError


def validate_text_input(text: Union[str, List[str]]) -> List[str]:
    """
    Validate text input for prediction.
    
    Args:
        text: Single text string or list of text strings
    
    Returns:
        List of validated text strings
    
    Raises:
        ValidationError: If input is invalid
    """
    if text is None:
        raise ValidationError("Input text cannot be None")
    
    # Convert single string to list
    if isinstance(text, str):
        text_list = [text]
    elif isinstance(text, list):
        text_list = text
    else:
        raise ValidationError("Input must be a string or list of strings")
    
    # Validate each text
    validated_texts = []
    for i, txt in enumerate(text_list):
        if not isinstance(txt, str):
            raise ValidationError(f"Text at index {i} must be a string")
        
        # Remove excessive whitespace and check if empty after cleaning
        cleaned = txt.strip()
        if not cleaned:
            raise ValidationError(f"Text at index {i} cannot be empty or whitespace only")
        
        # Check for extremely long text (potential memory issue)
        if len(cleaned) > 10000:
            raise ValidationError(f"Text at index {i} exceeds maximum length of 10000 characters")
        
        validated_texts.append(cleaned)
    
    return validated_texts


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model configuration parameters.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Validated configuration
    
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")
    
    # Define required and optional fields
    required_fields = ['model_type']
    optional_fields = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'random_state': 42,
        'max_iter': 1000
    }
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Required field '{field}' missing from configuration")
    
    # Validate model_type
    valid_model_types = ['logistic_regression', 'naive_bayes', 'decision_tree', 'bert', 'ensemble']
    if config['model_type'] not in valid_model_types:
        raise ValidationError(
            f"Invalid model_type '{config['model_type']}'. "
            f"Valid types: {valid_model_types}"
        )
    
    # Set default values for optional fields
    validated_config = config.copy()
    for field, default_value in optional_fields.items():
        if field not in validated_config:
            validated_config[field] = default_value
    
    # Validate specific fields
    if 'max_features' in validated_config:
        if not isinstance(validated_config['max_features'], int) or validated_config['max_features'] <= 0:
            raise ValidationError("max_features must be a positive integer")
    
    if 'ngram_range' in validated_config:
        ngram_range = validated_config['ngram_range']
        if (not isinstance(ngram_range, tuple) or 
            len(ngram_range) != 2 or 
            not all(isinstance(x, int) and x > 0 for x in ngram_range) or
            ngram_range[0] > ngram_range[1]):
            raise ValidationError("ngram_range must be a tuple of two positive integers (min_ngram, max_ngram)")
    
    return validated_config


def validate_file_path(file_path: str, must_exist: bool = True) -> str:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
    
    Returns:
        Validated file path
    
    Raises:
        ValidationError: If path is invalid
    """
    if not isinstance(file_path, str):
        raise ValidationError("File path must be a string")
    
    if not file_path.strip():
        raise ValidationError("File path cannot be empty")
    
    from pathlib import Path
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    if must_exist and not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    return str(path.absolute())


def validate_hyperparameters(params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Validate hyperparameters for specific model types.
    
    Args:
        params: Hyperparameter dictionary
        model_type: Type of model
    
    Returns:
        Validated hyperparameters
    
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError("Hyperparameters must be a dictionary")
    
    # Model-specific validation
    if model_type == 'logistic_regression':
        if 'C' in params and params['C'] <= 0:
            raise ValidationError("C parameter must be positive for logistic regression")
    
    elif model_type == 'naive_bayes':
        if 'alpha' in params and params['alpha'] < 0:
            raise ValidationError("alpha parameter must be non-negative for naive bayes")
    
    elif model_type == 'decision_tree':
        if 'max_depth' in params and params['max_depth'] is not None and params['max_depth'] <= 0:
            raise ValidationError("max_depth must be positive or None for decision tree")
        if 'min_samples_split' in params and params['min_samples_split'] < 2:
            raise ValidationError("min_samples_split must be at least 2 for decision tree")
    
    return params
