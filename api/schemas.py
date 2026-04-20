"""
Pydantic schemas for API request/response models.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    NAIVE_BAYES = "naive_bayes"
    DECISION_TREE = "decision_tree"
    BERT = "bert"
    ENSEMBLE = "ensemble"


class FeatureType(str, Enum):
    """Supported feature types."""
    BOW = "bow"
    TFIDF = "tfidf"


class PredictionRequest(BaseModel):
    """Request schema for sentiment prediction."""
    
    text: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to analyze",
        example="I love this product! It's amazing."
    )
    
    model_name: Optional[str] = Field(
        None,
        description="Specific model to use for prediction",
        example="logistic_regression_bow"
    )
    
    return_probabilities: bool = Field(
        True,
        description="Whether to return prediction probabilities"
    )
    
    preprocess: bool = Field(
        True,
        description="Whether to preprocess the text"
    )
    
    @validator('text')
    def validate_text_input(cls, v):
        """Validate text input."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Text cannot be empty")
            return v
        elif isinstance(v, list):
            if not v:
                raise ValueError("Text list cannot be empty")
            for i, text in enumerate(v):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Text at index {i} must be a non-empty string")
            return v
        else:
            raise ValueError("Text must be a string or list of strings")


class PredictionResponse(BaseModel):
    """Response schema for sentiment prediction."""
    
    success: bool = Field(
        True,
        description="Whether the prediction was successful"
    )
    
    predictions: List[str] = Field(
        ...,
        description="Predicted sentiment labels",
        example=["Positive"]
    )
    
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Prediction probabilities for each class",
        example=[{"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}]
    )
    
    model_used: str = Field(
        ...,
        description="Model used for prediction",
        example="logistic_regression_bow"
    )
    
    processing_time: float = Field(
        ...,
        description="Time taken for prediction in seconds",
        example=0.123
    )
    
    input_text_count: int = Field(
        ...,
        description="Number of texts processed",
        example=1
    )


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    model_name: str = Field(
        ...,
        description="Name of the model"
    )
    
    model_type: str = Field(
        ...,
        description="Type of the model"
    )
    
    is_fitted: bool = Field(
        ...,
        description="Whether the model is trained"
    )
    
    classes: List[str] = Field(
        ...,
        description="Available classes"
    )
    
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Performance metrics if available"
    )
    
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Model configuration"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    success: bool = Field(False, description="Always false for errors")
    
    error_code: str = Field(
        ...,
        description="Error code",
        example="PREDICTION_ERROR"
    )
    
    error_message: str = Field(
        ...,
        description="Human-readable error message",
        example="Model prediction failed"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    
    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_items=1,
        max_items=1000
    )
    
    model_name: Optional[str] = Field(
        None,
        description="Specific model to use for prediction"
    )
    
    return_probabilities: bool = Field(
        True,
        description="Whether to return prediction probabilities"
    )
    
    preprocess: bool = Field(
        True,
        description="Whether to preprocess the text"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate text list."""
        if not v:
            raise ValueError("Texts list cannot be empty")
        
        for i, text in enumerate(v):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Text at index {i} must be a non-empty string")
        
        return v


class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison."""
    
    models: Dict[str, ModelInfoResponse] = Field(
        ...,
        description="Information about all available models"
    )
    
    comparison_table: List[Dict[str, Any]] = Field(
        ...,
        description="Performance comparison table"
    )
    
    best_model: str = Field(
        ...,
        description="Name of the best performing model"
    )
    
    total_models: int = Field(
        ...,
        description="Total number of models"
    )


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(
        ...,
        description="Service health status",
        example="healthy"
    )
    
    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )
    
    models_loaded: int = Field(
        ...,
        description="Number of loaded models"
    )
    
    uptime: float = Field(
        ...,
        description="Service uptime in seconds",
        example=3600.0
    )


class TrainingRequest(BaseModel):
    """Request schema for model training."""
    
    model_type: ModelType = Field(
        ...,
        description="Type of model to train"
    )
    
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Model configuration"
    )
    
    feature_type: Optional[FeatureType] = Field(
        None,
        description="Feature type for traditional models"
    )
    
    validation_split: float = Field(
        0.2,
        description="Validation split ratio",
        ge=0.1,
        le=0.5
    )


class TrainingResponse(BaseModel):
    """Response schema for model training."""
    
    success: bool = Field(
        ...,
        description="Whether training was successful"
    )
    
    model_name: str = Field(
        ...,
        description="Name of the trained model"
    )
    
    training_time: float = Field(
        ...,
        description="Training time in seconds"
    )
    
    training_samples: int = Field(
        ...,
        description="Number of training samples"
    )
    
    validation_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Validation metrics"
    )
    
    model_info: Optional[ModelInfoResponse] = Field(
        None,
        description="Information about the trained model"
    )
