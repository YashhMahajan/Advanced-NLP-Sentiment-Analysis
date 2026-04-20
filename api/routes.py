"""
FastAPI routes for sentiment analysis service.
"""

import time
import psutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from .schemas import (
    PredictionRequest, PredictionResponse, ModelInfoResponse,
    ErrorResponse, BatchPredictionRequest, ModelComparisonResponse,
    HealthCheckResponse, TrainingRequest, TrainingResponse
)
from src.models.model_manager import ModelManager
from src.utils.exceptions import ModelError, PredictionError, ValidationError
from src.utils.logger import get_logger

# Initialize router
router = APIRouter()

# Global model manager instance
model_manager = None
service_start_time = time.time()

logger = get_logger("api_routes")


def get_model_manager() -> ModelManager:
    """Get or initialize the model manager."""
    global model_manager
    
    if model_manager is None:
        try:
            # Try to load existing models
            models_path = Path("models")
            if models_path.exists():
                model_manager = ModelManager()
                model_manager.load_all_models(str(models_path))
                logger.info("Loaded existing models")
            else:
                # Initialize with default models
                model_manager = ModelManager()
                model_manager.register_default_models()
                logger.info("Initialized default models")
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize model service"
            )
    
    return model_manager


def handle_exceptions(func):
    """Decorator to handle exceptions and return appropriate HTTP responses."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "VALIDATION_ERROR",
                    "error_message": str(e)
                }
            )
        except PredictionError as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "PREDICTION_ERROR",
                    "error_message": str(e)
                }
            )
        except ModelError as e:
            logger.error(f"Model error: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "MODEL_ERROR",
                    "error_message": str(e)
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "INTERNAL_ERROR",
                    "error_message": "An unexpected error occurred"
                }
            )
    return wrapper


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        uptime = time.time() - service_start_time
        
        # Get system info
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Get model manager info
        manager = get_model_manager()
        manager_info = manager.get_manager_info()
        
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=manager_info['fitted_models'],
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@router.get("/models", response_model=Dict[str, ModelInfoResponse])
@handle_exceptions
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """List all available models."""
    models_info = {}
    
    for model_name, model in manager.models.items():
        try:
            model_info = model.get_model_info()
            
            # Get performance metrics if available
            performance_metrics = None
            if hasattr(manager, 'results') and 'evaluation' in manager.results:
                eval_results = manager.results['evaluation'].get(model_name, {})
                if 'f1_score' in eval_results:
                    performance_metrics = {
                        'accuracy': eval_results.get('accuracy', 0),
                        'precision': eval_results.get('precision', 0),
                        'recall': eval_results.get('recall', 0),
                        'f1_score': eval_results.get('f1_score', 0)
                    }
            
            models_info[model_name] = ModelInfoResponse(
                model_name=model_info['model_name'],
                model_type=model_info.get('model_type', 'unknown'),
                is_fitted=model_info['is_fitted'],
                classes=model_info.get('classes_', []),
                performance_metrics=performance_metrics,
                config=model_info.get('config', {})
            )
        except Exception as e:
            logger.warning(f"Failed to get info for model {model_name}: {e}")
            continue
    
    return models_info


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
@handle_exceptions
async def get_model_info(
    model_name: str,
    manager: ModelManager = Depends(get_model_manager)
):
    """Get detailed information about a specific model."""
    if model_name not in manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    model = manager.models[model_name]
    model_info = model.get_model_info()
    
    # Get performance metrics if available
    performance_metrics = None
    if hasattr(manager, 'results') and 'evaluation' in manager.results:
        eval_results = manager.results['evaluation'].get(model_name, {})
        if 'f1_score' in eval_results:
            performance_metrics = {
                'accuracy': eval_results.get('accuracy', 0),
                'precision': eval_results.get('precision', 0),
                'recall': eval_results.get('recall', 0),
                'f1_score': eval_results.get('f1_score', 0)
            }
    
    return ModelInfoResponse(
        model_name=model_info['model_name'],
        model_type=model_info.get('model_type', 'unknown'),
        is_fitted=model_info['is_fitted'],
        classes=model_info.get('classes_', []),
        performance_metrics=performance_metrics,
        config=model_info.get('config', {})
    )


@router.post("/predict", response_model=PredictionResponse)
@handle_exceptions
async def predict_sentiment(
    request: PredictionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """Predict sentiment for single or multiple texts."""
    start_time = time.time()
    
    # Convert single text to list for uniform processing
    if isinstance(request.text, str):
        texts = [request.text]
    else:
        texts = request.text
    
    # Determine which model to use
    if request.model_name:
        if request.model_name not in manager.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        model_name = request.model_name
    else:
        # Use best model if none specified
        if hasattr(manager, 'results') and 'evaluation' in manager.results:
            best_model_name, _ = manager.get_best_model()
            model_name = best_model_name
        else:
            # Fallback to first available model
            fitted_models = [name for name, model in manager.models.items() if model.is_fitted]
            if not fitted_models:
                raise HTTPException(
                    status_code=503,
                    detail="No fitted models available"
                )
            model_name = fitted_models[0]
    
    # Make predictions
    predictions = manager.predict_with_model(model_name, texts)
    
    # Get probabilities if requested
    probabilities = None
    if request.return_probabilities:
        model = manager.models[model_name]
        
        # Prepare features for probability prediction
        if model.model_name == 'BERT':
            X = texts
        else:
            if request.preprocess:
                processed_texts = manager.preprocessor.preprocess_batch(texts)
            else:
                processed_texts = texts
            
            if hasattr(model, 'feature_extractor'):
                X = model.feature_extractor.transform(processed_texts)
            else:
                feature_type = 'bow' if 'bow' in model_name else 'tfidf'
                extractor = manager.feature_extractors[feature_type]
                X = extractor.transform(processed_texts)
        
        probas = model.predict_proba(X)
        
        # Convert to dictionary format
        class_names = ['Negative', 'Neutral', 'Positive']
        probabilities = []
        for proba in probas:
            prob_dict = dict(zip(class_names, proba.tolist()))
            probabilities.append(prob_dict)
    
    # Convert predictions to string labels
    label_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    string_predictions = [label_map.get(pred, 'Unknown') for pred in predictions]
    
    processing_time = time.time() - start_time
    
    return PredictionResponse(
        success=True,
        predictions=string_predictions,
        probabilities=probabilities,
        model_used=model_name,
        processing_time=processing_time,
        input_text_count=len(texts)
    )


@router.post("/predict/batch", response_model=PredictionResponse)
@handle_exceptions
async def predict_batch(
    request: BatchPredictionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """Predict sentiment for a batch of texts."""
    start_time = time.time()
    
    # Determine which model to use
    if request.model_name:
        if request.model_name not in manager.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        model_name = request.model_name
    else:
        # Use best model if none specified
        if hasattr(manager, 'results') and 'evaluation' in manager.results:
            best_model_name, _ = manager.get_best_model()
            model_name = best_model_name
        else:
            # Fallback to first available model
            fitted_models = [name for name, model in manager.models.items() if model.is_fitted]
            if not fitted_models:
                raise HTTPException(
                    status_code=503,
                    detail="No fitted models available"
                )
            model_name = fitted_models[0]
    
    # Make predictions
    predictions = manager.predict_with_model(model_name, request.texts)
    
    # Get probabilities if requested
    probabilities = None
    if request.return_probabilities:
        model = manager.models[model_name]
        
        # Prepare features for probability prediction
        if model.model_name == 'BERT':
            X = request.texts
        else:
            if request.preprocess:
                processed_texts = manager.preprocessor.preprocess_batch(request.texts)
            else:
                processed_texts = request.texts
            
            if hasattr(model, 'feature_extractor'):
                X = model.feature_extractor.transform(processed_texts)
            else:
                feature_type = 'bow' if 'bow' in model_name else 'tfidf'
                extractor = manager.feature_extractors[feature_type]
                X = extractor.transform(processed_texts)
        
        probas = model.predict_proba(X)
        
        # Convert to dictionary format
        class_names = ['Negative', 'Neutral', 'Positive']
        probabilities = []
        for proba in probas:
            prob_dict = dict(zip(class_names, proba.tolist()))
            probabilities.append(prob_dict)
    
    # Convert predictions to string labels
    label_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    string_predictions = [label_map.get(pred, 'Unknown') for pred in predictions]
    
    processing_time = time.time() - start_time
    
    return PredictionResponse(
        success=True,
        predictions=string_predictions,
        probabilities=probabilities,
        model_used=model_name,
        processing_time=processing_time,
        input_text_count=len(request.texts)
    )


@router.get("/models/compare", response_model=ModelComparisonResponse)
@handle_exceptions
async def compare_models(manager: ModelManager = Depends(get_model_manager)):
    """Compare all available models."""
    if not hasattr(manager, 'results') or 'evaluation' not in manager.results:
        raise HTTPException(
            status_code=404,
            detail="No evaluation results available. Train and evaluate models first."
        )
    
    # Get comparison table
    comparison_df = manager.get_model_comparison_table()
    comparison_table = comparison_df.to_dict('records')
    
    # Get best model
    best_model_name, _ = manager.get_best_model()
    
    # Get model info for all models
    models_info = {}
    for model_name in manager.models.keys():
        model = manager.models[model_name]
        model_info = model.get_model_info()
        
        # Get performance metrics
        eval_results = manager.results['evaluation'].get(model_name, {})
        performance_metrics = None
        if 'f1_score' in eval_results:
            performance_metrics = {
                'accuracy': eval_results.get('accuracy', 0),
                'precision': eval_results.get('precision', 0),
                'recall': eval_results.get('recall', 0),
                'f1_score': eval_results.get('f1_score', 0)
            }
        
        models_info[model_name] = ModelInfoResponse(
            model_name=model_info['model_name'],
            model_type=model_info.get('model_type', 'unknown'),
            is_fitted=model_info['is_fitted'],
            classes=model_info.get('classes_', []),
            performance_metrics=performance_metrics,
            config=model_info.get('config', {})
        )
    
    return ModelComparisonResponse(
        models=models_info,
        comparison_table=comparison_table,
        best_model=best_model_name,
        total_models=len(manager.models)
    )


@router.post("/models/train", response_model=TrainingResponse)
@handle_exceptions
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    manager: ModelManager = Depends(get_model_manager)
):
    """Train a new model (async operation)."""
    # This is a placeholder for training functionality
    # In a real implementation, you would:
    # 1. Load training data
    # 2. Train the model
    # 3. Save the model
    # 4. Update the model manager
    
    raise HTTPException(
        status_code=501,
        detail="Model training not implemented in this demo"
    )


@router.delete("/models/{model_name}")
@handle_exceptions
async def delete_model(
    model_name: str,
    manager: ModelManager = Depends(get_model_manager)
):
    """Delete a model."""
    if model_name not in manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    # Remove model from manager
    del manager.models[model_name]
    
    # Delete model file if it exists
    model_file = Path("models") / f"{model_name}.pkl"
    if model_file.exists():
        model_file.unlink()
    
    return {"message": f"Model '{model_name}' deleted successfully"}


@router.get("/stats")
@handle_exceptions
async def get_service_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get service statistics."""
    try:
        manager_info = manager.get_manager_info()
        
        # System stats
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        stats = {
            "service": {
                "uptime": time.time() - service_start_time,
                "start_time": datetime.fromtimestamp(service_start_time).isoformat(),
                "version": "1.0.0"
            },
            "models": manager_info,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "memory_available_gb": memory_info.available / (1024**3),
                "memory_total_gb": memory_info.total / (1024**3)
            }
        }
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get service stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get service statistics"
        )
