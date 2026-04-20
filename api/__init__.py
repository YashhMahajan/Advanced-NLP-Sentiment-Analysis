"""
FastAPI service for Advanced NLP sentiment analysis.
"""

from .main import app
from .schemas import PredictionRequest, PredictionResponse, ModelInfoResponse
from .routes import router

__all__ = [
    "app",
    "PredictionRequest",
    "PredictionResponse", 
    "ModelInfoResponse",
    "router"
]
