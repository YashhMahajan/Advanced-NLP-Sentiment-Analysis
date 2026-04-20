"""
Advanced NLP Sentiment Analysis Package

A production-grade sentiment analysis system with modular architecture,
supporting traditional ML models and transformer-based approaches.
"""

__version__ = "1.0.0"
__author__ = "NLP Team"

from .models import ModelManager
from .preprocessing import TextPreprocessor
from .ensemble import EnsembleModel
from .utils import setup_logger

__all__ = [
    "ModelManager",
    "TextPreprocessor", 
    "EnsembleModel",
    "setup_logger"
]
