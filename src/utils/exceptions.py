"""
Custom exceptions for the Advanced NLP Sentiment Analysis package.
"""

class NLPAException(Exception):
    """Base exception for all NLP Analysis errors."""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataProcessingError(NLPAException):
    """Raised when data processing operations fail."""
    
    def __init__(self, message: str, operation: str = None):
        self.operation = operation
        error_code = "DATA_PROCESSING_ERROR"
        super().__init__(message, error_code)


class ModelError(NLPAException):
    """Raised when model operations fail."""
    
    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        error_code = "MODEL_ERROR"
        super().__init__(message, error_code)


class PredictionError(NLPAException):
    """Raised when prediction operations fail."""
    
    def __init__(self, message: str, input_data: str = None):
        self.input_data = input_data
        error_code = "PREDICTION_ERROR"
        super().__init__(message, error_code)


class ValidationError(NLPAException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: str = None):
        self.field_name = field_name
        error_code = "VALIDATION_ERROR"
        super().__init__(message, error_code)


class ConfigurationError(NLPAException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        error_code = "CONFIGURATION_ERROR"
        super().__init__(message, error_code)
