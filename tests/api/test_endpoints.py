"""
API endpoint tests for Advanced NLP sentiment analysis service.
"""

import pytest
from fastapi.testclient import TestClient
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from api.main import app
    from api.routes import get_model_manager
except ImportError:
    pytest.skip("API modules not available", allow_module_level=True)

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns correct status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data
        assert "uptime" in data


class TestModels:
    """Test model-related endpoints."""
    
    def test_list_models(self):
        """Test listing all available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
        
        # Check if models have required fields
        for model_name, model_info in data.items():
            assert "model_name" in model_info
            assert "model_type" in model_info
            assert "is_fitted" in model_info
            assert "classes" in model_info
    
    def test_get_model_info(self):
        """Test getting information about a specific model."""
        # First, get list of models to find a valid one
        models_response = client.get("/api/v1/models")
        if models_response.status_code == 200:
            models = models_response.json()
            if models:
                model_name = list(models.keys())[0]
                
                response = client.get(f"/api/v1/models/{model_name}")
                assert response.status_code == 200
                
                data = response.json()
                assert data["model_name"] == model_name
                assert "model_type" in data
                assert "is_fitted" in data
    
    def test_get_nonexistent_model(self):
        """Test getting information about a non-existent model."""
        response = client.get("/api/v1/models/nonexistent_model")
        assert response.status_code == 404


class TestPrediction:
    """Test prediction endpoints."""
    
    def test_predict_single_text(self):
        """Test single text prediction."""
        payload = {
            "text": "I love this product! It is amazing.",
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["input_text_count"] == 1
        assert "model_used" in data
        assert "processing_time" in data
        
        # Check if prediction is valid
        valid_predictions = ["Negative", "Neutral", "Positive"]
        assert data["predictions"][0] in valid_predictions
        
        # Check probabilities if requested
        if data["probabilities"]:
            assert len(data["probabilities"]) == 1
            prob_dict = data["probabilities"][0]
            for pred_class in valid_predictions:
                assert pred_class in prob_dict
                assert 0 <= prob_dict[pred_class] <= 1
    
    def test_predict_multiple_texts(self):
        """Test multiple text prediction."""
        payload = {
            "text": [
                "I love this product!",
                "This is terrible.",
                "The service was okay."
            ],
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 3
        assert data["input_text_count"] == 3
        
        if data["probabilities"]:
            assert len(data["probabilities"]) == 3
    
    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint."""
        payload = {
            "texts": [
                "Excellent product!",
                "Not worth the money.",
                "Average experience."
            ],
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 3
        assert data["input_text_count"] == 3
    
    def test_predict_empty_text(self):
        """Test prediction with empty text."""
        payload = {
            "text": "",
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 400
    
    def test_predict_invalid_text_list(self):
        """Test prediction with invalid text list."""
        payload = {
            "text": ["Valid text", ""],
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 400
    
    def test_predict_with_specific_model(self):
        """Test prediction with specific model."""
        # First get available models
        models_response = client.get("/api/v1/models")
        if models_response.status_code == 200:
            models = models_response.json()
            if models:
                model_name = list(models.keys())[0]
                
                payload = {
                    "text": "This is a test.",
                    "model_name": model_name,
                    "return_probabilities": True
                }
                
                response = client.post("/api/v1/predict", json=payload)
                assert response.status_code == 200
                
                data = response.json()
                assert data["model_used"] == model_name
    
    def test_predict_with_nonexistent_model(self):
        """Test prediction with non-existent model."""
        payload = {
            "text": "This is a test.",
            "model_name": "nonexistent_model",
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 404


class TestModelComparison:
    """Test model comparison endpoint."""
    
    def test_model_comparison(self):
        """Test model comparison endpoint."""
        response = client.get("/api/v1/models/compare")
        
        # This might return 404 if no evaluation results are available
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "models" in data
            assert "best_model" in data
            assert "total_models" in data


class TestServiceStats:
    """Test service statistics endpoint."""
    
    def test_service_stats(self):
        """Test service statistics endpoint."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "models" in data
        assert "system" in data
        
        # Check service info
        service_info = data["service"]
        assert "uptime" in service_info
        assert "version" in service_info
        
        # Check system info
        system_info = data["system"]
        assert "cpu_percent" in system_info
        assert "memory_percent" in system_info


class TestErrorHandling:
    """Test error handling in API."""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        payload = {}  # Missing required 'text' field
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        payload = {
            "text": 123,  # Should be string or list
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns basic info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


@pytest.fixture(scope="session")
def setup_test_data():
    """Setup test data for API tests."""
    # This could be used to setup test models or data
    pass


@pytest.fixture(autouse=True)
def cleanup_after_tests():
    """Cleanup after tests."""
    yield
    # Add any cleanup code here if needed
