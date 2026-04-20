"""
Integration tests for the full Advanced NLP sentiment analysis pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.models.model_manager import ModelManager
    from src.preprocessing import TextPreprocessor, FeatureExtractor
    from src.validation import CrossValidator, ModelValidator
    from src.ensemble import VotingEnsemble, StackingEnsemble
except ImportError:
    pytest.skip("Source modules not available", allow_module_level=True)


class TestFullPipeline:
    """Test the complete pipeline from preprocessing to prediction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        texts = [
            "I love this product! It's amazing and works perfectly.",
            "This is terrible. I hate it and want my money back.",
            "The service was okay, nothing special but not bad either.",
            "Wow! This exceeded all my expectations. Highly recommended!",
            "Complete waste of time and money. Very disappointed.",
            "Average quality, does what it says but nothing more.",
            "Outstanding customer service and excellent product quality!",
            "Poor packaging, product arrived damaged. Not happy.",
            "Good value for money, would recommend to others.",
            "Not worth the price, there are better alternatives."
        ]
        
        labels = np.array([1, -1, 0, 1, -1, 0, 1, -1, 1, -1])  # 1: Positive, 0: Neutral, -1: Negative
        
        return texts, labels
    
    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance."""
        manager = ModelManager()
        manager.register_default_models()
        return manager
    
    def test_preprocessing_pipeline(self, sample_data):
        """Test the complete preprocessing pipeline."""
        texts, labels = sample_data
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Preprocess texts
        processed_texts = preprocessor.preprocess_batch(texts)
        
        assert len(processed_texts) == len(texts)
        assert all(isinstance(text, str) for text in processed_texts)
        assert all(text.strip() for text in processed_texts)
        
        # Test feature extraction
        feature_extractor = FeatureExtractor(method='tfidf', max_features=100)
        features = feature_extractor.fit_transform(processed_texts)
        
        assert features.shape[0] == len(texts)
        assert features.shape[1] <= 100
    
    def test_model_training_and_prediction(self, sample_data, model_manager):
        """Test model training and prediction pipeline."""
        texts, labels = sample_data
        
        # Train models
        training_results = model_manager.train_all_models(texts, labels, feature_types=['tfidf'])
        
        assert 'training' in training_results
        assert len(training_results['training']) > 0
        
        # Check that at least some models were trained successfully
        successful_models = [
            name for name, result in training_results['training'].items()
            if result['status'] == 'success'
        ]
        assert len(successful_models) > 0
        
        # Test predictions
        for model_name in successful_models[:2]:  # Test first 2 successful models
            predictions = model_manager.predict_with_model(model_name, texts)
            assert len(predictions) == len(texts)
            assert all(pred in [-1, 0, 1] for pred in predictions)
    
    def test_model_evaluation(self, sample_data, model_manager):
        """Test model evaluation pipeline."""
        texts, labels = sample_data
        
        # Split data
        split_idx = len(texts) // 2
        train_texts, test_texts = texts[:split_idx], texts[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        
        # Train models
        model_manager.train_all_models(train_texts, train_labels, feature_types=['tfidf'])
        
        # Evaluate models
        evaluation_results = model_manager.evaluate_all_models(test_texts, test_labels)
        
        assert 'evaluation' in evaluation_results
        
        # Check evaluation metrics
        for model_name, results in evaluation_results['evaluation'].items():
            if 'status' in results and results['status'] == 'failed':
                continue
            
            assert 'accuracy' in results
            assert 'precision' in results
            assert 'recall' in results
            assert 'f1_score' in results
            assert 0 <= results['accuracy'] <= 1
            assert 0 <= results['f1_score'] <= 1
    
    def test_cross_validation(self, sample_data):
        """Test cross-validation pipeline."""
        texts, labels = sample_data
        
        # Initialize components
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor(method='tfidf', max_features=50)
        cross_validator = CrossValidator(cv_folds=3)  # Use fewer folds for small dataset
        
        # Create a simple model for testing
        from src.models.traditional_models import LogisticRegressionModel
        model = LogisticRegressionModel({'max_iter': 100})
        
        # Perform cross-validation
        cv_results = cross_validator.cross_validate_model(
            model=model,
            X=texts,
            y=labels,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor
        )
        
        assert cv_results['model_name'] == 'LogisticRegression'
        assert cv_results['cv_folds'] == 3
        assert 'mean_scores' in cv_results
        assert 'std_scores' in cv_results
        assert len(cv_results['fold_results']) == 3
        
        # Check that scores are reasonable
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in cv_results['mean_scores']:
                assert 0 <= cv_results['mean_scores'][metric] <= 1
    
    def test_ensemble_methods(self, sample_data):
        """Test ensemble methods."""
        texts, labels = sample_data
        
        # Split data
        split_idx = len(texts) // 2
        train_texts, test_texts = texts[:split_idx], texts[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        
        # Preprocess and extract features
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor(method='tfidf', max_features=50)
        
        train_features = feature_extractor.fit_transform(preprocessor.preprocess_batch(train_texts))
        test_features = feature_extractor.transform(preprocessor.preprocess_batch(test_texts))
        
        # Create base models
        from src.models.traditional_models import LogisticRegressionModel, NaiveBayesModel
        
        models = [
            LogisticRegressionModel({'max_iter': 100}),
            NaiveBayesModel()
        ]
        
        # Test voting ensemble
        voting_ensemble = VotingEnsemble(
            name="TestVoting",
            models=models,
            voting="soft"
        )
        
        voting_ensemble.fit(train_features, train_labels)
        voting_predictions = voting_ensemble.predict(test_features)
        voting_probabilities = voting_ensemble.predict_proba(test_features)
        
        assert len(voting_predictions) == len(test_labels)
        assert voting_probabilities.shape[0] == len(test_labels)
        assert voting_probabilities.shape[1] == 3  # 3 classes
        
        # Test stacking ensemble
        stacking_ensemble = StackingEnsemble(
            name="TestStacking",
            models=models,
            cv_folds=2  # Use fewer folds for small dataset
        )
        
        stacking_ensemble.fit(train_features, train_labels)
        stacking_predictions = stacking_ensemble.predict(test_features)
        
        assert len(stacking_predictions) == len(test_labels)
    
    def test_model_persistence(self, sample_data, model_manager):
        """Test model saving and loading."""
        texts, labels = sample_data
        
        # Train a model
        model_manager.train_all_models(texts, labels, feature_types=['tfidf'])
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save models
            model_manager.save_all_models(temp_dir)
            
            # Check that files were created
            temp_path = Path(temp_dir)
            model_files = list(temp_path.glob("*.pkl"))
            assert len(model_files) > 0
            
            # Create new manager and load models
            new_manager = ModelManager()
            new_manager.load_all_models(temp_dir)
            
            # Check that models were loaded
            assert len(new_manager.models) > 0
            
            # Test predictions with loaded models
            for model_name in list(new_manager.models.keys())[:2]:
                predictions = new_manager.predict_with_model(model_name, texts)
                assert len(predictions) == len(texts)
    
    def test_error_handling(self, sample_data):
        """Test error handling throughout the pipeline."""
        texts, labels = sample_data
        
        # Test invalid inputs
        preprocessor = TextPreprocessor()
        
        # Empty text
        with pytest.raises(Exception):
            preprocessor.preprocess("")
        
        # None input
        with pytest.raises(Exception):
            preprocessor.preprocess(None)
        
        # Test model errors
        from src.models.traditional_models import LogisticRegressionModel
        model = LogisticRegressionModel()
        
        # Predict without training
        with pytest.raises(Exception):
            model.predict(["test text"])
        
        # Train with mismatched data
        with pytest.raises(Exception):
            model.fit(["text1"], np.array([1, 2]))  # Mismatched lengths
    
    def test_performance_metrics(self, sample_data):
        """Test performance metrics calculation."""
        texts, labels = sample_data
        
        # Initialize validator
        validator = ModelValidator()
        
        # Create a simple model
        from src.models.traditional_models import LogisticRegressionModel
        model = LogisticRegressionModel({'max_iter': 100})
        
        # Train and predict
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor(method='tfidf', max_features=50)
        
        features = feature_extractor.fit_transform(preprocessor.preprocess_batch(texts))
        model.fit(features, labels)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Validate performance
        validation_results = validator.validate_model_performance(
            model=model,
            X_test=texts,
            y_test=labels,
            detailed=True
        )
        
        assert validation_results['model_name'] == 'LogisticRegression'
        assert 'basic_metrics' in validation_results
        assert 'classification_report' in validation_results
        assert 'confusion_matrix' in validation_results
        assert 'error_analysis' in validation_results
        assert 'confidence_analysis' in validation_results
        assert 'class_wise_metrics' in validation_results
        
        # Check basic metrics
        basic_metrics = validation_results['basic_metrics']
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            assert metric in basic_metrics
            assert 0 <= basic_metrics[metric] <= 1


class TestAPIIntegration:
    """Test integration with the API service."""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client."""
        try:
            from fastapi.testclient import TestClient
            from api.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("API not available")
    
    def test_api_full_workflow(self, api_client, sample_data):
        """Test full workflow through API."""
        texts, labels = sample_data
        
        # Test health check
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Test model listing
        response = api_client.get("/api/v1/models")
        assert response.status_code == 200
        
        # Test prediction
        payload = {
            "text": texts[0],
            "return_probabilities": True
        }
        
        response = api_client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 1
        assert data["probabilities"] is not None
        
        # Test batch prediction
        batch_payload = {
            "texts": texts[:3],
            "return_probabilities": True
        }
        
        response = api_client.post("/api/v1/predict/batch", json=batch_payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 3
        assert len(data["probabilities"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
