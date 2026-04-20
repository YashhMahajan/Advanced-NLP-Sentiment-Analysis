# Advanced NLP Sentiment Analysis - Production Grade

A comprehensive, production-grade sentiment analysis system built with modern ML engineering practices. This project demonstrates advanced NLP capabilities with multiple models, ensemble methods, and a scalable API service.

## Features

### Core Capabilities
- **Multiple ML Models**: Logistic Regression, Naive Bayes, Decision Tree, and BERT transformer
- **Advanced Ensemble Methods**: Voting and Stacking ensembles for improved performance
- **Comprehensive Cross-Validation**: Robust model evaluation with multiple strategies
- **Production API**: FastAPI service with health monitoring and error handling
- **Containerization**: Docker support with multi-stage builds
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment

### Technical Highlights
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Error Handling**: Comprehensive exception management and validation
- **Logging & Monitoring**: Structured logging with performance metrics
- **Model Management**: Model versioning, persistence, and comparison
- **Scalable Design**: Ready for production deployment and scaling

## Project Structure

```
Advanced NLP Sentiment Analysis/
|-- src/                          # Core Python package
|   |-- __init__.py
|   |-- models/                   # ML models
|   |   |-- __init__.py
|   |   |-- base_model.py         # Abstract base class
|   |   |-- traditional_models.py # Logistic, NB, Decision Tree
|   |   |-- bert_model.py         # BERT transformer
|   |   |-- model_manager.py      # Model orchestration
|   |-- preprocessing/            # Text processing
|   |   |-- __init__.py
|   |   |-- text_processor.py     # Text cleaning & preprocessing
|   |   |-- feature_extractor.py  # BoW, TF-IDF features
|   |-- ensemble/                 # Ensemble methods
|   |   |-- __init__.py
|   |   |-- ensemble_model.py     # Base ensemble class
|   |   |-- voting_ensemble.py     # Voting ensemble
|   |   |-- stacking_ensemble.py  # Stacking ensemble
|   |-- validation/               # Model validation
|   |   |-- __init__.py
|   |   |-- cross_validation.py   # CV framework
|   |   |-- model_validator.py    # Validation utilities
|   |-- utils/                    # Utilities
|   |   |-- __init__.py
|   |   |-- exceptions.py         # Custom exceptions
|   |   |-- logger.py             # Logging setup
|   |   |-- validators.py         # Input validation
|-- api/                          # FastAPI service
|   |-- __init__.py
|   |-- main.py                   # FastAPI app
|   |-- routes.py                 # API endpoints
|   |-- schemas.py                # Pydantic models
|-- sentiment_analysis_clean.ipynb # Original notebook (kept for learning)
|-- requirements.txt              # Python dependencies
|-- Dockerfile                    # Container configuration
|-- docker-compose.yml            # Multi-container setup
|-- .dockerignore                 # Docker ignore file
|-- README.md                     # This file
```

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (for containerized deployment)

### Local Development

1. **Clone and Setup**
```bash
git clone <repository-url>
cd "Advanced NLP Sentiment Analysis"
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the API**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

1. **Development Environment**
```bash
docker-compose --profile dev up
```

2. **Production Environment**
```bash
docker-compose up -d
```

3. **With Redis Cache**
```bash
docker-compose --profile cache up
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### List Available Models
```bash
curl http://localhost:8000/api/v1/models
```

### Single Text Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
-H "Content-Type: application/json" \
-d '{
  "text": "I love this product! It is amazing.",
  "return_probabilities": true
}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
-H "Content-Type: application/json" \
-d '{
  "texts": [
    "This is terrible!",
    "The service was okay.",
    "Excellent experience!"
  ],
  "return_probabilities": true
}'
```

### Model Comparison
```bash
curl http://localhost:8000/api/v1/models/compare
```

## Model Performance

Based on the Twitter sentiment dataset (162,969 samples):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (BoW) | 0.877 | 0.878 | 0.877 | 0.876 |
| Logistic Regression (TF-IDF) | 0.873 | 0.875 | 0.873 | 0.872 |
| Decision Tree (BoW) | 0.817 | 0.816 | 0.817 | 0.816 |
| Naive Bayes (BoW) | 0.755 | 0.760 | 0.755 | 0.757 |
| BERT (Base) | ~0.90* | ~0.91* | ~0.90* | ~0.90* |

*BERT performance based on typical results; actual performance may vary.

## Advanced Features

### Ensemble Methods

The system supports both voting and stacking ensembles:

```python
from src.ensemble import VotingEnsemble, StackingEnsemble
from src.models import LogisticRegressionModel, NaiveBayesModel

# Create voting ensemble
voting_ensemble = VotingEnsemble(
    name="SentimentVoting",
    models=[log_reg_model, nb_model],
    voting="soft"
)

# Create stacking ensemble
stacking_ensemble = StackingEnsemble(
    name="SentimentStacking",
    models=[log_reg_model, nb_model],
    meta_learner="logistic_regression"
)
```

### Cross-Validation

Comprehensive cross-validation framework:

```python
from src.validation import CrossValidator

cv = CrossValidator(
    cv_strategy="stratified_kfold",
    cv_folds=5
)

results = cv.cross_validate_model(model, X, y)
```

### Model Management

Advanced model management capabilities:

```python
from src.models import ModelManager

manager = ModelManager()
manager.register_default_models()
manager.train_all_models(X_train, y_train)
results = manager.evaluate_all_models(X_test, y_test)

# Get best model
best_name, best_model = manager.get_best_model()
```

## Configuration

### Environment Variables
- `PYTHONPATH`: Python path (default: `/app`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `MODEL_PATH`: Path to saved models (default: `./models`)

### Model Configuration
Models can be configured via dictionaries:

```python
config = {
    'model_type': 'logistic_regression',
    'random_state': 42,
    'max_iter': 1000,
    'C': 1.0
}
```

## Monitoring & Logging

The system includes comprehensive logging:
- Structured logging with timestamps and log levels
- Performance metrics tracking
- Error reporting with stack traces
- API request/response logging

Access logs at:
- Console output (development)
- `/app/logs` directory (production)

## Testing

### Unit Tests
```bash
pytest tests/
```

### API Tests
```bash
pytest tests/api/
```

### Integration Tests
```bash
pytest tests/integration/
```

## Deployment

### Production Deployment

1. **Build and Deploy**
```bash
docker-compose -f docker-compose.yml --profile production up -d
```

2. **Scale Services**
```bash
docker-compose up -d --scale api=4
```

3. **Health Monitoring**
```bash
curl http://localhost:8000/api/v1/health
```

### Environment Setup

- **Development**: Use `docker-compose --profile dev`
- **Staging**: Use `docker-compose` with Redis cache
- **Production**: Use full stack with Nginx reverse proxy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original dataset from Kaggle Twitter Sentiment Analysis
- NLTK for text preprocessing utilities
- Hugging Face Transformers for BERT implementation
- Scikit-learn for traditional ML models
- FastAPI for the web service framework

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the health check endpoint
3. Check logs for detailed error information
4. Create an issue in the repository

---

**Note**: The original Jupyter notebook (`sentiment_analysis_clean.ipynb`) is preserved for educational purposes and reference.
