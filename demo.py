#!/usr/bin/env python3
"""
Demonstration script for Advanced NLP Sentiment Analysis System

This script shows how to use all the components and see their outputs,
similar to the Jupyter notebook experience but with the production code.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_preprocessing():
    """Demonstrate text preprocessing with output."""
    print("=" * 60)
    print("1. TEXT PREPROCESSING DEMO")
    print("=" * 60)
    
    from preprocessing import TextPreprocessor
    
    # Sample texts
    sample_texts = [
        "I LOVE this product!!! It's AMAZING :)",
        "This is terrible... I hate it so much!!! https://spam.com",
        "The service was okay, nothing special but not bad either.",
        "@user mention #hashtag check this out!"
    ]
    
    print("Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "-" * 40)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    print("Preprocessed texts:")
    for i, text in enumerate(sample_texts, 1):
        processed = preprocessor.preprocess(text)
        print(f"{i}. {processed}")
        
        # Show preprocessing stats
        stats = preprocessor.get_preprocessing_stats(text, processed)
        print(f"   Stats: {stats['original_length']} -> {stats['processed_length']} chars, "
              f"reduction: {stats['reduction_ratio']:.1%}")
    
    print("\n")


def demo_feature_extraction():
    """Demonstrate feature extraction with output."""
    print("=" * 60)
    print("2. FEATURE EXTRACTION DEMO")
    print("=" * 60)
    
    from preprocessing import TextPreprocessor, FeatureExtractor
    
    # Sample texts
    texts = [
        "I love this amazing product",
        "This is terrible and awful",
        "The service was okay",
        "Excellent customer service",
        "Poor quality and bad experience"
    ]
    
    # Preprocess
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts)
    
    print("Processed texts for feature extraction:")
    for i, text in enumerate(processed_texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "-" * 40)
    
    # Bag of Words
    print("Bag of Words Features:")
    bow_extractor = FeatureExtractor(method='bow', max_features=10)
    bow_features = bow_extractor.fit_transform(processed_texts)
    
    print(f"Feature matrix shape: {bow_features.shape}")
    print("Feature names (first 10):", bow_extractor.get_feature_names()[:10])
    print("Sample feature vectors:")
    for i, features in enumerate(bow_features[:3]):  # Show first 3
        if hasattr(features, 'toarray'):
            features = features.toarray().flatten()
        non_zero_indices = features.nonzero()[0] if hasattr(features, 'nonzero') else np.where(features > 0)[0]
        feature_names = bow_extractor.get_feature_names()
        print(f"  Text {i+1}: {[(feature_names[idx], int(features[idx])) for idx in non_zero_indices[:5]]}")
    
    print("\n" + "-" * 40)
    
    # TF-IDF
    print("TF-IDF Features:")
    tfidf_extractor = FeatureExtractor(method='tfidf', max_features=10)
    tfidf_features = tfidf_extractor.fit_transform(processed_texts)
    
    print(f"Feature matrix shape: {tfidf_features.shape}")
    print("Top TF-IDF features:")
    top_features = tfidf_extractor.get_top_features(n=5)
    print(top_features)
    
    print("\n")


def demo_traditional_models():
    """Demonstrate traditional ML models with output."""
    print("=" * 60)
    print("3. TRADITIONAL ML MODELS DEMO")
    print("=" * 60)
    
    from preprocessing import TextPreprocessor, FeatureExtractor
    from models.traditional_models import LogisticRegressionModel, NaiveBayesModel, DecisionTreeModel
    
    # Sample dataset
    texts = [
        "I love this amazing product! Highly recommended.",
        "This is terrible. I hate it and want my money back.",
        "The service was okay, nothing special but not bad.",
        "Excellent quality and fast shipping!",
        "Poor customer service, very disappointed.",
        "Average product, does what it says.",
        "Outstanding experience, will buy again!",
        "Not worth the price, better alternatives exist.",
        "Good value for money, satisfied with purchase.",
        "Complete waste of time, avoid this product."
    ]
    
    labels = np.array([1, -1, 0, 1, -1, 0, 1, -1, 1, -1])  # 1: Positive, 0: Neutral, -1: Negative
    
    print(f"Dataset: {len(texts)} texts, {len(np.unique(labels))} classes")
    print(f"Class distribution: {np.bincount(labels + 1)}")  # Shift to 0-based for bincount
    
    # Preprocess and extract features
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts)
    
    feature_extractor = FeatureExtractor(method='tfidf', max_features=100)
    features = feature_extractor.fit_transform(processed_texts)
    
    print(f"Feature matrix shape: {features.shape}")
    
    # Split data
    split_idx = len(texts) // 2
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    print("\n" + "-" * 40)
    
    # Test different models
    models = {
        "Logistic Regression": LogisticRegressionModel({'max_iter': 1000}),
        "Naive Bayes": NaiveBayesModel(),
        "Decision Tree": DecisionTreeModel({'max_depth': 5})
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}:")
        
        # Train
        model.fit(X_train, y_train)
        print(f"  Model trained successfully")
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Show sample predictions
        print(f"  Sample predictions: {y_pred[:3]}")
        print(f"  True labels:      {y_test[:3]}")
        
        # Show probabilities for first sample
        print(f"  Sample probabilities: {y_proba[0]}")
    
    print("\n" + "-" * 40)
    print("Model Comparison:")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))
    
    print("\n")


def demo_ensemble_methods():
    """Demonstrate ensemble methods with output."""
    print("=" * 60)
    print("4. ENSEMBLE METHODS DEMO")
    print("=" * 60)
    
    from preprocessing import TextPreprocessor, FeatureExtractor
    from models.traditional_models import LogisticRegressionModel, NaiveBayesModel
    from ensemble import VotingEnsemble, StackingEnsemble
    
    # Sample dataset
    texts = [
        "I love this amazing product! Highly recommended.",
        "This is terrible. I hate it and want my money back.",
        "The service was okay, nothing special but not bad.",
        "Excellent quality and fast shipping!",
        "Poor customer service, very disappointed.",
        "Average product, does what it says.",
        "Outstanding experience, will buy again!",
        "Not worth the price, better alternatives exist.",
        "Good value for money, satisfied with purchase.",
        "Complete waste of time, avoid this product."
    ]
    
    labels = np.array([1, -1, 0, 1, -1, 0, 1, -1, 1, -1])
    
    # Preprocess and extract features
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts)
    
    feature_extractor = FeatureExtractor(method='tfidf', max_features=50)
    features = feature_extractor.fit_transform(processed_texts)
    
    # Split data
    split_idx = len(texts) // 2
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    # Create base models
    base_models = [
        LogisticRegressionModel({'max_iter': 1000}),
        NaiveBayesModel()
    ]
    
    print(f"Base models: {[model.model_name for model in base_models]}")
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    print("\n" + "-" * 40)
    
    # Voting Ensemble
    print("Voting Ensemble (Soft Voting):")
    voting_ensemble = VotingEnsemble(
        name="DemoVoting",
        models=base_models,
        voting="soft"
    )
    
    voting_ensemble.fit(X_train, y_train)
    voting_predictions = voting_ensemble.predict(X_test)
    voting_probabilities = voting_ensemble.predict_proba(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    voting_accuracy = accuracy_score(y_test, voting_predictions)
    voting_f1 = f1_score(y_test, voting_predictions, average='weighted', zero_division=0)
    
    print(f"  Accuracy: {voting_accuracy:.4f}")
    print(f"  F1-Score: {voting_f1:.4f}")
    print(f"  Predictions: {voting_predictions}")
    print(f"  Probabilities (first sample): {voting_probabilities[0]}")
    
    # Show voting statistics
    voting_stats = voting_ensemble.get_voting_stats(X_test)
    print(f"  Agreement rate: {voting_stats['agreement_rate']:.2%}")
    
    print("\n" + "-" * 40)
    
    # Stacking Ensemble
    print("Stacking Ensemble:")
    stacking_ensemble = StackingEnsemble(
        name="DemoStacking",
        models=base_models,
        cv_folds=2  # Use fewer folds for small dataset
    )
    
    stacking_ensemble.fit(X_train, y_train)
    stacking_predictions = stacking_ensemble.predict(X_test)
    stacking_probabilities = stacking_ensemble.predict_proba(X_test)
    
    stacking_accuracy = accuracy_score(y_test, stacking_predictions)
    stacking_f1 = f1_score(y_test, stacking_predictions, average='weighted', zero_division=0)
    
    print(f"  Accuracy: {stacking_accuracy:.4f}")
    print(f"  F1-Score: {stacking_f1:.4f}")
    print(f"  Predictions: {stacking_predictions}")
    print(f"  Probabilities (first sample): {stacking_probabilities[0]}")
    
    # Show stacking info
    stacking_info = stacking_ensemble.get_stacking_info()
    print(f"  Meta-learner: {stacking_info['meta_learner_type']}")
    print(f"  CV folds used: {stacking_info['cv_folds']}")
    
    print("\n")


def demo_cross_validation():
    """Demonstrate cross-validation with output."""
    print("=" * 60)
    print("5. CROSS-VALIDATION DEMO")
    print("=" * 60)
    
    from preprocessing import TextPreprocessor, FeatureExtractor
    from validation import CrossValidator
    from models.traditional_models import LogisticRegressionModel
    
    # Sample dataset
    texts = [
        "I love this amazing product! Highly recommended.",
        "This is terrible. I hate it and want my money back.",
        "The service was okay, nothing special but not bad.",
        "Excellent quality and fast shipping!",
        "Poor customer service, very disappointed.",
        "Average product, does what it says.",
        "Outstanding experience, will buy again!",
        "Not worth the price, better alternatives exist.",
        "Good value for money, satisfied with purchase.",
        "Complete waste of time, avoid this product.",
        "Fantastic quality and great service!",
        "Mediocre experience, nothing special.",
        "Worst purchase ever, totally regret it.",
        "Pretty good, would recommend to friends.",
        "Terrible quality, broke after one use."
    ]
    
    labels = np.array([1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 1, 0, -1, 1, -1])
    
    print(f"Dataset: {len(texts)} texts, {len(np.unique(labels))} classes")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor(method='tfidf', max_features=50)
    model = LogisticRegressionModel({'max_iter': 1000})
    cross_validator = CrossValidator(cv_folds=3)  # Use 3 folds for small dataset
    
    print(f"Cross-validation strategy: {cross_validator.cv_strategy}")
    print(f"Number of folds: {cross_validator.cv_folds}")
    
    print("\n" + "-" * 40)
    
    # Perform cross-validation
    cv_results = cross_validator.cross_validate_model(
        model=model,
        X=texts,
        y=labels,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        return_predictions=True
    )
    
    # Display results
    print(f"Model: {cv_results['model_name']}")
    print(f"Successful folds: {cv_results['successful_folds']}")
    print(f"Total time: {cv_results['total_time']:.2f} seconds")
    
    print("\nCross-validation scores:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in cv_results['mean_scores']:
            mean_score = cv_results['mean_scores'][metric]
            std_score = cv_results['std_scores'][metric]
            print(f"  {metric.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    print("\nFold-by-fold results:")
    for fold_result in cv_results['fold_results']:
        fold_num = fold_result['fold']
        if 'f1' in fold_result:
            print(f"  Fold {fold_num}: F1 = {fold_result['f1']:.4f}")
        else:
            print(f"  Fold {fold_num}: Failed - {fold_result.get('error', 'Unknown error')}")
    
    print("\n")


def demo_model_manager():
    """Demonstrate model manager with output."""
    print("=" * 60)
    print("6. MODEL MANAGER DEMO")
    print("=" * 60)
    
    from models.model_manager import ModelManager
    
    # Sample dataset
    texts = [
        "I love this amazing product! Highly recommended.",
        "This is terrible. I hate it and want my money back.",
        "The service was okay, nothing special but not bad.",
        "Excellent quality and fast shipping!",
        "Poor customer service, very disappointed.",
        "Average product, does what it says.",
        "Outstanding experience, will buy again!",
        "Not worth the price, better alternatives exist.",
        "Good value for money, satisfied with purchase.",
        "Complete waste of time, avoid this product."
    ]
    
    labels = np.array([1, -1, 0, 1, -1, 0, 1, -1, 1, -1])
    
    print(f"Dataset: {len(texts)} texts")
    
    # Initialize model manager
    manager = ModelManager()
    
    print(f"Registered models: {list(manager.models.keys())}")
    
    print("\n" + "-" * 40)
    
    # Train models
    print("Training all models...")
    training_results = manager.train_all_models(texts, labels, feature_types=['tfidf'])
    
    successful_models = [
        name for name, result in training_results['training'].items()
        if result['status'] == 'success'
    ]
    
    print(f"Successfully trained: {successful_models}")
    failed_models = [
        name for name, result in training_results['training'].items()
        if result['status'] == 'failed'
    ]
    if failed_models:
        print(f"Failed to train: {failed_models}")
    
    print("\n" + "-" * 40)
    
    # Evaluate models
    if successful_models:
        print("Evaluating models...")
        
        # Split for evaluation
        split_idx = len(texts) // 2
        test_texts = texts[split_idx:]
        test_labels = labels[split_idx:]
        
        evaluation_results = manager.evaluate_all_models(test_texts, test_labels)
        
        print("Model performance:")
        for model_name, results in evaluation_results['evaluation'].items():
            if 'status' in results and results['status'] == 'failed':
                print(f"  {model_name}: Failed")
            else:
                print(f"  {model_name}: F1 = {results.get('f1_score', 0):.4f}")
        
        # Show comparison table
        print("\nComparison table:")
        comparison_df = manager.get_model_comparison_table()
        print(comparison_df.round(4))
        
        # Get best model
        if not comparison_df.empty:
            best_model_name, best_model = manager.get_best_model()
            print(f"\nBest model: {best_model_name}")
    
    print("\n")


def demo_api_usage():
    """Demonstrate API usage with output."""
    print("=" * 60)
    print("7. API USAGE DEMO")
    print("=" * 60)
    
    print("To test the API, run the following commands:")
    
    print("\n1. Start the API server:")
    print("   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("   OR")
    print("   docker-compose up")
    
    print("\n2. Test endpoints:")
    
    # Health check
    print("\n   Health Check:")
    print("   curl http://localhost:8000/api/v1/health")
    
    # List models
    print("\n   List Models:")
    print("   curl http://localhost:8000/api/v1/models")
    
    # Single prediction
    print("\n   Single Prediction:")
    print('   curl -X POST "http://localhost:8000/api/v1/predict" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "I love this product!", "return_probabilities": true}\'')
    
    # Batch prediction
    print("\n   Batch Prediction:")
    print('   curl -X POST "http://localhost:8000/api/v1/predict/batch" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"texts": ["Great product!", "Terrible service"], "return_probabilities": true}\'')
    
    # Model comparison
    print("\n   Model Comparison:")
    print("   curl http://localhost:8000/api/v1/models/compare")
    
    print("\n3. View API documentation:")
    print("   Open http://localhost:8000/docs in your browser")
    
    print("\n")


def main():
    """Run all demonstrations."""
    print("ADVANCED NLP SENTIMENT ANALYSIS - PRODUCTION SYSTEM DEMO")
    print("=" * 60)
    print("This script demonstrates all components of the production system")
    print("with detailed outputs, similar to a Jupyter notebook experience.")
    print("=" * 60)
    print()
    
    try:
        # Run all demos
        demo_preprocessing()
        demo_feature_extraction()
        demo_traditional_models()
        demo_ensemble_methods()
        demo_cross_validation()
        demo_model_manager()
        demo_api_usage()
        
        print("=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("All components demonstrated successfully!")
        print("You can now run the API server and test the endpoints.")
        print()
        print("Next steps:")
        print("1. Start the API: python -m uvicorn api.main:app --reload")
        print("2. Open http://localhost:8000/docs for API documentation")
        print("3. Test with curl or the provided examples")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
