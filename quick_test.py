#!/usr/bin/env python3
"""
Quick test script to see immediate outputs from our production methods
"""

import sys
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_preprocessing():
    """Test text preprocessing with visible output."""
    print("=" * 50)
    print("TEXT PREPROCESSING TEST")
    print("=" * 50)
    
    try:
        from src.preprocessing.text_processor import TextPreprocessor
        
        # Sample texts
        texts = [
            "I LOVE this product!!! It's AMAZING :)",
            "This is terrible... I hate it so much!!! https://spam.com",
            "The service was okay, nothing special but not bad either.",
            "@user mention #hashtag check this out!"
        ]
        
        preprocessor = TextPreprocessor()
        
        print("Original vs Processed texts:")
        for i, text in enumerate(texts, 1):
            processed = preprocessor.preprocess(text)
            print(f"\n{i}. Original: {text}")
            print(f"   Processed: {processed}")
            
            # Show stats
            stats = preprocessor.get_preprocessing_stats(text, processed)
            print(f"   Stats: {stats['original_length']} -> {stats['processed_length']} chars "
                  f"(reduction: {stats['reduction_ratio']:.1%})")
        
        print("\n" + "=" * 50)
        print("Preprocessing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in preprocessing test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test feature extraction with visible output."""
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION TEST")
    print("=" * 50)
    
    try:
        from src.preprocessing.text_processor import TextPreprocessor
        from src.preprocessing.feature_extractor import FeatureExtractor
        
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
        
        print("Processed texts:")
        for i, text in enumerate(processed_texts, 1):
            print(f"{i}. {text}")
        
        # Test Bag of Words
        print("\n--- Bag of Words ---")
        bow_extractor = FeatureExtractor(method='bow', max_features=10)
        bow_features = bow_extractor.fit_transform(processed_texts)
        
        print(f"Feature matrix shape: {bow_features.shape}")
        print("Feature names:", bow_extractor.get_feature_names())
        
        # Test TF-IDF
        print("\n--- TF-IDF ---")
        tfidf_extractor = FeatureExtractor(method='tfidf', max_features=10)
        tfidf_features = tfidf_extractor.fit_transform(processed_texts)
        
        print(f"Feature matrix shape: {tfidf_features.shape}")
        print("Top features:")
        top_features = tfidf_extractor.get_top_features(n=5)
        for feature, score in top_features['top_features']:
            print(f"  {feature}: {score:.4f}")
        
        print("\n" + "=" * 50)
        print("Feature extraction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in feature extraction test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_traditional_models():
    """Test traditional ML models with visible output."""
    print("\n" + "=" * 50)
    print("TRADITIONAL MODELS TEST")
    print("=" * 50)
    
    try:
        from src.preprocessing.text_processor import TextPreprocessor
        from src.preprocessing.feature_extractor import FeatureExtractor
        from src.models.traditional_models import LogisticRegressionModel, NaiveBayesModel
        
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
        
        print(f"Dataset: {len(texts)} texts")
        print(f"Classes: {np.unique(labels)}")
        
        # Preprocess and extract features
        preprocessor = TextPreprocessor()
        processed_texts = preprocessor.preprocess_batch(texts)
        
        feature_extractor = FeatureExtractor(method='tfidf', max_features=50)
        features = feature_extractor.fit_transform(processed_texts)
        
        print(f"Feature matrix shape: {features.shape}")
        
        # Split data
        split_idx = len(texts) // 2
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        # Test Logistic Regression
        print("\n--- Logistic Regression ---")
        lr_model = LogisticRegressionModel({'max_iter': 1000})
        lr_model.fit(X_train, y_train)
        
        lr_pred = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test)
        
        from sklearn.metrics import accuracy_score, f1_score
        lr_accuracy = accuracy_score(y_test, lr_pred)
        lr_f1 = f1_score(y_test, lr_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {lr_accuracy:.4f}")
        print(f"F1-Score: {lr_f1:.4f}")
        print(f"Predictions: {lr_pred}")
        print(f"True labels: {y_test}")
        print(f"Probabilities (first 2):")
        for i in range(min(2, len(lr_proba))):
            print(f"  Sample {i+1}: {lr_proba[i]}")
        
        # Test Naive Bayes
        print("\n--- Naive Bayes ---")
        nb_model = NaiveBayesModel()
        nb_model.fit(X_train, y_train)
        
        nb_pred = nb_model.predict(X_test)
        nb_proba = nb_model.predict_proba(X_test)
        
        nb_accuracy = accuracy_score(y_test, nb_pred)
        nb_f1 = f1_score(y_test, nb_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {nb_accuracy:.4f}")
        print(f"F1-Score: {nb_f1:.4f}")
        print(f"Predictions: {nb_pred}")
        print(f"True labels: {y_test}")
        
        print("\n" + "=" * 50)
        print("Traditional models test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in traditional models test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble():
    """Test ensemble methods with visible output."""
    print("\n" + "=" * 50)
    print("ENSEMBLE METHODS TEST")
    print("=" * 50)
    
    try:
        from src.preprocessing.text_processor import TextPreprocessor
        from src.preprocessing.feature_extractor import FeatureExtractor
        from src.models.traditional_models import LogisticRegressionModel, NaiveBayesModel
        from src.ensemble.voting_ensemble import VotingEnsemble
        
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
        
        # Test Voting Ensemble
        print("\n--- Voting Ensemble (Soft Voting) ---")
        voting_ensemble = VotingEnsemble(
            name="TestVoting",
            models=base_models,
            voting="soft"
        )
        
        voting_ensemble.fit(X_train, y_train)
        voting_pred = voting_ensemble.predict(X_test)
        voting_proba = voting_ensemble.predict_proba(X_test)
        
        from sklearn.metrics import accuracy_score, f1_score
        voting_accuracy = accuracy_score(y_test, voting_pred)
        voting_f1 = f1_score(y_test, voting_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {voting_accuracy:.4f}")
        print(f"F1-Score: {voting_f1:.4f}")
        print(f"Predictions: {voting_pred}")
        print(f"True labels: {y_test}")
        
        # Show voting statistics
        voting_stats = voting_ensemble.get_voting_stats(X_test)
        print(f"Agreement rate: {voting_stats['agreement_rate']:.2%}")
        print(f"Disagreement cases: {voting_stats['total_disagreements']}")
        
        print("\n" + "=" * 50)
        print("Ensemble methods test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in ensemble test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all quick tests."""
    print("ADVANCED NLP SENTIMENT ANALYSIS - QUICK OUTPUT TEST")
    print("=" * 50)
    print("This script shows immediate outputs from our production methods")
    print("just like a Jupyter notebook!")
    print("=" * 50)
    
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Feature Extraction", test_feature_extraction),
        ("Traditional Models", test_traditional_models),
        ("Ensemble Methods", test_ensemble)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Failed to run {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Your production system is working correctly!")
    else:
        print("\nSome tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
