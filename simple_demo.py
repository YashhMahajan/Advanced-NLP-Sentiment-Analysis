#!/usr/bin/env python3
"""
Simple demonstration of how to see outputs from our production methods
without requiring external dependencies.
"""

import re
import string

def simple_text_preprocessing_demo():
    """Show text preprocessing outputs."""
    print("=" * 60)
    print("1. TEXT PREPROCESSING OUTPUT DEMO")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "I LOVE this product!!! It's AMAZING :)",
        "This is terrible... I hate it so much!!! https://spam.com",
        "The service was okay, nothing special but not bad either.",
        "@user mention #hashtag check this out!"
    ]
    
    print("Original texts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "-" * 40)
    print("Processed texts (showing our method output):")
    
    # Simple preprocessing (like our TextPreprocessor does)
    for i, text in enumerate(texts, 1):
        # Convert to lowercase
        processed = text.lower()
        
        # Remove URLs
        processed = re.sub(r'http\S+|www\S+', ' ', processed)
        
        # Remove user mentions and hashtags
        processed = re.sub(r'@\w+|#\w+', ' ', processed)
        
        # Remove special characters and punctuation
        processed = processed.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        print(f"{i}. {processed}")
        
        # Show statistics (like our get_preprocessing_stats method)
        original_len = len(text)
        processed_len = len(processed)
        reduction = (original_len - processed_len) / original_len if original_len > 0 else 0
        
        print(f"   Stats: {original_len} -> {processed_len} chars (reduction: {reduction:.1%})")
    
    print("\n")

def simple_feature_extraction_demo():
    """Show feature extraction outputs."""
    print("=" * 60)
    print("2. FEATURE EXTRACTION OUTPUT DEMO")
    print("=" * 60)
    
    # Sample processed texts
    texts = [
        "love amazing product",
        "terrible awful hate",
        "service okay nothing special",
        "excellent customer service",
        "poor quality bad experience"
    ]
    
    print("Processed texts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "-" * 40)
    print("Bag of Words features (like our FeatureExtractor):")
    
    # Simple vocabulary creation (like BoW)
    all_words = []
    for text in texts:
        words = text.split()
        all_words.extend(words)
    
    # Get most common words
    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and get top 10
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    vocabulary = [word for word, count in sorted_words]
    
    print(f"Vocabulary (top {len(vocabulary)} words): {vocabulary}")
    
    print("\nFeature matrix (like our BoW output):")
    for i, text in enumerate(texts):
        words = text.split()
        features = [words.count(word) for word in vocabulary]
        print(f"Text {i+1}: {features}")
    
    print("\n" + "-" * 40)
    print("TF-IDF style features (simplified):")
    
    # Simple TF-IDF calculation
    total_docs = len(texts)
    for i, text in enumerate(texts):
        words = text.split()
        tfidf_features = []
        
        for word in vocabulary:
            # Term frequency
            tf = words.count(word) / len(words) if words else 0
            
            # Document frequency
            doc_count = sum(1 for t in texts if word in t.split())
            
            # Inverse document frequency (simplified)
            idf = 1 if doc_count > 0 else 0
            
            tfidf_features.append(tf * idf)
        
        print(f"Text {i+1}: {[f'{x:.3f}' for x in tfidf_features]}")
    
    print("\n")

def simple_model_prediction_demo():
    """Show model prediction outputs."""
    print("=" * 60)
    print("3. MODEL PREDICTION OUTPUT DEMO")
    print("=" * 60)
    
    # Simulate model predictions (like our models do)
    test_texts = [
        "I love this amazing product!",
        "This is terrible and awful.",
        "The service was okay."
    ]
    
    print("Test texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "-" * 40)
    print("Model predictions (like our predict method):")
    
    # Simulate predictions from different models
    models = {
        "Logistic Regression": [1, -1, 0],
        "Naive Bayes": [1, -1, 0], 
        "Decision Tree": [1, -1, 1]
    }
    
    class_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    
    for model_name, predictions in models.items():
        print(f"\n{model_name}:")
        print(f"  Predictions: {predictions}")
        print(f"  Labels: {[class_names[pred] for pred in predictions]}")
        
        # Simulate probabilities (like our predict_proba method)
        probabilities = [
            [0.1, 0.2, 0.7],  # Text 1: Negative, Neutral, Positive
            [0.8, 0.1, 0.1],  # Text 2: Negative, Neutral, Positive  
            [0.2, 0.6, 0.2]   # Text 3: Negative, Neutral, Positive
        ]
        
        print(f"  Probabilities:")
        for i, proba in enumerate(probabilities):
            # Map indices 0,1,2 to -1,0,1 then to class names
            class_names_full = {-1: "Negative", 0: "Neutral", 1: "Positive"}
            prob_dict = {class_names_full[j-1]: proba[j] for j in range(3)}
            print(f"    Text {i+1}: {prob_dict}")
    
    print("\n" + "-" * 40)
    print("Performance metrics (like our evaluate method):")
    
    # Simulate true labels and calculate metrics
    true_labels = [1, -1, 0]  # Actual sentiments
    
    for model_name, predictions in models.items():
        # Calculate accuracy (simplified)
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(predictions)
        
        print(f"{model_name}:")
        print(f"  True labels:     {true_labels}")
        print(f"  Predicted labels: {predictions}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Correct: {correct}/{len(predictions)}")
    
    print("\n")

def simple_ensemble_demo():
    """Show ensemble method outputs."""
    print("=" * 60)
    print("4. ENSEMBLE METHODS OUTPUT DEMO")
    print("=" * 60)
    
    # Simulate ensemble predictions
    test_texts = [
        "I love this amazing product!",
        "This is terrible and awful.",
        "The service was okay."
    ]
    
    print("Test texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "-" * 40)
    
    # Base model predictions
    base_predictions = {
        "Logistic Regression": [1, -1, 0],
        "Naive Bayes": [1, -1, 1],
        "Decision Tree": [1, -1, 0]
    }
    
    class_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    
    print("Base model predictions:")
    for model_name, preds in base_predictions.items():
        print(f"{model_name}: {[class_names[p] for p in preds]}")
    
    print("\n" + "-" * 40)
    print("Voting Ensemble (like our VotingEnsemble):")
    
    # Soft voting (average probabilities)
    base_probabilities = {
        "Logistic Regression": [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]],
        "Naive Bayes": [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.1, 0.7, 0.2]],
        "Decision Tree": [[0.2, 0.2, 0.6], [0.9, 0.05, 0.05], [0.3, 0.4, 0.3]]
    }
    
    # Calculate average probabilities (soft voting)
    avg_probabilities = []
    for i in range(len(test_texts)):
        avg_proba = []
        for j in range(3):  # 3 classes
            avg = sum(base_probabilities[model][i][j] for model in base_probabilities) / len(base_probabilities)
            avg_proba.append(avg)
        avg_probabilities.append(avg_proba)
    
    # Get final predictions (manual argmax)
    final_predictions = []
    for proba in avg_probabilities:
        max_idx = proba.index(max(proba))
        final_predictions.append(max_idx - 1)  # Convert to -1,0,1
    
    print("Average probabilities:")
    for i, proba in enumerate(avg_probabilities):
        # Map indices 0,1,2 to -1,0,1 then to class names
        class_names_full = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        prob_dict = {class_names_full[j-1]: proba[j] for j in range(3)}
        print(f"  Text {i+1}: {prob_dict}")
    
    print(f"\nFinal predictions: {[class_names[p] for p in final_predictions]}")
    
    # Show voting statistics
    print("\nVoting statistics:")
    for i in range(len(test_texts)):
        individual_preds = [base_predictions[model][i] for model in base_predictions]
        unique_preds = list(set(individual_preds))
        agreement = len(unique_preds) == 1
        
        print(f"  Text {i+1}: {individual_preds} -> {'Agreement' if agreement else 'Disagreement'}")
    
    print("\n")

def simple_cross_validation_demo():
    """Show cross-validation outputs."""
    print("=" * 60)
    print("5. CROSS-VALIDATION OUTPUT DEMO")
    print("=" * 60)
    
    # Simulate 3-fold cross-validation
    print("Dataset: 12 samples, 3-fold cross-validation")
    
    # Simulate fold results
    fold_results = [
        {"fold": 1, "accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85},
        {"fold": 2, "accuracy": 0.78, "precision": 0.80, "recall": 0.75, "f1": 0.77},
        {"fold": 3, "accuracy": 0.82, "precision": 0.85, "recall": 0.78, "f1": 0.81}
    ]
    
    print("\nFold-by-fold results (like our CrossValidator):")
    for result in fold_results:
        print(f"  Fold {result['fold']}:")
        print(f"    Accuracy: {result['accuracy']:.4f}")
        print(f"    Precision: {result['precision']:.4f}")
        print(f"    Recall: {result['recall']:.4f}")
        print(f"    F1-Score: {result['f1']:.4f}")
    
    # Calculate mean and std
    metrics = ["accuracy", "precision", "recall", "f1"]
    mean_scores = {}
    std_scores = {}
    
    for metric in metrics:
        scores = [result[metric] for result in fold_results]
        mean_scores[metric] = sum(scores) / len(scores)
        std_scores[metric] = (sum((x - mean_scores[metric])**2 for x in scores) / len(scores))**0.5
    
    print("\nCross-validation summary:")
    print(f"  Mean scores:   {mean_scores}")
    print(f"  Std scores:    {std_scores}")
    print(f"  Total time:    2.34 seconds")
    print(f"  Successful folds: 3/3")
    
    print("\n")

def simple_api_demo():
    """Show API usage examples."""
    print("=" * 60)
    print("6. API USAGE EXAMPLES")
    print("=" * 60)
    
    print("To see outputs from our API service:")
    
    print("\n1. Start the server:")
    print("   python -m uvicorn api.main:app --reload")
    print("   OR: docker-compose up")
    
    print("\n2. Test endpoints and see outputs:")
    
    print("\n   Health Check:")
    print("   curl http://localhost:8000/api/v1/health")
    print("   Expected output:")
    print('   {"status": "healthy", "version": "1.0.0", "models_loaded": 3, "uptime": 123.45}')
    
    print("\n   Single Prediction:")
    print('   curl -X POST "http://localhost:8000/api/v1/predict" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "I love this product!", "return_probabilities": true}\'')
    print("   Expected output:")
    print('   {"success": true, "predictions": ["Positive"], "probabilities": [{"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}], "model_used": "logistic_regression_bow", "processing_time": 0.123, "input_text_count": 1}')
    
    print("\n   Model Comparison:")
    print("   curl http://localhost:8000/api/v1/models/compare")
    print("   Expected output: Comparison table with all model performances")
    
    print("\n3. View interactive docs:")
    print("   Open http://localhost:8000/docs in browser")
    
    print("\n")

def main():
    """Run all simple demonstrations."""
    print("ADVANCED NLP SENTIMENT ANALYSIS - METHOD OUTPUT DEMONSTRATION")
    print("=" * 60)
    print("This shows you exactly what outputs our production methods generate")
    print("so you can see the results just like in your Jupyter notebook!")
    print("=" * 60)
    print()
    
    # Run all demonstrations
    simple_text_preprocessing_demo()
    simple_feature_extraction_demo()
    simple_model_prediction_demo()
    simple_ensemble_demo()
    simple_cross_validation_demo()
    simple_api_demo()
    
    print("=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("Now you know exactly what outputs to expect from each method!")
    print()
    print("To see the real outputs:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run: python demo.py")
    print("3. Start API: python -m uvicorn api.main:app --reload")
    print("4. Test with: curl http://localhost:8000/api/v1/predict")

if __name__ == "__main__":
    main()
