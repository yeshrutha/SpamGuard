import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_and_evaluate_models():
    # Setup paths
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    static_dir = "static"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # Load preprocessed dataset
    cleaned_file_path = os.path.join(processed_dir, "cleaned_spam_data.csv")
    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_file_path}. Please run preprocess.py first.")
        
    df = pd.read_csv(cleaned_file_path)
    
    # Handle any nulls that might have occurred during cleaning (e.g., messages containing only stopwords)
    df['processed_message'] = df['processed_message'].fillna("")
    
    # Encode labels: ham -> 0, spam -> 1
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['processed_message']
    y = df['label_encoded']
    
    # 1. Stratified Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n--- Train/Test Split Summary ---")
    print(f"Training samples: {len(X_train)} (Spam: {sum(y_train)}, Ham: {len(y_train) - sum(y_train)})")
    print(f"Testing samples:  {len(X_test)} (Spam: {sum(y_test)}, Ham: {len(y_test) - sum(y_test)})")
    
    # 2. TF-IDF Vectorization
    print("\nExtracting features using TF-IDF...")
    # Use unigrams and bigrams, ignore terms that appear in less than 2 documents
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save the TF-IDF Vectorizer
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved TF-IDF Vectorizer to: {vectorizer_path}")
    
    # 3. Model Definitions
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42)
    }
    
    # Dictionaries to store evaluation metrics
    results = {}
    confusion_matrices = {}
    
    # 4. Train and Evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Model_Obj": model
        }
        confusion_matrices[model_name] = cm
        
        # Print metrics
        print(f"Evaluation results for {model_name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f} (Ability to avoid false alarms)")
        print(f"  Recall   : {rec:.4f} (Ability to find all spam)")
        print(f"  F1-Score : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        # Save model file
        model_filename = model_name.lower().replace(" ", "_") + ".pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model to: {model_path}")
        
    # 5. Compare models and select best model
    print("\n--- Model Comparison ---")
    comparison_df = pd.DataFrame(results).T.drop(columns=["Model_Obj"])
    print(comparison_df.to_string())
    
    # Choose best model based on F1-Score
    best_model_name = max(results, key=lambda k: results[k]["F1-Score"])
    best_model = results[best_model_name]["Model_Obj"]
    print(f"\nSelected Best Model: {best_model_name} (F1-Score: {results[best_model_name]['F1-Score']:.4f})")
    
    # Save best model copy
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    
    # Save a metadata file indicating which model is the best
    with open(os.path.join(models_dir, "best_model_info.txt"), "w") as f:
        f.write(f"Best Model Name: {best_model_name}\n")
        f.write(f"F1-Score: {results[best_model_name]['F1-Score']:.4f}\n")
        f.write(f"Precision: {results[best_model_name]['Precision']:.4f}\n")
        f.write(f"Recall: {results[best_model_name]['Recall']:.4f}\n")
        f.write(f"Accuracy: {results[best_model_name]['Accuracy']:.4f}\n")
        
    print(f"Saved Best Model alias to: {best_model_path}")
    
    # 6. Plot and Save Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, (model_name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Predicted Ham', 'Predicted Spam'],
                    yticklabels=['Actual Ham', 'Actual Spam'],
                    cbar=False, annot_kws={"size": 14, "weight": "bold"})
        axes[i].set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
        axes[i].set_xlabel('Predicted Label', fontsize=12)
        axes[i].set_ylabel('True Label', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("Saved confusion matrices plot to static/confusion_matrices.png")

if __name__ == "__main__":
    train_and_evaluate_models()
