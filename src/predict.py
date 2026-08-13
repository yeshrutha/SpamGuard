import os
import sys
import joblib
import pandas as pd

# Add the current directory to sys.path to allow imports when run from different directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_and_preprocess_text, download_nltk_resources
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SpamDetector:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        self.model_path = os.path.join(models_dir, "best_model.pkl")
        
        # Load vectorizer and best model
        if not os.path.exists(self.vectorizer_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError("Trained model or vectorizer files not found. Please run train.py first.")
            
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)
        
        # Initialize NLTK components for preprocessing
        download_nltk_resources()
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            # Fallback if NLTK fails
            self.stop_words = set()
            class SimpleLemmatizer:
                def lemmatize(self, word):
                    return word
            self.lemmatizer = SimpleLemmatizer()

    def predict(self, raw_message: str):
        """
        Classifies a raw message as Spam or Ham, and returns the label and confidence.
        """
        if not raw_message or not raw_message.strip():
            return {
                "label": "HAM / NOT SPAM",
                "confidence": 1.0,
                "is_spam": False,
                "processed_message": ""
            }
            
        # 1. Clean and preprocess message
        cleaned_text = clean_and_preprocess_text(raw_message, self.stop_words, self.lemmatizer)
        
        # 2. Extract features using saved vectorizer
        features = self.vectorizer.transform([cleaned_text])
        
        # 3. Perform prediction
        pred = self.model.predict(features)[0]
        
        # 4. Get confidence / probability
        try:
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[pred]
        except (AttributeError, IndexError):
            # Fallback if predict_proba is not supported
            confidence = 1.0
            
        is_spam = bool(pred == 1)
        label = "SPAM" if is_spam else "HAM / NOT SPAM"
        
        return {
            "label": label,
            "confidence": float(confidence),
            "is_spam": is_spam,
            "processed_message": cleaned_text
        }

if __name__ == "__main__":
    # Quick CLI test
    detector = SpamDetector()
    
    test_messages = [
        "WINNER! You won a cash prize of £1000. Call 08000930705 now to claim your reward!",
        "Hey, are we still meeting for lunch today? Let me know.",
        "URGENT! Your mobile number has been selected for a £2000 bonus. Call now!"
    ]
    
    print("\n--- Running Quick Prediction Tests ---")
    for msg in test_messages:
        result = detector.predict(msg)
        print(f"\nOriginal message: '{msg}'")
        print(f"Processed text  : '{result['processed_message']}'")
        print(f"Prediction      : {result['label']} (Confidence: {result['confidence']*100:.2f}%)")
