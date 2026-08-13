# SpamGuard AI: SMS Spam Detector Machine Learning Project

This project builds a professional-grade machine learning classifier that automatically distinguishes between **Spam** and **Ham (legitimate)** messages. Built as part of the **QSkill AI & ML internship**, it provides a complete pipeline from raw text ingestion to model training, evaluation, and a highly polished interactive prediction dashboard.

---

## 🎯 Project Objective
The goal is to develop a text classification system using standard natural language processing (NLP) and supervised machine learning techniques. Using the **SMS Spam Collection dataset**, we construct feature representations using Term Frequency-Inverse Document Frequency (TF-IDF) and evaluate multiple classification algorithms to determine the best model for production deployment.

---

## 📊 Dataset Overview
The project uses the publicly available **SMS Spam Collection dataset** from the UCI Machine Learning Repository.

### Dataset Profile:
* **Total Messages**: 5,572
* **Original Class Distribution**:
  * **Ham (Non-Spam)**: 4,825 (86.59%)
  * **Spam**: 747 (13.41%)
* **Missing Values**: 0
* **Duplicate Rows**: 403 (Dropped to prevent data leakage and metric inflation)
* **Post-Cleaning Size**: 5,169 unique messages
* **Cleaned Class Distribution**:
  * **Ham (Non-Spam)**: 4,516 (87.37%)
  * **Spam**: 653 (12.63%)

---

## ⚙️ Text Preprocessing Pipeline
Text classification requires transforming raw, unstructured text into clean tokens. The preprocessing pipeline consists of:

1. **Lowercasing**: Converting all text to lowercase to ensure consistency (e.g., "SPAM", "Spam", and "spam" map to the same token).
2. **Punctuation and Special Character Removal**: Using regular expressions to strip out symbols, currency characters, and punctuation that do not add semantic value.
3. **Tokenization**: Segmenting sentences into individual word units.
4. **Stopwords Removal**: Filtering out common English words (such as "the", "is", "at", "which") that appear frequently but carry no discriminative information.
5. **Lemmatization**: Reducing words to their dictionary base form (e.g., "running" $\rightarrow$ "run", "winners" $\rightarrow$ "winner") using the NLTK WordNet Lemmatizer to group word variations together.
6. **Feature Length Filtering**: Deleting any tokens shorter than 2 characters.

---

## 🧮 TF-IDF Vectorization
We use **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform tokenized messages into numerical vectors. 
* **Term Frequency ($TF$)**: Measures the frequency of a word in a specific document.
* **Inverse Document Frequency ($IDF$)**: Measures how common or rare a word is across all documents in the corpus. Words that appear in almost all documents (e.g., "message") are penalized, while rare words that carry high informative value (e.g., "claim", "free", "urgent") are weighted higher.

We fit a `TfidfVectorizer` on the training corpus with **unigrams and bigrams** (`ngram_range=(1, 2)`) and limit the feature space to the top **5,000 features** to maintain computational efficiency and prevent overfitting.

---

## 🤖 Models & Evaluation Results
The dataset was split using a **Stratified Train-Test Split** ($80\%$ training, $20\%$ testing) to preserve the class balance. We trained and evaluated two classical machine learning classifiers:

1. **Multinomial Naive Bayes (MNB)**
2. **Logistic Regression (LR)**

### Evaluation Metrics on Test Set (1,034 messages):

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Multinomial Naive Bayes** | **96.91%** | **100.00%** | **75.57%** | **86.09%** |
| **Logistic Regression** | 95.84% | 95.83% | 70.23% | 81.06% |

---

## 🏆 Model Comparison & Selection
* **Accuracy & F1-Score**: Multinomial Naive Bayes outperforms Logistic Regression across all metrics.
* **Precision is Critical**: In spam detection, a **False Positive** (classifying a legitimate email/message as spam) is highly damaging because the user might miss a critical personal message.
  * **Multinomial Naive Bayes** achieved **100.00% Precision** on the test set. Out of 131 test spam messages, it correctly flagged 99 of them, and misclassified **zero** ham messages as spam.
  * **Logistic Regression** achieved **95.83% Precision** (misclassifying 4 legitimate messages as spam).
* **Verdict**: Multinomial Naive Bayes was selected as the production model and serialized as `best_model.pkl`.

---

## 📁 Project Directory Structure
```text
SpamGuard/
├── data/
│   ├── raw/                  # Downloaded raw dataset
│   └── processed/            # Cleaned CSV file
├── models/                   # Serialized ML model and TF-IDF weights
│   ├── multinomial_naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── best_model.pkl        # Copy of Multinomial NB
│   └── best_model_info.txt   # Model parameters and scores
├── src/
│   ├── utils.py              # Download/extraction utils
│   ├── preprocess.py         # NLP cleaning and EDA scripts
│   ├── train.py              # Train/val and metrics generation
│   └── predict.py            # Local CLI testing/prediction library
├── static/                   # Output visualization png files
│   ├── class_distribution.png
│   ├── message_length_dist.png
│   └── confusion_matrices.png
├── app.py                    # Streamlit Web Dashboard
└── requirements.txt          # Package requirements
```

---

## 🚀 How to Run the Project

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### 1. Clone or Open Workspace
Navigate to your project root folder:
```bash
cd SpamGuard
```

### 2. Install Dependencies
Install all the required libraries:
```bash
python -m pip install -r requirements.txt
```

### 3. Run Preprocessing & Download Dataset
This downloads the official dataset, preprocesses it, and creates the EDA visualizations:
```bash
python src/preprocess.py
```

### 4. Train Models
This vectorizes the data, runs training, outputs comparison metrics, and saves the final models to the `models/` directory:
```bash
python src/train.py
```

### 5. Run the Local CLI Test
Test the prediction module from your console using:
```bash
python src/predict.py
```

### 6. Start the Web App
Run the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
This will automatically open the web application in your default browser at `http://localhost:8501`.

---

## 💡 Example Predictions

* **Ham Message**:
  > *"Hey, are we still meeting for lunch today? Let me know."*
  * **Verdict**: `HAM / NOT SPAM` (Confidence: **99.73%**)

* **Spam Message**:
  > *"WINNER! You won a cash prize of £1000. Call 08000930705 now to claim your reward!"*
  * **Verdict**: `SPAM` (Confidence: **98.38%**)
