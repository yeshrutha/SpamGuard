import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils import download_and_extract_dataset

# Ensure NLTK packages are downloaded
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {e}")

# Text cleaning function
def clean_and_preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove unnecessary punctuation and special characters
    # Keep alphanumeric characters and whitespace
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 3. Tokenize text
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback to simple split if tokenizer fails
        tokens = text.split()
        
    # 4. Remove stopwords and apply lemmatization
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
            
    return " ".join(cleaned_tokens)

def run_preprocessing_pipeline():
    # Setup paths
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    static_dir = "static"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # Download dataset
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    download_and_extract_dataset(dataset_url, raw_dir)
    
    # Load raw dataset
    raw_file = os.path.join(raw_dir, "SMSSpamCollection")
    # SMSSpamCollection is a tab-separated file without a header
    df = pd.read_csv(raw_file, sep='\t', names=['label', 'message'])
    
    print("\n--- Raw Data Inspection ---")
    print(f"Total number of messages: {len(df)}")
    print("\nClass distribution:")
    print(df['label'].value_counts())
    print("\nClass distribution (percentage):")
    print(df['label'].value_counts(normalize=True) * 100)
    print(f"\nMissing values in label: {df['label'].isnull().sum()}")
    print(f"Missing values in message: {df['message'].isnull().sum()}")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    duplicate_msg_count = df['message'].duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")
    print(f"Duplicate messages: {duplicate_msg_count}")
    
    # Calculate lengths before cleaning
    df['char_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
    
    # Generate EDA Visualizations BEFORE dropping duplicates for complete inspection,
    # or generate them after dropping. Let's do it on the unique messages to represent true distribution.
    
    # Handle duplicate records
    print("\nHandling duplicate records...")
    df_cleaned = df.drop_duplicates(subset=['message'], keep='first').copy()
    print(f"Dataset size after removing duplicate messages: {len(df_cleaned)}")
    print("\nClass distribution after cleaning:")
    print(df_cleaned['label'].value_counts())
    print(df_cleaned['label'].value_counts(normalize=True) * 100)
    
    # Perform Text Preprocessing
    print("\nPreprocessing text column...")
    download_nltk_resources()
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except Exception:
        # Fallback basic stop words list
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                      'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
                      'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                      'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                      'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                      'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                      'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                      'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
        class SimpleLemmatizer:
            def lemmatize(self, word):
                return word
        lemmatizer = SimpleLemmatizer()
        
    df_cleaned['processed_message'] = df_cleaned['message'].apply(
        lambda x: clean_and_preprocess_text(x, stop_words, lemmatizer)
    )
    
    # Recalculate lengths for cleaned messages
    df_cleaned['clean_char_length'] = df_cleaned['processed_message'].apply(len)
    df_cleaned['clean_word_count'] = df_cleaned['processed_message'].apply(lambda x: len(str(x).split()))
    
    # Save cleaned data
    cleaned_file_path = os.path.join(processed_dir, "cleaned_spam_data.csv")
    df_cleaned.to_csv(cleaned_file_path, index=False)
    print(f"Preprocessed dataset saved to: {cleaned_file_path}")
    
    # Generate Plots
    print("\nGenerating and saving EDA plots...")
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(7, 5))
    # Elegant dark/modern color palette (Steel Blue and Muted Coral)
    colors = ['#4682B4', '#CD5C5C']
    ax = sns.countplot(x='label', data=df_cleaned, palette=colors, hue='label', legend=False)
    plt.title('SMS Spam vs Ham Class Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Message Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}\n({height/len(df_cleaned)*100:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "class_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Message Length Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_cleaned[df_cleaned['label'] == 'ham'], x='char_length', 
                fill=True, label='Ham (Non-Spam)', color='#4682B4', alpha=0.6, linewidth=2)
    sns.kdeplot(data=df_cleaned[df_cleaned['label'] == 'spam'], x='char_length', 
                fill=True, label='Spam', color='#CD5C5C', alpha=0.6, linewidth=2)
    
    # Limit x-axis to zoom in on typical message lengths
    plt.xlim(0, 300)
    plt.title('Distribution of Message Character Length (Cap at 300 chars)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Character Length', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "message_length_dist.png"), dpi=150)
    plt.close()
    
    print("EDA Visualizations saved successfully.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
