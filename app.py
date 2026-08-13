import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import SpamDetector

# Set Page Config
st.set_page_config(
    page_title="SpamGuard AI - SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Theme & Glassmorphism)
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151a24 100%);
        color: #ecf0f1;
    }
    
    /* Title Styling */
    .title-container {
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .main-title {
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Cards & Components */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    }
    
    /* Result Box Styling */
    .result-box {
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1.5rem;
        font-weight: 600;
        font-size: 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-out;
    }
    
    .result-spam {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15) 0%, rgba(231, 76, 60, 0.3) 100%);
        border: 1.5px solid #e74c3c;
        color: #ff6b6b;
    }
    
    .result-ham {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(46, 204, 113, 0.3) 100%);
        border: 1.5px solid #2ecc71;
        color: #2ecc71;
    }
    
    /* Keyframes for animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(30, 60, 114, 0.4) !important;
    }
    
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown("""
    <div class="title-container">
        <h1 class="main-title">🛡️ SpamGuard AI</h1>
        <p class="subtitle">Machine Learning Powered SMS Spam Mail Detector</p>
    </div>
""", unsafe_allow_html=True)

# Load Detector
@st.cache_resource
def get_detector():
    try:
        return SpamDetector(models_dir="models")
    except Exception as e:
        st.error(f"Error loading models: {e}. Make sure you run model training first.")
        return None

detector = get_detector()

# Sidebar Setup
st.sidebar.markdown("### 📊 Project Metadata")
st.sidebar.info("""
- **Course/Internship**: QSkill AI & ML
- **Dataset**: SMS Spam Collection
- **Vectorizer**: TF-IDF (Unigrams & Bigrams)
- **Primary Model**: Multinomial Naive Bayes
- **Developer**: AI Assistant
""")

# Load Best Model Info
best_model_name = "Multinomial Naive Bayes"
if os.path.exists("models/best_model_info.txt"):
    with open("models/best_model_info.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Best Model Name:" in line:
                best_model_name = line.split(":", 1)[1].strip()

st.sidebar.markdown(f"### 🏆 Selected Classifier")
st.sidebar.success(f"**{best_model_name}**")

# Main Tabs Layout
tab1, tab2, tab3 = st.tabs(["🔍 Spam Classifier", "📈 Model Performance", "📊 Dataset Insights (EDA)"])

# TAB 1: Spam Classifier
with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Message Analysis Input")
        
        user_input = st.text_area(
            "Enter the message you want to check for spam:",
            placeholder="Paste text here...",
            height=180
        )
        
        col_buttons = st.columns([1, 4])
        with col_buttons[0]:
            submit_btn = st.button("Check Message")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submit_btn or user_input:
            if not user_input.strip():
                st.warning("Please enter a valid message.")
            elif detector:
                # Perform prediction
                result = detector.predict(user_input)
                
                # Render results card
                if result['is_spam']:
                    st.markdown(f"""
                        <div class="result-box result-spam">
                            🚨 DETECTED SPAM
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-box result-ham">
                            ✅ HAM / NOT SPAM
                        </div>
                    """, unsafe_allow_html=True)
                
                # Details Section
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Model Verdict Details")
                
                col_metric1, col_metric2 = st.columns(2)
                
                # Confidence Progress
                confidence_pct = result['confidence'] * 100
                col_metric1.metric("Prediction Confidence", f"{confidence_pct:.2f}%")
                col_metric2.metric("Filtered Message Length (Words)", len(result['processed_message'].split()))
                
                st.markdown("##### Preprocessed tokens fed to the model:")
                if result['processed_message']:
                    st.code(result['processed_message'], language='text')
                else:
                    st.info("The message didn't contain any informative terms after stopword removal and tokenization.")
                st.markdown('</div>', unsafe_allow_html=True)
                
    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💡 Sample Messages to Try")
        st.write("Click on any preset message below to load it into the detector:")
        
        samples = [
            {
                "text": "WINNER! You won a cash prize of £1000. Call 08000930705 now to claim your reward!",
                "label": "Spam Sample 1"
            },
            {
                "text": "Hey, are we still meeting for lunch today? Let me know.",
                "label": "Ham Sample 1"
            },
            {
                "text": "URGENT! Your mobile number has been selected for a £2000 bonus. Call now!",
                "label": "Spam Sample 2"
            },
            {
                "text": "Can you please grab some milk on your way back home? Thanks.",
                "label": "Ham Sample 2"
            }
        ]
        
        for idx, sample in enumerate(samples):
            # Unique key for each button
            if st.button(sample['label'], key=f"sample_btn_{idx}"):
                # Trigger a rerun by assigning value to text area
                st.session_state["input_text_val"] = sample['text']
                st.rerun()
                
        # Handle session state for preset messages
        if "input_text_val" in st.session_state:
            st.info(f"Loaded preset message. Click **'Check Message'** to run the prediction.")
            user_input = st.text_area("Enter the message you want to check for spam:", value=st.session_state["input_text_val"], height=180, key="preset_input")
            del st.session_state["input_text_val"] # clear after displaying
            
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Model Performance
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance Comparison")
    st.write("Both Naive Bayes and Logistic Regression classifiers were trained and compared. Below is the performance summary on the held-out test split (20% of dataset).")
    
    # Model comparison table
    comparison_data = {
        "Metric": ["Accuracy", "Precision (No False Spam)", "Recall (Spam Detection)", "F1-Score"],
        "Multinomial Naive Bayes": ["96.91%", "100.00%", "75.57%", "86.09%"],
        "Logistic Regression": ["95.84%", "95.83%", "70.23%", "81.06%"]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df.set_index("Metric"))
    
    st.markdown("""
    > [!IMPORTANT]
    > **Why Multinomial Naive Bayes was selected:**
    > For spam filtering, **Precision** is extremely critical. A precision of 100.00% means that **zero legitimate (ham) messages were misclassified as spam**. This ensures users never miss crucial personal or business emails because they were mistakenly moved to the spam folder.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Confusion Matrices Visualization
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Confusion Matrices (Test Set)")
    
    confusion_matrices_img = "static/confusion_matrices.png"
    if os.path.exists(confusion_matrices_img):
        st.image(confusion_matrices_img, caption="Confusion Matrix: Multinomial Naive Bayes vs. Logistic Regression", use_container_width=True)
    else:
        st.warning("Confusion matrix plot not found. Run training script to generate.")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Dataset Insights
with tab3:
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Class Distribution (Spam vs. Ham)")
        st.write("The raw SMS Spam Collection contains 5,572 messages. After removing duplicate entries (403 rows), the distribution is highly imbalanced with **12.63% spam** and **87.37% ham**.")
        
        class_dist_img = "static/class_distribution.png"
        if os.path.exists(class_dist_img):
            st.image(class_dist_img, caption="Class Distribution of Preprocessed SMS Messages", use_container_width=True)
        else:
            st.warning("Class distribution plot not found.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_eda2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Message Length Analysis")
        st.write("Spam messages tend to be significantly longer and more standardized in character length compared to Ham messages. This makes character count and vocabulary diversity excellent discriminative features.")
        
        len_dist_img = "static/message_length_dist.png"
        if os.path.exists(len_dist_img):
            st.image(len_dist_img, caption="Kernel Density Estimate (KDE) of Message Character Lengths", use_container_width=True)
        else:
            st.warning("Message length distribution plot not found.")
        st.markdown('</div>', unsafe_allow_html=True)
