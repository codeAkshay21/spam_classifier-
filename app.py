import streamlit as st
import joblib
from src.preprocess import clean_text
import os 

# ROBUST LOADING LOGIC
# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the model file
model_path = os.path.join(current_dir, 'spam_classifier.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("🚨 Model not found! Please run 'python3 train_model.py' first.")
    st.stop()
    
# 1. Load the Pipeline
model = joblib.load('spam_classifier.pkl')

# 2. UI Layout
st.title("📧 Spam Email Detector")
st.write("Is that email real or a scam? Paste the text below to find out.")

# 3. User Input
user_input = st.text_area("Paste Email/SMS Text Here:", height=150)

# 4. Prediction Logic
if st.button("Check for Spam"):
    if user_input:
        # Preprocess the input exactly like training data
        cleaned_input = clean_text(user_input)
        
        # Predict (The pipeline handles vectorization automatically!)
        prediction = model.predict([cleaned_input])[0]
        probability = model.predict_proba([cleaned_input])[0]
        
        if prediction == 1:
            st.error(f"🚨 SPAM DETECTED! (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ Looks like a normal email. (Confidence: {probability[0]*100:.1f}%)")
    else:
        st.warning("Please enter some text first.")