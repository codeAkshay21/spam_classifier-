import pandas as pd
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))
from src.preprocess import clean_text

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load & Clean Data
print("1. Loading Data...")
csv_path = 'data/spam.csv'

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File not found at {csv_path}")

try:
    df = pd.read_csv(csv_path, encoding='latin-1')
except:
    df = pd.read_csv(csv_path, encoding='utf-8')

df = df.iloc[:, :2]
df.columns = ['label', 'message']

print("2. Cleaning Text (Stemming & Tokenizing)...")
# Note: This might take 10-20 seconds
df['clean_message'] = df['message'].apply(clean_text)

# Encode Labels (Spam = 1, Ham = 0)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], 
    df['label_num'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label_num'] # Ensure Spam% is same in Train and Test
)

# 3. Build The Pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('classifier', SVC(probability=True, class_weight='balanced')) 
    # class_weight='balanced' fixes the "Rare Spam" problem automatically
])

# 4. Hyperparameter Tuning (The "Pro" Step)
# We define a "Grid" of settings to test
param_grid = {
    # Try different N-Grams (Single words vs. Phrases)
    'vectorizer__ngram_range': [(1, 1), (1, 2)], 
    
    # Try different regularization strengths (C)
    # C=0.1: Simple model (avoids overfitting)
    # C=10: Complex model (catches subtle patterns)
    'classifier__C': [0.1, 1, 10],
    
    # Kernel type (Linear is usually best for text, but RBF can find curves)
    'classifier__kernel': ['linear', 'rbf']
}

print("3. Tuning Hyperparameters (This will take a minute)...")
# n_jobs=-1 uses ALL your CPU cores to speed this up
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

# 5. Results