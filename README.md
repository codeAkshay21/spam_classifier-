# 📧 SMS Spam Classifier (NLP & SVM)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

A machine learning project that accurately classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) techniques and Support Vector Machines (SVM).

The project features a **Streamlit Web App** for real-time predictions and uses **GridSearchCV** to scientifically optimize model performance.

---

## 🚀 Live Demo
You can run the application locally to test predictions instantly.

1. **Enter a message:** "URGENT! You have won a £1000 cash prize."
2. **Result:** 🚨 **SPAM DETECTED** (99.8% Confidence)

![App Screenshot](app_screenshot.png) 
*(Note: Add a screenshot of your Streamlit app here)*

---

## 🧠 Key Features & Technical Approach

### 1. Advanced Text Preprocessing (NLP)
Raw text data is messy. I implemented a custom preprocessing pipeline using **NLTK**:
* **Tokenization:** Breaking sentences into individual words.
* **Stemming (PorterStemmer):** Reducing words to their root form (e.g., "running", "ran", "runs" → "run") to reduce feature dimensionality.
* **Vectorization (TF-IDF):** Converted text to numerical vectors, weighing unique words higher than common stopwords.

### 2. Model Optimization
Instead of guessing parameters, I used **GridSearchCV** to tune the model:
* **Algorithm:** Support Vector Machine (SVM) with a Linear Kernel (best for high-dimensional text data).
* **N-Grams:** Analyzed both unigrams ("free") and bi-grams ("free entry") to capture context.
* **Class Balancing:** Applied `class_weight='balanced'` to handle the dataset imbalance (since Spam is rare compared to Ham).

---

## 📊 Business Logic: The Precision/Recall Tradeoff

In spam detection, **Precision is king.**

* **High Precision:** The model rarely flags a real email as spam. (Crucial: We don't want users missing job offers or OTPs).
* **High Recall:** The model catches *all* spam.

**My Approach:**
I optimized the model to maximize **Precision** for the 'Spam' class. A False Negative (missing a spam message) is annoying, but a False Positive (deleting a real email) is catastrophic.

**Performance Metrics:**
* **Precision:** ~98%
* **Accuracy:** ~98.5%

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* pip

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your-username/spam-email-classifier.git](https://github.com/your-username/spam-email-classifier.git)
cd spam-email-classifier