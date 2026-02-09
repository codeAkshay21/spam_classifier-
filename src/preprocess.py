import string
import nltk
from nltk.stem import PorterStemmer

# Download the necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ps = PorterStemmer()

def clean_text(text):
    """
    Transforms raw text into clean, stemmed tokens.
    Input: "I am running!"
    Output: "run"
    """
    # 1. Lowercase
    text = str(text).lower()
    
    # 2. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenize & Stem (The Upgrade)
    # Split into words, stem each word, join back together
    words = text.split()
    stemmed_words = [ps.stem(word) for word in words]
    
    return " ".join(stemmed_words)