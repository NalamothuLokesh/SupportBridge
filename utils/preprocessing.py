import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def clean_text(text):
    """
    Clean and preprocess ticket text.
    
    Args:
        text: Raw ticket text
        
    Returns:
        Cleaned text ready for ML model
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def preprocess_tickets(df):
    """
    Preprocess entire dataframe of tickets.
    
    Args:
        df: DataFrame with ticket data
        
    Returns:
        DataFrame with cleaned text and combined features
    """
    df = df.copy()
    
    # Combine subject and description for analysis
    df['combined_text'] = df['subject'].fillna('') + ' ' + df['description'].fillna('')
    
    # Clean the text
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    return df


def extract_features(text):
    """
    Extract linguistic features from ticket text.
    
    Args:
        text: Cleaned ticket text
        
    Returns:
        Dictionary of extracted features
    """
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'unique_words': len(set(text.split())),
        'avg_word_length': len(text) / max(len(text.split()), 1)
    }
    
    return features


def get_priority_keywords():
    """Define keyword patterns for priority detection."""
    return {
        'critical': ['urgent', 'critical', 'crash', 'down', 'error', 'failed', 'cannot', 'blocked'],
        'high': ['issue', 'problem', 'error', 'not working', 'broken', 'urgent'],
        'medium': ['slow', 'delayed', 'inconsistent', 'improvement'],
        'low': ['feature', 'request', 'enhancement', 'how to', 'help']
    }


def get_category_keywords():
    """Define keyword patterns for category detection."""
    return {
        'Account': ['account', 'login', 'password', 'authentication', 'access', 'credentials', 'locked'],
        'Billing': ['billing', 'invoice', 'payment', 'charge', 'refund', 'discount', 'price', 'subscription'],
        'Technical': ['error', 'bug', 'crash', 'slow', 'database', 'api', 'sync', 'timeout', 'connection'],
        'Feature Request': ['feature', 'request', 'enhancement', 'add', 'new', 'dark mode', 'export']
    }
