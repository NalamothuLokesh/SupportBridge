"""
Machine Learning Training Script for Ticket Classification
Trains category and priority prediction models on ticket data
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

from utils.preprocessing import clean_text, preprocess_tickets


def load_and_prepare_data(csv_path):
    """
    Load ticket data and prepare for model training.
    
    Args:
        csv_path: Path to tickets.csv
        
    Returns:
        Tuple of (features, category_labels, priority_labels)
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Total tickets: {len(df)}")
    
    # Preprocess tickets
    df = preprocess_tickets(df)
    
    # Prepare features
    X = df['cleaned_text'].values
    y_category = df['category'].values
    y_priority = df['priority'].values
    
    return X, y_category, y_priority, df


def train_category_model(X, y_category):
    """
    Train text vectorizer and category classification model.
    
    Args:
        X: Feature texts
        y_category: Category labels
        
    Returns:
        Tuple of (vectorizer, model)
    """
    print("\n" + "="*60)
    print("TRAINING CATEGORY CLASSIFIER")
    print("="*60)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Vectorize text
    X_vectorized = vectorizer.fit_transform(X)
    
    # Convert to numpy array to avoid pandas/sklearn compatibility issues
    y_category_np = np.array(y_category)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_category_np, test_size=0.2, random_state=42, stratify=y_category_np
    )
    
    # Train model
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print("Training Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nCategory Classification Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return vectorizer, model, accuracy


def train_priority_model(X, y_priority):
    """
    Train priority classification model using same vectorizer.
    
    Args:
        X: Feature texts (already vectorized)
        y_priority: Priority labels
        
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING PRIORITY CLASSIFIER")
    print("="*60)
    
    # Use existing vectorizer would be better, but create new for independence
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_vectorized = vectorizer.fit_transform(X)
    
    # Convert to numpy array to avoid pandas/sklearn compatibility issues
    y_priority_np = np.array(y_priority)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_priority_np, test_size=0.2, random_state=42, stratify=y_priority_np
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print("Training Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPriority Classification Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return vectorizer, model, accuracy


def save_models(vectorizer, category_model, priority_vectorizer, priority_model, models_dir='models'):
    """
    Save trained models and vectorizers to disk.
    
    Args:
        vectorizer: TF-IDF vectorizer for categories
        category_model: Trained category classifier
        priority_vectorizer: TF-IDF vectorizer for priority
        priority_model: Trained priority classifier
        models_dir: Directory to save models
    """
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Save vectorizer (used for both)
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Saved vectorizer.pkl")
    
    # Save category model
    with open(os.path.join(models_dir, 'category_model.pkl'), 'wb') as f:
        pickle.dump(category_model, f)
    print(f"✓ Saved category_model.pkl")
    
    # Save priority model
    with open(os.path.join(models_dir, 'priority_model.pkl'), 'wb') as f:
        pickle.dump(priority_model, f)
    print(f"✓ Saved priority_model.pkl")
    
    print(f"\nAll models saved to '{models_dir}/' directory")


def main():
    """Main training pipeline."""
    import sys
    
    csv_path = 'tickets.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        sys.exit(1)
    
    # Load data
    X, y_category, y_priority, df = load_and_prepare_data(csv_path)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"\nCategory Distribution:")
    print(df['category'].value_counts())
    print(f"\nPriority Distribution:")
    print(df['priority'].value_counts())
    
    # Train models
    vectorizer, category_model, cat_accuracy = train_category_model(X, y_category)
    priority_vectorizer, priority_model, pri_accuracy = train_priority_model(X, y_priority)
    
    # Save models
    save_models(vectorizer, category_model, priority_vectorizer, priority_model)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Category Model Accuracy: {cat_accuracy:.4f}")
    print(f"Priority Model Accuracy: {pri_accuracy:.4f}")
    print(f"Average Accuracy: {(cat_accuracy + pri_accuracy) / 2:.4f}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
