import numpy as np
from lime.lime_text import LimeTextExplainer
import streamlit as st

class ModelExplainer:
    """
    Wrapper for LIME Text Explainer to interpret model predictions.
    """
    
    def __init__(self, vectorizer, model, class_names):
        """
        Initialize the explainer.
        
        Args:
            vectorizer: Fitted vectorizer (e.g., TfidfVectorizer)
            model: Fitted classifier (e.g., SGDClassifier)
            class_names: List of class names
        """
        self.vectorizer = vectorizer
        self.model = model
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)
        
    def predict_proba(self, texts):
        """
        Prediction pipeline for LIME: Text -> Vector -> Probability
        """
        features = self.vectorizer.transform(texts)
        return self.model.predict_proba(features)
    
    def explain_instance(self, text, num_features=10, num_samples=1000):
        """
        Explain a single text instance.
        
        Returns:
            LIME Explanation object
        """
        exp = self.explainer.explain_instance(
            text, 
            self.predict_proba, 
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1
        )
        return exp

def plot_lime_explanation(explanation):
    """
    Visualize LIME explanation in Streamlit.
    """
    # Get the explanation for the top predicted class
    top_label = explanation.top_labels[0]
    local_exp = explanation.local_exp[top_label]
    
    # Sort by absolute weight
    local_exp.sort(key=lambda x: abs(x[1]), reverse=False)
    
    features = [x[0] for x in local_exp]
    weights = [x[1] for x in local_exp]
    feature_names = [explanation.domain_mapper.indexed_string.word(x) for x in features]
    
    # Create plot
    import plotly.graph_objects as go
    
    # Colors: Green for positive contribution, Red for negative
    colors = ['#10B981' if w > 0 else '#EF4444' for w in weights]
    
    fig = go.Figure(go.Bar(
        x=weights,
        y=feature_names,
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title=f"Word Contributions to Prediction",
        xaxis_title="Weight (Contribution)",
        yaxis_title="Word",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig
