import pandas as pd
import re
import os
from pathlib import Path
import ssl
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# SSL workaround for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class TextPreprocessor:
    """Basic text preprocessing for reviews"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www.\S+", " URL ", text)  # replace URLs
        text = re.sub(r"[^a-z0-9\s]", "", text)          # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def get_sentiment_score(self, text):
        if not text or pd.isna(text):
            return 0.0
        scores = self.sia.polarity_scores(str(text))
        return scores['compound']
    
    def preprocess_dataframe(self, df, text_column):
        """Preprocess entire dataframe"""
        result_df = df.copy()
        result_df['cleaned_text'] = result_df[text_column].apply(self.clean_text)
        result_df['sentiment_score'] = result_df['cleaned_text'].apply(self.get_sentiment_score)
        result_df['review_length'] = result_df[text_column].str.len().fillna(0)
        result_df['word_count'] = result_df[text_column].str.split().str.len().fillna(0)
        return result_df

class PolicyFeatureExtractor:
    """Extract basic policy violation features"""
    
    def __init__(self):
        pass
    
    def extract_features_dataframe(self, df, text_column):
        """Extract policy features from dataframe"""
        result_df = df.copy()
        
        # Basic policy features
        texts = df[text_column].fillna('')
        
        # Advertisement indicators
        result_df['has_url'] = texts.str.contains(r'http|www\.', case=False, na=False)
        result_df['has_phone'] = texts.str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}', na=False)
        result_df['has_discount'] = texts.str.contains(r'\d+%.*off|discount', case=False, na=False)
        
        # Irrelevant indicators  
        result_df['mentions_technology'] = texts.str.contains(r'phone|wifi|internet|app', case=False, na=False)
        result_df['mentions_unrelated'] = texts.str.contains(r'weather|politics|sports', case=False, na=False)
        
        # Rant indicators
        result_df['never_visited'] = texts.str.contains(r'never.*been|haven\'t.*been', case=False, na=False)
        result_df['hearsay'] = texts.str.contains(r'heard.*from|friend.*said', case=False, na=False)
        
        return result_df

def clean_text(text):
    """Legacy function for compatibility"""
    processor = TextPreprocessor()
    return processor.clean_text(text)

def preprocess(in_path, out_path):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(out_path)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(in_path)
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in dataset")

    df["review_clean"] = df["text"].apply(clean_text)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned data to {out_path}")

if __name__ == "__main__":
    # Ask user for input file
    in_path = input("Enter the path to your reviews CSV file: ").strip()

    # Default output file
    filename = os.path.basename(in_path).replace(".csv", "_clean.csv")
    out_path = os.path.join("data/processed", filename)

    preprocess(in_path, out_path)