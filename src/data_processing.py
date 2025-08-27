import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.sia = SentimentIntensityAnalyzer()
        
    def clean_text(self, text):
        """Basic text cleaning function"""
        if pd.isna(text):
            return ""
            
        text = str(text)
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        # Remove special characters and digits (keep only letters, spaces, and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text):
        """Tokenize and stem text"""
        tokens = word_tokenize(text)
        filtered_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return filtered_tokens
    
    def get_sentiment_score(self, text):
        """Get sentiment score using VADER"""
        return self.sia.polarity_scores(text)['compound']
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if nlp is None:
            return []
            
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    def preprocess_dataframe(self, df, text_column='review'):
        """
        Preprocess an entire dataframe with multiple text features
        
        Parameters:
        df: pandas DataFrame
        text_column: name of the column containing review text
        
        Returns:
        Preprocessed DataFrame with additional features
        """
        df_clean = df.copy()
        
        # Clean text
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        
        # Text statistics
        df_clean['review_length'] = df_clean['cleaned_text'].apply(len)
        df_clean['word_count'] = df_clean['cleaned_text'].apply(lambda x: len(x.split()))
        
        # Sentiment analysis
        df_clean['sentiment_score'] = df_clean['cleaned_text'].apply(self.get_sentiment_score)
        
        # Tokenization and stemming
        df_clean['tokens'] = df_clean['cleaned_text'].apply(self.tokenize_and_stem)
        df_clean['token_count'] = df_clean['tokens'].apply(len)
        
        # Extract entities (if spaCy is available)
        if nlp:
            df_clean['entities'] = df_clean['cleaned_text'].apply(self.extract_entities)
            df_clean['entity_count'] = df_clean['entities'].apply(len)
        
        return df_clean


class PolicyFeatureExtractor:
    def __init__(self):
        # Patterns for detecting advertisements
        self.ad_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'visit our (?:website|site|page)',
            r'discount|promo|promotion|offer|sale|deal|coupon',
            r'%\s*off|%\s*discount|\$\d+ off',
            r'call now|contact us|order now|buy now',
            r'facebook\.com|instagram\.com|twitter\.com|youtube\.com',
            r'@[a-zA-Z0-9_]+',  # Social media handles
        ]
        
        # Patterns for detecting irrelevant content
        self.irrelevant_patterns = [
            r'phone|computer|laptop|tv|television|car|vehicle|game|console',
            r'movie|film|netflix|youtube|video|tiktok|social media',
            r'government|politics|election|president|vote',
            r'sports|basketball|football|soccer|baseball|hockey',
            r'weather|rain|snow|temperature|forecast',
        ]
        
        # Patterns for detecting rants from non-visitors
        self.rant_patterns = [
            r'never (?:been|visited|gone|ate)',
            r'haven\'t (?:been|visited|gone|ate)',
            r"have n't (?:been|visited|gone|ate)",
            r'didn\'t (?:go|visit|eat|try)',
            r"did n't (?:go|visit|eat|try)",
            r'heard (?:it|they|that)',
            r'friend told|someone said|people say',
            r'based on (?:reviews|pictures|what i heard)',
            r'looks (?:bad|terrible|awful|disgusting)',
            r'sounds (?:bad|terrible|awful|disgusting)',
        ]
        
    def extract_advertisement_features(self, text):
        """Extract features related to advertisements"""
        features = {}
        text = str(text).lower()
        
        for i, pattern in enumerate(self.ad_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'ad_pattern_{i}'] = len(matches)
            
        features['has_url'] = 1 if any(term in text for term in ['http', 'www.', '.com', '.org']) else 0
        features['has_discount_mention'] = 1 if any(term in text for term in ['discount', 'promo', 'offer', 'sale']) else 0
        features['has_contact_mention'] = 1 if any(term in text for term in ['call', 'contact', 'order', 'buy']) else 0
        
        return features
    
    def extract_irrelevancy_features(self, text):
        """Extract features related to irrelevant content"""
        features = {}
        text = str(text).lower()
        
        for i, pattern in enumerate(self.irrelevant_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'irrelevant_pattern_{i}'] = len(matches)
            
        features['has_off_topic_mention'] = 1 if any(term in text for term in [
            'phone', 'movie', 'government', 'sports', 'weather'
        ]) else 0
        
        return features
    
    def extract_rant_features(self, text):
        """Extract features related to rants from non-visitors"""
        features = {}
        text = str(text).lower()
        
        for i, pattern in enumerate(self.rant_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'rant_pattern_{i}'] = len(matches)
            
        features['has_never_mention'] = 1 if 'never' in text and any(term in text for term in [
            'been', 'visited', 'gone', 'ate'
        ]) else 0
        
        features['has_heard_mention'] = 1 if any(term in text for term in [
            'heard', 'told', 'said', 'according to'
        ]) else 0
        
        return features
    
    def extract_all_features(self, text):
        """Extract all policy-related features"""
        features = {}
        features.update(self.extract_advertisement_features(text))
        features.update(self.extract_irrelevancy_features(text))
        features.update(self.extract_rant_features(text))
        return features
    
    def extract_features_dataframe(self, df, text_column='cleaned_text'):
        """Extract policy features for an entire dataframe"""
        df_features = df.copy()
        
        # Extract all features
        feature_dicts = df_features[text_column].apply(self.extract_all_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_dicts.tolist(), index=df_features.index)
        
        # Combine with original DataFrame
        df_features = pd.concat([df_features, features_df], axis=1)
        
        return df_features


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'review': [
            'Great food and amazing service! Will definitely come back again.',
            'Visit our website for discounts: www.promo.com Best pizza in town!',
            'I love my new phone but this place is too noisy. Not recommended.',
            'Never been here but heard its terrible from a friend.',
            'The pizza was delicious and staff was friendly. Great atmosphere!',
            'Worst experience ever. Food was cold and service was rude.',
            'Check out our promotion: 20% off on all items this week only!',
            'This review is not about the restaurant but about my new car.',
            'Average place, nothing special. Would not go out of my way to visit.',
            'Amazing ambiance and the staff went above and beyond!'
        ],
        'rating': [5, 1, 2, 1, 5, 1, 3, 2, 3, 5],
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test TextPreprocessor
    print("Testing TextPreprocessor...")
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataframe(df, 'review')
    print("Cleaned DataFrame:")
    print(df_clean[['review', 'cleaned_text', 'sentiment_score', 'word_count']].head())
    print()
    
    # Test PolicyFeatureExtractor
    print("Testing PolicyFeatureExtractor...")
    feature_extractor = PolicyFeatureExtractor()
    df_features = feature_extractor.extract_features_dataframe(df_clean)
    
    # Show some key features
    ad_features = [col for col in df_features.columns if 'ad_pattern' in col or 'has_url' in col]
    irrelevant_features = [col for col in df_features.columns if 'irrelevant_pattern' in col]
    rant_features = [col for col in df_features.columns if 'rant_pattern' in col]
    
    print("Advertisement features for sample reviews:")
    print(df_features[['review'] + ad_features[:3]].head())
    print()
    
    print("Sample review analysis complete!")