#!/usr/bin/env python3
"""
Advanced NLP Feature Engineering for TikTok Hackathon
NLP Engineer: Enhanced features beyond basic preprocessing
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import math

# Core NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Advanced NLP libraries
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Linguistic analysis
import textstat

class AdvancedNLPFeatureExtractor:
    """
    Advanced NLP feature extraction for review quality assessment
    Goes beyond basic preprocessing to extract deep linguistic patterns
    """
    
    def __init__(self):
        print("🚀 Initializing Advanced NLP Feature Extractor...")
        
        # Initialize NLTK components
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except OSError:
            print("⚠️ spaCy model not found - some features will be disabled")
            self.nlp = None
        
        # Initialize BERT tokenizer and model
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()  # Set to evaluation mode
            print("✅ BERT model loaded")
        except Exception as e:
            print(f"⚠️ BERT model loading failed: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
        
        # Policy violation patterns (enhanced from basic version)
        self.policy_patterns = self._init_policy_patterns()
        
        print("✅ Advanced NLP Feature Extractor ready!")
    
    def _init_policy_patterns(self):
        """Initialize sophisticated policy violation patterns"""
        
        return {
            'advertisement_patterns': {
                # URL and website patterns
                'url_mentions': [
                    r'https?://[^\s]+',
                    r'www\.[^\s]+\.[a-z]{2,}',
                    r'[a-zA-Z0-9-]+\.(com|org|net|edu|gov)',
                    r'visit.*website',
                    r'check.*out.*site'
                ],
                
                # Promotional language
                'promotional_language': [
                    r'\d+%\s*off',
                    r'\d+%\s*discount',
                    r'special.*offer',
                    r'limited.*time',
                    r'act.*now',
                    r'call.*now',
                    r'order.*now',
                    r'book.*now'
                ],
                
                # Contact information
                'contact_info': [
                    r'\(\d{3}\)\s*\d{3}-\d{4}',  # Phone numbers
                    r'\d{3}-\d{3}-\d{4}',
                    r'call\s+\d+',
                    r'phone.*\d+',
                    r'contact.*us',
                    r'reach.*out'
                ],
                
                # Social media promotion
                'social_media': [
                    r'@[a-zA-Z0-9_]+',  # Handles
                    r'follow.*us.*on',
                    r'like.*our.*page',
                    r'facebook\.com',
                    r'instagram\.com',
                    r'twitter\.com',
                    r'tiktok\.com'
                ]
            },
            
            'irrelevant_patterns': {
                # Technology mentions
                'technology': [
                    r'my\s+(phone|laptop|computer|tablet|iphone|android)',
                    r'wifi|internet|charging|battery',
                    r'app|software|download',
                    r'netflix|youtube|streaming',
                    r'social.*media|facebook|instagram'
                ],
                
                # Unrelated topics
                'off_topic': [
                    r'weather|rain|snow|sunny|cloudy',
                    r'politics|government|election|president',
                    r'sports|football|basketball|soccer|baseball',
                    r'movie|film|cinema|theater',
                    r'school|work|job|office|meeting'
                ],
                
                # Personal unrelated activities
                'personal_activities': [
                    r'shopping|mall|store(?!.*restaurant)',
                    r'doctor|dentist|hospital|appointment',
                    r'parking|traffic|driving|car(?!.*service)',
                    r'hotel|airport|flight|travel(?!.*food)'
                ]
            },
            
            'rant_no_visit_patterns': {
                # Direct admission of no visit
                'never_visited': [
                    r'never\s+(been|visited|went|tried)',
                    r'haven\'t\s+(been|visited|gone|tried)',
                    r'have\s+not\s+(been|visited|gone|tried)',
                    r'didn\'t\s+(go|visit|try)',
                    r'will\s+never\s+(go|visit|try)'
                ],
                
                # Indirect evidence
                'indirect_evidence': [
                    r'heard\s+(from|that|it)',
                    r'someone\s+(told|said)',
                    r'people\s+say',
                    r'according\s+to',
                    r'based\s+on\s+(reviews|photos|what)',
                    r'looks\s+like',
                    r'seems\s+like'
                ],
                
                # Hearsay indicators
                'hearsay': [
                    r'my\s+(friend|family|coworker)\s+(said|told)',
                    r'i\s+heard',
                    r'word\s+is',
                    r'reputation\s+is',
                    r'everyone\s+says'
                ]
            }
        }
    
    def extract_linguistic_features(self, text):
        """Extract linguistic and readability features"""
        
        features = {}
        
        if not text or pd.isna(text):
            return {f'linguistic_{k}': 0 for k in ['readability', 'complexity', 'formality', 'coherence']}
        
        text = str(text)
        
        # Readability scores
        try:
            features['linguistic_flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['linguistic_flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            features['linguistic_gunning_fog'] = textstat.gunning_fog(text)
            features['linguistic_automated_readability'] = textstat.automated_readability_index(text)
        except:
            features.update({
                'linguistic_flesch_reading_ease': 0,
                'linguistic_flesch_kincaid_grade': 0,
                'linguistic_gunning_fog': 0,
                'linguistic_automated_readability': 0
            })
        
        # Text complexity features
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        features['linguistic_sentence_count'] = len(sentences)
        features['linguistic_avg_sentence_length'] = len(words) / max(len(sentences), 1)
        features['linguistic_long_words'] = sum(1 for word in words if len(word) > 6) / max(len(words), 1)
        
        # Vocabulary diversity
        unique_words = set(words)
        features['linguistic_vocabulary_diversity'] = len(unique_words) / max(len(words), 1)
        
        # Punctuation analysis
        punct_count = sum(1 for char in text if char in string.punctuation)
        features['linguistic_punctuation_ratio'] = punct_count / max(len(text), 1)
        features['linguistic_exclamation_count'] = text.count('!')
        features['linguistic_question_count'] = text.count('?')
        
        # Capitalization patterns
        features['linguistic_caps_ratio'] = sum(1 for char in text if char.isupper()) / max(len(text), 1)
        features['linguistic_all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        return features
    
    def extract_sentiment_features(self, text):
        """Extract advanced sentiment features"""
        
        features = {}
        
        if not text or pd.isna(text):
            return {f'sentiment_{k}': 0 for k in ['compound', 'positive', 'negative', 'neutral', 'subjectivity', 'polarity']}
        
        text = str(text)
        
        # VADER sentiment (from basic pipeline, but more detailed)
        vader_scores = self.sia.polarity_scores(text)
        features['sentiment_vader_compound'] = vader_scores['compound']
        features['sentiment_vader_positive'] = vader_scores['pos']
        features['sentiment_vader_negative'] = vader_scores['neg']
        features['sentiment_vader_neutral'] = vader_scores['neu']
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            features['sentiment_textblob_polarity'] = blob.sentiment.polarity
            features['sentiment_textblob_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_textblob_polarity'] = 0
            features['sentiment_textblob_subjectivity'] = 0
        
        # Sentiment word counting
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'perfect', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting', 'nasty', 'poor', 'disappointing']
        
        words = word_tokenize(text.lower())
        features['sentiment_positive_word_count'] = sum(1 for word in words if word in positive_words)
        features['sentiment_negative_word_count'] = sum(1 for word in words if word in negative_words)
        
        return features
    
    def extract_pos_features(self, text):
        """Extract Part-of-Speech features"""
        
        features = {}
        
        if not text or pd.isna(text):
            return {f'pos_{k}': 0 for k in ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']}
        
        try:
            tokens = word_tokenize(str(text))
            pos_tags = pos_tag(tokens)
            
            # Count POS categories
            pos_counts = Counter(tag for word, tag in pos_tags)
            total_tags = len(pos_tags)
            
            # Noun ratio (NN, NNS, NNP, NNPS)
            noun_tags = sum(pos_counts[tag] for tag in pos_counts if tag.startswith('NN'))
            features['pos_noun_ratio'] = noun_tags / max(total_tags, 1)
            
            # Verb ratio (VB, VBD, VBG, VBN, VBP, VBZ)
            verb_tags = sum(pos_counts[tag] for tag in pos_counts if tag.startswith('VB'))
            features['pos_verb_ratio'] = verb_tags / max(total_tags, 1)
            
            # Adjective ratio (JJ, JJR, JJS)
            adj_tags = sum(pos_counts[tag] for tag in pos_counts if tag.startswith('JJ'))
            features['pos_adjective_ratio'] = adj_tags / max(total_tags, 1)
            
            # Adverb ratio (RB, RBR, RBS)
            adv_tags = sum(pos_counts[tag] for tag in pos_counts if tag.startswith('RB'))
            features['pos_adverb_ratio'] = adv_tags / max(total_tags, 1)
            
            # Pronoun ratio (PRP, PRP$)
            pron_tags = sum(pos_counts[tag] for tag in pos_counts if tag.startswith('PRP'))
            features['pos_pronoun_ratio'] = pron_tags / max(total_tags, 1)
            
        except:
            features.update({
                'pos_noun_ratio': 0,
                'pos_verb_ratio': 0,
                'pos_adjective_ratio': 0,
                'pos_adverb_ratio': 0,
                'pos_pronoun_ratio': 0
            })
        
        return features
    
    def extract_named_entity_features(self, text):
        """Extract Named Entity Recognition features"""
        
        features = {}
        
        if not text or pd.isna(text) or not self.nlp:
            return {f'ner_{k}': 0 for k in ['person_count', 'org_count', 'location_count', 'misc_count']}
        
        try:
            doc = self.nlp(str(text))
            
            # Count entity types
            entity_counts = Counter(ent.label_ for ent in doc.ents)
            
            features['ner_person_count'] = entity_counts.get('PERSON', 0)
            features['ner_organization_count'] = entity_counts.get('ORG', 0)
            features['ner_location_count'] = entity_counts.get('GPE', 0) + entity_counts.get('LOC', 0)
            features['ner_money_count'] = entity_counts.get('MONEY', 0)
            features['ner_date_count'] = entity_counts.get('DATE', 0)
            features['ner_time_count'] = entity_counts.get('TIME', 0)
            features['ner_total_entities'] = len(doc.ents)
            
        except:
            features.update({
                'ner_person_count': 0,
                'ner_organization_count': 0,
                'ner_location_count': 0,
                'ner_money_count': 0,
                'ner_date_count': 0,
                'ner_time_count': 0,
                'ner_total_entities': 0
            })
        
        return features
    
    def extract_bert_embeddings(self, text, max_length=128):
        """Extract BERT embeddings (simplified version for hackathon)"""
        
        if not text or pd.isna(text) or not self.bert_model:
            # Return zero vector if BERT not available
            return {f'bert_dim_{i}': 0.0 for i in range(12)}  # Simplified to 12 dimensions
        
        try:
            # Tokenize and encode
            text = str(text)[:500]  # Limit text length for speed
            inputs = self.bert_tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding (first token)
                cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
                
                # Reduce dimensionality for hackathon (PCA-like reduction)
                # Take every 64th dimension to get 12 features
                reduced_embedding = cls_embedding[::64][:12]
                
                return {f'bert_dim_{i}': float(reduced_embedding[i]) for i in range(len(reduced_embedding))}
        
        except Exception as e:
            print(f"BERT embedding failed: {e}")
            return {f'bert_dim_{i}': 0.0 for i in range(12)}
    
    def extract_enhanced_policy_features(self, text):
        """Extract enhanced policy violation features"""
        
        features = {}
        
        if not text or pd.isna(text):
            return {f'policy_{k}': 0 for k in ['ad_score', 'irrelevant_score', 'rant_score']}
        
        text = str(text).lower()
        
        # Enhanced advertisement detection
        ad_score = 0
        for category, patterns in self.policy_patterns['advertisement_patterns'].items():
            category_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
            features[f'policy_ad_{category}'] = category_matches
            ad_score += category_matches
        
        features['policy_advertisement_score'] = ad_score
        
        # Enhanced irrelevant content detection
        irrelevant_score = 0
        for category, patterns in self.policy_patterns['irrelevant_patterns'].items():
            category_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
            features[f'policy_irrelevant_{category}'] = category_matches
            irrelevant_score += category_matches
        
        features['policy_irrelevant_score'] = irrelevant_score
        
        # Enhanced rant without visit detection
        rant_score = 0
        for category, patterns in self.policy_patterns['rant_no_visit_patterns'].items():
            category_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
            features[f'policy_rant_{category}'] = category_matches
            rant_score += category_matches
        
        features['policy_rant_score'] = rant_score
        
        return features
    
    def extract_all_advanced_features(self, text):
        """Extract all advanced NLP features for a single text"""
        
        features = {}
        
        # Extract each category of features
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_pos_features(text))
        features.update(self.extract_named_entity_features(text))
        features.update(self.extract_bert_embeddings(text))
        features.update(self.extract_enhanced_policy_features(text))
        
        return features
    
    def process_dataframe(self, df, text_column='review_text'):
        """Process entire dataframe with advanced NLP features"""
        
        print(f"🔧 Processing {len(df)} reviews with advanced NLP features...")
        
        # Extract features for all reviews
        all_features = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                print(f"   Processed {idx}/{len(df)} reviews...")
            
            features = self.extract_all_advanced_features(text)
            all_features.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features, index=df.index)
        
        # Combine with original dataframe
        result_df = pd.concat([df, features_df], axis=1)
        
        print(f"✅ Advanced NLP processing complete!")
        print(f"   Added {len(features_df.columns)} new features")
        print(f"   Total features: {len(result_df.columns)}")
        
        return result_df

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Advanced NLP Feature Extractor...")
    
    # Test with sample texts
    test_texts = [
        "Great food and amazing service! Will definitely come back.",
        "Visit our website www.promo.com for 50% off! Call now!",
        "I love my new iPhone but this place is too noisy.",
        "Never been here but heard it's terrible from friends."
    ]
    
    extractor = AdvancedNLPFeatureExtractor()
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text[:50]}...'")
        features = extractor.extract_all_advanced_features(text)
        print(f"Extracted {len(features)} features")
        
        # Show some interesting features
        interesting_features = [k for k in features.keys() if 
                              'sentiment' in k or 'policy' in k or 'linguistic' in k][:5]
        for feature in interesting_features:
            print(f"  {feature}: {features[feature]:.3f}")
    
    print("\n✅ Advanced NLP Feature Extractor test complete!")
