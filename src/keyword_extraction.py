#!/usr/bin/env python3
"""
Advanced Keyword Extraction for Restaurant Reviews
Identifies key terms, phrases, and insights from review text
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import math

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import spacy

class RestaurantKeywordExtractor:
    """
    Advanced keyword extraction specifically designed for restaurant reviews
    Extracts meaningful terms, phrases, and insights
    """
    
    def __init__(self):
        print("🔍 Initializing Restaurant Keyword Extractor...")
        
        self.stop_words = set(stopwords.words('english'))
        
        # Add domain-specific stop words
        restaurant_stopwords = {
            'restaurant', 'place', 'went', 'go', 'get', 'got', 'would', 'could',
            'one', 'time', 'back', 'way', 'well', 'much', 'also', 'even', 'think',
            'really', 'quite', 'pretty', 'very', 'just', 'still', 'always'
        }
        self.stop_words.update(restaurant_stopwords)
        
        # Load spaCy model for advanced extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except OSError:
            print("⚠️ spaCy model not found - some features disabled")
            self.nlp = None
        
        # Restaurant-specific keyword categories
        self.keyword_categories = {
            'food_items': [
                'pizza', 'pasta', 'burger', 'salad', 'soup', 'steak', 'chicken', 'fish',
                'seafood', 'sushi', 'sandwich', 'appetizer', 'dessert', 'bread', 'wine',
                'beer', 'cocktail', 'coffee', 'tea', 'sauce', 'cheese', 'vegetables'
            ],
            'food_quality': [
                'delicious', 'tasty', 'fresh', 'flavorful', 'seasoned', 'tender', 'crispy',
                'juicy', 'spicy', 'sweet', 'salty', 'bitter', 'bland', 'overcooked',
                'undercooked', 'burnt', 'cold', 'hot', 'warm', 'perfect', 'amazing'
            ],
            'service_terms': [
                'service', 'waiter', 'waitress', 'server', 'staff', 'host', 'hostess',
                'manager', 'chef', 'bartender', 'friendly', 'rude', 'helpful', 'attentive',
                'slow', 'fast', 'quick', 'professional', 'courteous', 'accommodating'
            ],
            'ambiance_terms': [
                'atmosphere', 'ambiance', 'music', 'noise', 'loud', 'quiet', 'romantic',
                'cozy', 'comfortable', 'crowded', 'spacious', 'clean', 'dirty', 'lighting',
                'decor', 'interior', 'outdoor', 'patio', 'view', 'beautiful'
            ],
            'value_terms': [
                'price', 'expensive', 'cheap', 'affordable', 'reasonable', 'overpriced',
                'value', 'worth', 'money', 'cost', 'bill', 'tab', 'budget', 'deal',
                'special', 'promotion', 'discount', 'portion', 'size', 'generous'
            ],
            'location_terms': [
                'location', 'parking', 'downtown', 'convenient', 'accessible', 'area',
                'neighborhood', 'street', 'avenue', 'corner', 'nearby', 'close',
                'far', 'drive', 'walk', 'public', 'transport', 'metro', 'bus'
            ]
        }
        
        # Sentiment-bearing phrases
        self.sentiment_phrases = {
            'positive': [
                'highly recommend', 'definitely recommend', 'will return', 'coming back',
                'loved it', 'amazing experience', 'exceeded expectations', 'perfect for',
                'best ever', 'fantastic job', 'wonderful time', 'great choice'
            ],
            'negative': [
                'never again', 'waste of money', 'terrible experience', 'disappointed',
                'not worth', 'poor quality', 'bad service', 'would not recommend',
                'avoid this', 'stayed away', 'horrible food', 'worst ever'
            ]
        }
        
        print("✅ Keyword Extractor ready!")
    
    def extract_tfidf_keywords(self, texts, max_features=100, ngram_range=(1, 2)):
        """Extract keywords using TF-IDF scoring"""
        
        print(f"📊 Extracting TF-IDF keywords from {len(texts)} texts...")
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=ngram_range,
            stop_words=list(self.stop_words)
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Create keyword-score pairs
        keyword_scores = list(zip(feature_names, mean_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores
    
    def extract_pos_keywords(self, texts, target_pos=['NN', 'NNS', 'JJ', 'VB']):
        """Extract keywords based on Part-of-Speech tags"""
        
        print(f"🏷️ Extracting POS-based keywords...")
        
        pos_keywords = defaultdict(list)
        
        for text in texts:
            if pd.isna(text) or not text:
                continue
            
            # Tokenize and tag
            tokens = word_tokenize(str(text).lower())
            pos_tags = pos_tag(tokens)
            
            # Filter by POS and length
            for word, pos in pos_tags:
                if (pos in target_pos and 
                    len(word) > 2 and 
                    word not in self.stop_words and
                    word.isalpha()):
                    pos_keywords[pos].append(word)
        
        # Count frequencies and get top terms
        result = {}
        for pos, words in pos_keywords.items():
            word_counts = Counter(words)
            result[pos] = word_counts.most_common(20)
        
        return result
    
    def extract_named_entities(self, texts):
        """Extract named entities (restaurants, locations, people)"""
        
        if not self.nlp:
            print("⚠️ spaCy not available - skipping named entity extraction")
            return {}
        
        print(f"🏢 Extracting named entities...")
        
        entities_by_type = defaultdict(list)
        
        for text in texts:
            if pd.isna(text) or not text:
                continue
            
            doc = self.nlp(str(text))
            
            for ent in doc.ents:
                # Filter relevant entity types for restaurants
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LOC', 'MONEY', 'DATE']:
                    entities_by_type[ent.label_].append(ent.text.lower())
        
        # Count frequencies
        result = {}
        for ent_type, entities in entities_by_type.items():
            entity_counts = Counter(entities)
            result[ent_type] = entity_counts.most_common(10)
        
        return result
    
    def extract_sentiment_phrases(self, texts):
        """Extract sentiment-bearing phrases and expressions"""
        
        print(f"😊 Extracting sentiment phrases...")
        
        found_phrases = {
            'positive': [],
            'negative': []
        }
        
        for text in texts:
            if pd.isna(text) or not text:
                continue
            
            text_lower = str(text).lower()
            
            # Look for predefined sentiment phrases
            for sentiment, phrases in self.sentiment_phrases.items():
                for phrase in phrases:
                    if phrase in text_lower:
                        found_phrases[sentiment].append(phrase)
        
        # Count frequencies
        result = {}
        for sentiment, phrases in found_phrases.items():
            phrase_counts = Counter(phrases)
            result[sentiment] = phrase_counts.most_common(10)
        
        return result
    
    def extract_category_keywords(self, texts):
        """Extract keywords by restaurant categories"""
        
        print(f"🍽️ Extracting category-specific keywords...")
        
        category_matches = defaultdict(list)
        
        for text in texts:
            if pd.isna(text) or not text:
                continue
            
            text_lower = str(text).lower()
            words = word_tokenize(text_lower)
            
            # Match against category keywords
            for category, category_words in self.keyword_categories.items():
                for word in words:
                    if word in category_words:
                        category_matches[category].append(word)
        
        # Count frequencies
        result = {}
        for category, words in category_matches.items():
            word_counts = Counter(words)
            result[category] = word_counts.most_common(10)
        
        return result
    
    def extract_aspect_based_keywords(self, texts):
        """Extract keywords for specific restaurant aspects"""
        
        print(f"🎯 Extracting aspect-based keywords...")
        
        aspects = {
            'food': r'\b(?:food|dish|meal|cuisine|menu|taste|flavor)\b',
            'service': r'\b(?:service|waiter|waitress|server|staff)\b',
            'ambiance': r'\b(?:atmosphere|ambiance|environment|setting)\b',
            'price': r'\b(?:price|cost|expensive|cheap|value|money)\b'
        }
        
        aspect_keywords = defaultdict(list)
        
        for text in texts:
            if pd.isna(text) or not text:
                continue
            
            text_str = str(text)
            sentences = sent_tokenize(text_str)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check which aspects are mentioned in this sentence
                for aspect, pattern in aspects.items():
                    if re.search(pattern, sentence_lower):
                        # Extract adjectives from this sentence
                        tokens = word_tokenize(sentence_lower)
                        pos_tags = pos_tag(tokens)
                        
                        for word, pos in pos_tags:
                            if (pos.startswith('JJ') and  # Adjectives
                                len(word) > 2 and 
                                word not in self.stop_words):
                                aspect_keywords[aspect].append(word)
        
        # Count frequencies
        result = {}
        for aspect, words in aspect_keywords.items():
            word_counts = Counter(words)
            result[aspect] = word_counts.most_common(10)
        
        return result
    
    def extract_bigram_phrases(self, texts, min_freq=2):
        """Extract meaningful bigram phrases"""
        
        print(f"📝 Extracting bigram phrases...")
        
        all_bigrams = []
        
        for text in texts:
            if pd.isna(text) or not text:
                continue
            
            # Clean and tokenize
            text_clean = self._clean_text_for_ngrams(str(text))
            tokens = word_tokenize(text_clean.lower())
            
            # Filter tokens
            filtered_tokens = []
            for token in tokens:
                if (len(token) > 2 and 
                    token not in self.stop_words and 
                    token.isalpha()):
                    filtered_tokens.append(token)
            
            # Generate bigrams
            bigrams = list(ngrams(filtered_tokens, 2))
            all_bigrams.extend(bigrams)
        
        # Count and filter
        bigram_counts = Counter(all_bigrams)
        filtered_bigrams = [(bg, count) for bg, count in bigram_counts.items() if count >= min_freq]
        filtered_bigrams.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to readable format
        result = [(' '.join(bg), count) for bg, count in filtered_bigrams[:20]]
        
        return result
    
    def _preprocess_texts(self, texts):
        """Preprocess texts for keyword extraction"""
        
        processed = []
        
        for text in texts:
            if pd.isna(text) or not text:
                processed.append("")
                continue
            
            # Clean text
            text_clean = self._clean_text_for_ngrams(str(text))
            processed.append(text_clean)
        
        return processed
    
    def _clean_text_for_ngrams(self, text):
        """Clean text specifically for n-gram extraction"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_review_keywords(self, df, text_column='review_text'):
        """Comprehensive keyword analysis of review dataset"""
        
        print(f"🔍 COMPREHENSIVE KEYWORD ANALYSIS")
        print("=" * 50)
        
        texts = df[text_column].fillna('').tolist()
        
        results = {
            'dataset_size': len(texts),
            'tfidf_keywords': self.extract_tfidf_keywords(texts),
            'pos_keywords': self.extract_pos_keywords(texts),
            'named_entities': self.extract_named_entities(texts),
            'sentiment_phrases': self.extract_sentiment_phrases(texts),
            'category_keywords': self.extract_category_keywords(texts),
            'aspect_keywords': self.extract_aspect_based_keywords(texts),
            'bigram_phrases': self.extract_bigram_phrases(texts)
        }
        
        # Generate insights
        insights = self._generate_keyword_insights(results)
        results['insights'] = insights
        
        return results
    
    def _generate_keyword_insights(self, results):
        """Generate insights from keyword analysis"""
        
        insights = {}
        
        # Top categories mentioned
        category_totals = {}
        for category, keywords in results['category_keywords'].items():
            total_mentions = sum(count for _, count in keywords)
            category_totals[category] = total_mentions
        
        insights['most_discussed_category'] = max(category_totals, key=category_totals.get) if category_totals else 'none'
        
        # Sentiment balance
        pos_phrases = len(results['sentiment_phrases'].get('positive', []))
        neg_phrases = len(results['sentiment_phrases'].get('negative', []))
        insights['sentiment_balance'] = {
            'positive_phrases': pos_phrases,
            'negative_phrases': neg_phrases,
            'sentiment_ratio': pos_phrases / max(neg_phrases, 1)
        }
        
        # Most common aspects
        aspect_totals = {}
        for aspect, keywords in results['aspect_keywords'].items():
            total_mentions = sum(count for _, count in keywords)
            aspect_totals[aspect] = total_mentions
        
        insights['top_aspects'] = sorted(aspect_totals.items(), key=lambda x: x[1], reverse=True)
        
        return insights
    
    def get_review_keywords(self, text, top_k=5):
        """Extract keywords from a single review"""
        
        # Quick TF-IDF for single text (compare against common words)
        blob = TextBlob(text)
        words = blob.words
        
        # Filter and score words
        word_scores = {}
        for word in words:
            word_lower = word.lower()
            if (len(word_lower) > 2 and 
                word_lower not in self.stop_words and
                word_lower.isalpha()):
                
                # Simple scoring: length + frequency in categories
                score = len(word_lower)
                
                # Boost if it's a category word
                for category, category_words in self.keyword_categories.items():
                    if word_lower in category_words:
                        score += 10
                        break
                
                word_scores[word_lower] = score
        
        # Get top keywords
        top_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [word for word, score in top_keywords]


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Restaurant Keyword Extractor...")
    
    # Sample restaurant reviews
    sample_reviews = [
        "The pizza was absolutely delicious with fresh mozzarella and perfect crust. Our waiter was very attentive.",
        "Great atmosphere for a romantic dinner. The pasta was amazing and reasonably priced.",
        "Terrible service and overpriced food. The steak was overcooked and the staff was rude.",
        "Perfect location downtown with easy parking. The sushi was fresh and the service was quick.",
        "Cozy ambiance with beautiful lighting. The wine selection was excellent and portions were generous.",
        "Disappointing experience. The soup was cold and the restaurant was too noisy for conversation.",
        "Highly recommend this place! The chicken was tender and flavorful, definitely coming back.",
        "Worst restaurant ever. Poor quality food and the manager was unprofessional.",
        "Amazing dessert and fantastic coffee. Great value for money and wonderful atmosphere.",
        "The seafood was fresh and the outdoor patio has a beautiful view of the city."
    ]
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'review_text': sample_reviews,
        'rating': [5, 4, 1, 4, 5, 2, 5, 1, 4, 5]
    })
    
    # Initialize extractor
    extractor = RestaurantKeywordExtractor()
    
    # Analyze keywords
    results = extractor.analyze_review_keywords(test_df)
    
    print(f"\n📊 Keyword Analysis Results:")
    print(f"   Dataset size: {results['dataset_size']}")
    print(f"   Most discussed category: {results['insights']['most_discussed_category']}")
    print(f"   Sentiment ratio: {results['insights']['sentiment_balance']['sentiment_ratio']:.2f}")
    
    print(f"\n🔝 Top TF-IDF Keywords:")
    for word, score in results['tfidf_keywords'][:5]:
        print(f"     {word}: {score:.3f}")
    
    print(f"\n🍽️ Food Quality Keywords:")
    food_keywords = results['category_keywords'].get('food_quality', [])
    for word, count in food_keywords[:5]:
        print(f"     {word}: {count}")
    
    print(f"\n📝 Top Bigram Phrases:")
    for phrase, count in results['bigram_phrases'][:5]:
        print(f"     '{phrase}': {count}")
    
    # Test single review
    test_review = "The pasta was incredibly delicious and our server was very professional."
    keywords = extractor.get_review_keywords(test_review)
    print(f"\n🔍 Single Review Keywords:")
    print(f"   Review: '{test_review}'")
    print(f"   Keywords: {keywords}")
    
    print("\n✅ Keyword Extractor test complete!")
