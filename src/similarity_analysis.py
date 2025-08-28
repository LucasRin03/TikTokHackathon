#!/usr/bin/env python3
"""
Advanced Review Similarity Analysis for Duplicate/Fake Detection
Detects duplicate reviews, bot patterns, and suspicious review patterns
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import math
from datetime import datetime, timedelta

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy

# Fuzzy matching
from difflib import SequenceMatcher

class ReviewSimilarityAnalyzer:
    """
    Advanced similarity analysis for detecting:
    1. Duplicate reviews (exact and near-duplicate)
    2. Bot-generated reviews (pattern detection)
    3. Fake review campaigns (coordinated posting)
    4. Template-based reviews (suspicious patterns)
    """
    
    def __init__(self):
        print("🔍 Initializing Review Similarity Analyzer...")
        
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy for advanced text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except OSError:
            print("⚠️ spaCy model not found - some features disabled")
            self.nlp = None
        
        # Similarity thresholds
        self.thresholds = {
            'exact_duplicate': 0.95,        # Almost identical text
            'near_duplicate': 0.85,         # Very similar content
            'suspicious_similar': 0.75,     # Potentially template-based
            'bot_pattern': 0.70,            # Bot-like repetition
            'campaign_threshold': 3         # Min reviews for campaign detection
        }
        
        # Bot detection patterns
        self.bot_indicators = {
            'generic_phrases': [
                'great place', 'nice food', 'good service', 'will come back',
                'highly recommend', 'perfect location', 'amazing experience',
                'excellent quality', 'fantastic restaurant', 'love this place'
            ],
            'template_structures': [
                r'^(great|good|nice|amazing|excellent)\s+(food|service|place)',
                r'(recommend|suggest)\s+(this|it)\s+(place|restaurant)',
                r'(will|would)\s+(definitely|surely)\s+(come|go)\s+back',
                r'(best|worst)\s+(restaurant|food|service)\s+(ever|in)',
                r'(love|hate)\s+(this|the)\s+(place|restaurant|food)'
            ],
            'repetitive_patterns': [
                r'(.+)\s+\1',  # Word repetition
                r'!{2,}',      # Multiple exclamations
                r'\.{2,}',     # Multiple periods
                r'[A-Z]{3,}'   # Excessive caps
            ]
        }
        
        print("✅ Similarity Analyzer ready!")
    
    def compute_text_similarity(self, text1, text2, method='tfidf'):
        """Compute similarity between two texts using various methods"""
        
        if not text1 or not text2 or pd.isna(text1) or pd.isna(text2):
            return 0.0
        
        text1, text2 = str(text1), str(text2)
        
        if method == 'exact':
            return 1.0 if text1.lower().strip() == text2.lower().strip() else 0.0
        
        elif method == 'sequence':
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        elif method == 'tfidf':
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                return similarity
            except:
                return 0.0
        
        elif method == 'jaccard':
            # Jaccard similarity for word sets
            words1 = set(word_tokenize(text1.lower()))
            words2 = set(word_tokenize(text2.lower()))
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
        
        else:
            return self.compute_text_similarity(text1, text2, 'tfidf')
    
    def detect_duplicates(self, df, text_column='review_text', threshold=0.85):
        """Detect duplicate and near-duplicate reviews"""
        
        print(f"🔍 Detecting duplicates in {len(df)} reviews...")
        
        reviews = df[text_column].fillna('').tolist()
        duplicate_groups = []
        processed = set()
        
        for i, review1 in enumerate(reviews):
            if i in processed:
                continue
            
            current_group = [i]
            
            for j, review2 in enumerate(reviews[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.compute_text_similarity(review1, review2, 'tfidf')
                
                if similarity >= threshold:
                    current_group.append(j)
                    processed.add(j)
            
            if len(current_group) > 1:
                duplicate_groups.append({
                    'indices': current_group,
                    'count': len(current_group),
                    'similarity_scores': [
                        self.compute_text_similarity(reviews[current_group[0]], reviews[idx], 'tfidf')
                        for idx in current_group[1:]
                    ]
                })
                processed.update(current_group)
        
        return duplicate_groups
    
    def detect_bot_patterns(self, df, text_column='review_text'):
        """Detect bot-generated review patterns"""
        
        print(f"🤖 Detecting bot patterns in {len(df)} reviews...")
        
        bot_scores = []
        
        for idx, review in enumerate(df[text_column].fillna('')):
            review_str = str(review).lower()
            bot_score = 0
            
            # Check for generic phrases
            generic_count = sum(1 for phrase in self.bot_indicators['generic_phrases'] 
                              if phrase in review_str)
            bot_score += min(generic_count * 0.2, 1.0)
            
            # Check for template structures
            template_count = sum(1 for pattern in self.bot_indicators['template_structures']
                                if re.search(pattern, review_str))
            bot_score += min(template_count * 0.3, 1.0)
            
            # Check for repetitive patterns
            repetitive_count = sum(1 for pattern in self.bot_indicators['repetitive_patterns']
                                  if re.search(pattern, review_str))
            bot_score += min(repetitive_count * 0.25, 1.0)
            
            # Check review length (very short or very long can be suspicious)
            word_count = len(word_tokenize(review_str))
            if word_count < 5 or word_count > 200:
                bot_score += 0.3
            
            # Check for excessive punctuation
            punct_ratio = sum(1 for char in review_str if char in '!?.,;:') / max(len(review_str), 1)
            if punct_ratio > 0.1:
                bot_score += 0.2
            
            bot_scores.append(min(bot_score, 1.0))
        
        return bot_scores
    
    def detect_review_campaigns(self, df, text_column='review_text', user_column=None, date_column=None):
        """Detect coordinated fake review campaigns"""
        
        print(f"📊 Detecting review campaigns...")
        
        campaigns = []
        
        # Time-based clustering (if date column available)
        if date_column and date_column in df.columns:
            campaigns.extend(self._detect_temporal_campaigns(df, text_column, date_column))
        
        # User-based clustering (if user column available)
        if user_column and user_column in df.columns:
            campaigns.extend(self._detect_user_campaigns(df, text_column, user_column))
        
        # Content-based clustering
        campaigns.extend(self._detect_content_campaigns(df, text_column))
        
        return campaigns
    
    def _detect_temporal_campaigns(self, df, text_column, date_column):
        """Detect campaigns based on temporal patterns"""
        
        campaigns = []
        
        try:
            # Convert dates
            df_temp = df.copy()
            df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')
            
            # Group by day and find suspicious spikes
            daily_counts = df_temp.groupby(df_temp[date_column].dt.date).size()
            
            # Find days with unusually high review counts
            mean_daily = daily_counts.mean()
            std_daily = daily_counts.std()
            threshold = mean_daily + 2 * std_daily
            
            suspicious_days = daily_counts[daily_counts > threshold].index
            
            for day in suspicious_days:
                day_reviews = df_temp[df_temp[date_column].dt.date == day]
                
                if len(day_reviews) >= self.thresholds['campaign_threshold']:
                    # Check similarity within the day
                    similarities = self._compute_group_similarity(
                        day_reviews[text_column].tolist()
                    )
                    
                    if similarities['avg_similarity'] > 0.6:
                        campaigns.append({
                            'type': 'temporal_spike',
                            'date': day,
                            'review_count': len(day_reviews),
                            'avg_similarity': similarities['avg_similarity'],
                            'indices': day_reviews.index.tolist()
                        })
        
        except Exception as e:
            print(f"⚠️ Temporal campaign detection failed: {e}")
        
        return campaigns
    
    def _detect_user_campaigns(self, df, text_column, user_column):
        """Detect campaigns based on user patterns"""
        
        campaigns = []
        
        # Group by user
        user_groups = df.groupby(user_column)
        
        for user, group in user_groups:
            if len(group) >= self.thresholds['campaign_threshold']:
                # Check if user has multiple very similar reviews
                reviews = group[text_column].tolist()
                similarities = self._compute_group_similarity(reviews)
                
                if similarities['avg_similarity'] > 0.7:
                    campaigns.append({
                        'type': 'user_repetition',
                        'user': user,
                        'review_count': len(group),
                        'avg_similarity': similarities['avg_similarity'],
                        'indices': group.index.tolist()
                    })
        
        return campaigns
    
    def _detect_content_campaigns(self, df, text_column):
        """Detect campaigns based on content similarity clustering"""
        
        campaigns = []
        
        try:
            reviews = df[text_column].fillna('').tolist()
            
            if len(reviews) < 10:  # Need minimum reviews for clustering
                return campaigns
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            
            vectors = vectorizer.fit_transform(reviews)
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(
                eps=0.3,  # Distance threshold
                min_samples=self.thresholds['campaign_threshold'],
                metric='cosine'
            )
            
            clusters = clustering.fit_predict(vectors)
            
            # Analyze clusters
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Noise cluster
                    continue
                
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                
                if len(cluster_indices) >= self.thresholds['campaign_threshold']:
                    cluster_reviews = [reviews[i] for i in cluster_indices]
                    similarities = self._compute_group_similarity(cluster_reviews)
                    
                    campaigns.append({
                        'type': 'content_cluster',
                        'cluster_id': cluster_id,
                        'review_count': len(cluster_indices),
                        'avg_similarity': similarities['avg_similarity'],
                        'indices': [df.index[i] for i in cluster_indices]
                    })
        
        except Exception as e:
            print(f"⚠️ Content campaign detection failed: {e}")
        
        return campaigns
    
    def _compute_group_similarity(self, reviews):
        """Compute average similarity within a group of reviews"""
        
        if len(reviews) < 2:
            return {'avg_similarity': 0.0, 'max_similarity': 0.0, 'min_similarity': 0.0}
        
        similarities = []
        
        for i in range(len(reviews)):
            for j in range(i+1, len(reviews)):
                sim = self.compute_text_similarity(reviews[i], reviews[j], 'tfidf')
                similarities.append(sim)
        
        return {
            'avg_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities)
        }
    
    def analyze_review_authenticity(self, df, text_column='review_text', user_column=None, date_column=None):
        """Comprehensive authenticity analysis"""
        
        print(f"🔒 COMPREHENSIVE AUTHENTICITY ANALYSIS")
        print("=" * 50)
        
        results = {
            'total_reviews': len(df),
            'duplicate_analysis': {},
            'bot_analysis': {},
            'campaign_analysis': {},
            'authenticity_scores': []
        }
        
        # 1. Duplicate detection
        duplicates = self.detect_duplicates(df, text_column)
        results['duplicate_analysis'] = {
            'duplicate_groups': len(duplicates),
            'total_duplicates': sum(group['count'] for group in duplicates),
            'duplicate_rate': sum(group['count'] for group in duplicates) / len(df)
        }
        
        # 2. Bot pattern detection
        bot_scores = self.detect_bot_patterns(df, text_column)
        results['bot_analysis'] = {
            'avg_bot_score': np.mean(bot_scores),
            'high_bot_score_count': sum(1 for score in bot_scores if score > 0.7),
            'bot_rate': sum(1 for score in bot_scores if score > 0.7) / len(df)
        }
        
        # 3. Campaign detection
        campaigns = self.detect_review_campaigns(df, text_column, user_column, date_column)
        results['campaign_analysis'] = {
            'campaign_count': len(campaigns),
            'reviews_in_campaigns': sum(campaign['review_count'] for campaign in campaigns),
            'campaign_rate': sum(campaign['review_count'] for campaign in campaigns) / len(df)
        }
        
        # 4. Overall authenticity scores
        authenticity_scores = []
        duplicate_indices = set()
        for group in duplicates:
            duplicate_indices.update(group['indices'])
        
        for i in range(len(df)):
            score = 1.0  # Start with authentic
            
            # Reduce score for duplicates
            if i in duplicate_indices:
                score -= 0.4
            
            # Reduce score for bot patterns
            score -= bot_scores[i] * 0.3
            
            # Reduce score if part of campaign
            in_campaign = any(i in campaign['indices'] for campaign in campaigns)
            if in_campaign:
                score -= 0.3
            
            authenticity_scores.append(max(score, 0.0))
        
        results['authenticity_scores'] = authenticity_scores
        results['avg_authenticity'] = np.mean(authenticity_scores)
        results['suspicious_count'] = sum(1 for score in authenticity_scores if score < 0.5)
        
        return results
    
    def generate_similarity_report(self, df, text_column='review_text', user_column=None, date_column=None):
        """Generate comprehensive similarity analysis report"""
        
        print(f"📋 GENERATING SIMILARITY ANALYSIS REPORT")
        print("=" * 50)
        
        # Run comprehensive analysis
        results = self.analyze_review_authenticity(df, text_column, user_column, date_column)
        
        # Create summary report
        report = {
            'summary': {
                'total_reviews': results['total_reviews'],
                'authenticity_rate': results['avg_authenticity'],
                'suspicious_reviews': results['suspicious_count'],
                'quality_score': 1.0 - (results['duplicate_analysis']['duplicate_rate'] + 
                                      results['bot_analysis']['bot_rate'] + 
                                      results['campaign_analysis']['campaign_rate']) / 3
            },
            'duplicate_analysis': results['duplicate_analysis'],
            'bot_analysis': results['bot_analysis'],
            'campaign_analysis': results['campaign_analysis'],
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Duplicate recommendations
        if results['duplicate_analysis']['duplicate_rate'] > 0.1:
            recommendations.append("High duplicate rate detected - implement duplicate filtering")
        
        # Bot recommendations
        if results['bot_analysis']['bot_rate'] > 0.2:
            recommendations.append("Significant bot activity detected - enhance bot detection")
        
        # Campaign recommendations
        if results['campaign_analysis']['campaign_rate'] > 0.15:
            recommendations.append("Review campaigns detected - investigate coordinated posting")
        
        # Overall quality
        if results['avg_authenticity'] < 0.7:
            recommendations.append("Overall authenticity is low - comprehensive review needed")
        
        if not recommendations:
            recommendations.append("Review quality appears good - continue monitoring")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Review Similarity Analyzer...")
    
    # Sample reviews with duplicates and bot patterns
    sample_reviews = [
        "Amazing food and great service! Will definitely come back.",
        "Amazing food and excellent service! Will definitely return.",  # Near duplicate
        "Great place, nice food, good service. Highly recommend!",     # Bot-like
        "Great place, nice food, good service. Highly recommend!",     # Exact duplicate
        "The pasta was delicious and the ambiance was perfect for our anniversary dinner.",
        "Terrible experience. Food was cold and service was awful.",
        "AMAZING!!! BEST RESTAURANT EVER!!! MUST TRY!!!",            # Bot-like
        "Great place nice food good service highly recommend",         # Bot-like
        "Beautiful restaurant with excellent Italian cuisine and attentive staff.",
        "Great place, nice food, good service. Will come back!"       # Template-like
    ]
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'review_text': sample_reviews,
        'rating': [5, 5, 4, 4, 5, 1, 5, 4, 5, 4],
        'user_id': ['user1', 'user2', 'user3', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9']
    })
    
    # Initialize analyzer
    analyzer = ReviewSimilarityAnalyzer()
    
    # Generate report
    report = analyzer.generate_similarity_report(test_df, user_column='user_id')
    
    print(f"\n📊 SIMILARITY ANALYSIS RESULTS:")
    print(f"   Total reviews: {report['summary']['total_reviews']}")
    print(f"   Authenticity rate: {report['summary']['authenticity_rate']:.2f}")
    print(f"   Suspicious reviews: {report['summary']['suspicious_reviews']}")
    print(f"   Quality score: {report['summary']['quality_score']:.2f}")
    
    print(f"\n🔍 DUPLICATE ANALYSIS:")
    print(f"   Duplicate groups: {report['duplicate_analysis']['duplicate_groups']}")
    print(f"   Duplicate rate: {report['duplicate_analysis']['duplicate_rate']:.2f}")
    
    print(f"\n🤖 BOT ANALYSIS:")
    print(f"   Average bot score: {report['bot_analysis']['avg_bot_score']:.2f}")
    print(f"   Bot rate: {report['bot_analysis']['bot_rate']:.2f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   • {rec}")
    
    print("\n✅ Similarity Analyzer test complete!")
