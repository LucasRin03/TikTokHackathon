#!/usr/bin/env python3
"""
Advanced Topic Modeling for Restaurant Reviews
Discovers hidden themes and patterns in review text
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Word cloud for visualization
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠️ WordCloud not available - install with: pip install wordcloud")

class RestaurantTopicModeler:
    """
    Advanced topic modeling specifically designed for restaurant reviews
    Identifies themes like food quality, service, ambiance, value, etc.
    """
    
    def __init__(self, n_topics=8):
        print("🎯 Initializing Restaurant Topic Modeler...")
        
        self.n_topics = n_topics
        self.stop_words = set(stopwords.words('english'))
        
        # Add domain-specific stop words
        restaurant_stopwords = {
            'restaurant', 'place', 'food', 'good', 'great', 'nice', 'really',
            'went', 'go', 'get', 'got', 'would', 'could', 'like', 'one',
            'time', 'back', 'way', 'well', 'much', 'also', 'even', 'think'
        }
        self.stop_words.update(restaurant_stopwords)
        
        # Initialize models
        self.lda_model = None
        self.nmf_model = None
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
        # Topic themes (will be automatically discovered)
        self.topic_themes = {}
        
        print("✅ Topic Modeler ready!")
    
    def preprocess_for_topics(self, texts):
        """Specialized preprocessing for topic modeling"""
        
        processed_texts = []
        
        for text in texts:
            if pd.isna(text) or not text:
                processed_texts.append("")
                continue
            
            text = str(text).lower()
            
            # Remove special characters but keep important punctuation context
            text = re.sub(r'[^\w\s\!\?]', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Filter tokens
            filtered_tokens = []
            for token in tokens:
                if (len(token) > 2 and 
                    token not in self.stop_words and 
                    token.isalpha() and
                    not token.isdigit()):
                    filtered_tokens.append(token)
            
            processed_texts.append(' '.join(filtered_tokens))
        
        return processed_texts
    
    def fit_lda_topics(self, texts, n_topics=None):
        """Fit Latent Dirichlet Allocation topic model"""
        
        if n_topics is None:
            n_topics = self.n_topics
        
        print(f"🔍 Fitting LDA model with {n_topics} topics...")
        
        # Preprocess texts
        processed_texts = self.preprocess_for_topics(texts)
        
        # Create count vectorizer (LDA works better with raw counts)
        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        count_matrix = self.count_vectorizer.fit_transform(processed_texts)
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        
        lda_topics = self.lda_model.fit_transform(count_matrix)
        
        # Extract topic themes
        feature_names = self.count_vectorizer.get_feature_names_out()
        self._extract_topic_themes(self.lda_model, feature_names, method='LDA')
        
        return lda_topics
    
    def fit_nmf_topics(self, texts, n_topics=None):
        """Fit Non-negative Matrix Factorization topic model"""
        
        if n_topics is None:
            n_topics = self.n_topics
        
        print(f"🔍 Fitting NMF model with {n_topics} topics...")
        
        # Preprocess texts
        processed_texts = self.preprocess_for_topics(texts)
        
        # Create TF-IDF vectorizer (NMF works well with TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Fit NMF model
        self.nmf_model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        nmf_topics = self.nmf_model.fit_transform(tfidf_matrix)
        
        # Extract topic themes
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self._extract_topic_themes(self.nmf_model, feature_names, method='NMF')
        
        return nmf_topics
    
    def _extract_topic_themes(self, model, feature_names, method='LDA'):
        """Extract and interpret topic themes"""
        
        print(f"🎭 Extracting {method} topic themes...")
        
        # Predefined restaurant themes for interpretation
        restaurant_themes = {
            'food_quality': ['delicious', 'tasty', 'fresh', 'flavor', 'quality', 'amazing', 'excellent', 'best', 'perfect', 'wonderful'],
            'service': ['service', 'staff', 'waiter', 'server', 'friendly', 'attentive', 'helpful', 'rude', 'slow', 'quick'],
            'ambiance': ['atmosphere', 'ambiance', 'music', 'noise', 'quiet', 'romantic', 'cozy', 'comfortable', 'crowded', 'clean'],
            'value': ['price', 'expensive', 'cheap', 'worth', 'value', 'money', 'cost', 'affordable', 'overpriced', 'reasonable'],
            'experience': ['experience', 'visit', 'dinner', 'lunch', 'breakfast', 'date', 'family', 'celebration', 'special', 'occasion'],
            'location': ['location', 'parking', 'downtown', 'convenient', 'area', 'neighborhood', 'accessible', 'view', 'outside', 'inside']
        }
        
        topics_info = {}
        
        for topic_idx, topic in enumerate(model.components_):
            # Get top words for this topic
            top_words_idx = topic.argsort()[-15:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [topic[i] for i in top_words_idx]
            
            # Attempt to classify this topic
            theme_scores = {}
            for theme, theme_words in restaurant_themes.items():
                score = sum(1 for word in top_words[:10] if word in theme_words)
                theme_scores[theme] = score
            
            # Assign best matching theme
            best_theme = max(theme_scores, key=theme_scores.get) if max(theme_scores.values()) > 0 else 'other'
            
            topics_info[f'topic_{topic_idx}'] = {
                'theme': best_theme,
                'top_words': top_words[:10],
                'top_scores': top_scores[:10],
                'theme_confidence': max(theme_scores.values()) / 10.0
            }
            
            print(f"   Topic {topic_idx} ({best_theme}): {', '.join(top_words[:5])}")
        
        self.topic_themes[method] = topics_info
    
    def predict_review_topics(self, texts, method='LDA'):
        """Predict topic distributions for new reviews"""
        
        if method == 'LDA' and self.lda_model is None:
            raise ValueError("LDA model not fitted. Call fit_lda_topics first.")
        if method == 'NMF' and self.nmf_model is None:
            raise ValueError("NMF model not fitted. Call fit_nmf_topics first.")
        
        # Preprocess texts
        processed_texts = self.preprocess_for_topics(texts)
        
        if method == 'LDA':
            # Transform using count vectorizer
            count_matrix = self.count_vectorizer.transform(processed_texts)
            topic_distributions = self.lda_model.transform(count_matrix)
        else:  # NMF
            # Transform using TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.transform(processed_texts)
            topic_distributions = self.nmf_model.transform(tfidf_matrix)
        
        return topic_distributions
    
    def get_review_insights(self, text, method='LDA'):
        """Get detailed insights for a single review"""
        
        topic_dist = self.predict_review_topics([text], method)[0]
        
        # Get dominant topics
        dominant_topics = []
        for i, score in enumerate(topic_dist):
            if score > 0.1:  # Threshold for significance
                topic_info = self.topic_themes[method][f'topic_{i}']
                dominant_topics.append({
                    'topic_id': i,
                    'theme': topic_info['theme'],
                    'score': score,
                    'top_words': topic_info['top_words'][:5]
                })
        
        # Sort by score
        dominant_topics.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'dominant_topics': dominant_topics,
            'topic_distribution': topic_dist
        }
    
    def analyze_dataset_topics(self, df, text_column='review_text'):
        """Analyze topics across entire dataset"""
        
        print(f"📊 Analyzing topics in {len(df)} reviews...")
        
        texts = df[text_column].fillna('').tolist()
        
        # Fit both models
        lda_topics = self.fit_lda_topics(texts)
        nmf_topics = self.fit_nmf_topics(texts)
        
        # Add topic assignments to dataframe
        result_df = df.copy()
        
        # LDA topics
        dominant_lda_topics = np.argmax(lda_topics, axis=1)
        result_df['lda_dominant_topic'] = dominant_lda_topics
        result_df['lda_topic_confidence'] = np.max(lda_topics, axis=1)
        
        # NMF topics
        dominant_nmf_topics = np.argmax(nmf_topics, axis=1)
        result_df['nmf_dominant_topic'] = dominant_nmf_topics
        result_df['nmf_topic_confidence'] = np.max(nmf_topics, axis=1)
        
        # Topic theme labels
        lda_themes = [self.topic_themes['LDA'][f'topic_{i}']['theme'] for i in dominant_lda_topics]
        nmf_themes = [self.topic_themes['NMF'][f'topic_{i}']['theme'] for i in dominant_nmf_topics]
        
        result_df['lda_theme'] = lda_themes
        result_df['nmf_theme'] = nmf_themes
        
        # Generate summary
        topic_summary = self._generate_topic_summary(result_df)
        
        return result_df, topic_summary
    
    def _generate_topic_summary(self, df):
        """Generate comprehensive topic analysis summary"""
        
        summary = {
            'total_reviews': len(df),
            'lda_topic_distribution': df['lda_theme'].value_counts().to_dict(),
            'nmf_topic_distribution': df['nmf_theme'].value_counts().to_dict(),
            'average_topic_confidence': {
                'lda': df['lda_topic_confidence'].mean(),
                'nmf': df['nmf_topic_confidence'].mean()
            }
        }
        
        # Topic-rating correlation
        if 'rating' in df.columns:
            theme_ratings = {}
            for theme in df['lda_theme'].unique():
                theme_df = df[df['lda_theme'] == theme]
                theme_ratings[theme] = {
                    'avg_rating': theme_df['rating'].mean(),
                    'count': len(theme_df)
                }
            summary['theme_ratings'] = theme_ratings
        
        return summary
    
    def create_topic_visualizations(self, df, save_path='data/processed/'):
        """Create visualizations for topic analysis"""
        
        print("📈 Creating topic visualizations...")
        
        # Topic distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # LDA theme distribution
        lda_counts = df['lda_theme'].value_counts()
        axes[0, 0].pie(lda_counts.values, labels=lda_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('LDA Topic Distribution')
        
        # NMF theme distribution
        nmf_counts = df['nmf_theme'].value_counts()
        axes[0, 1].pie(nmf_counts.values, labels=nmf_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('NMF Topic Distribution')
        
        # Topic confidence distributions
        axes[1, 0].hist(df['lda_topic_confidence'], bins=20, alpha=0.7, label='LDA')
        axes[1, 0].hist(df['nmf_topic_confidence'], bins=20, alpha=0.7, label='NMF')
        axes[1, 0].set_xlabel('Topic Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Topic Confidence Distribution')
        axes[1, 0].legend()
        
        # Theme vs Rating (if available)
        if 'rating' in df.columns:
            theme_ratings = df.groupby('lda_theme')['rating'].mean().sort_values(ascending=False)
            axes[1, 1].bar(range(len(theme_ratings)), theme_ratings.values)
            axes[1, 1].set_xticks(range(len(theme_ratings)))
            axes[1, 1].set_xticklabels(theme_ratings.index, rotation=45)
            axes[1, 1].set_ylabel('Average Rating')
            axes[1, 1].set_title('Average Rating by Topic Theme')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}topic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Word clouds for top topics
        if WORDCLOUD_AVAILABLE:
            self._create_topic_wordclouds(save_path)
        
        print(f"✅ Visualizations saved to {save_path}")
    
    def _create_topic_wordclouds(self, save_path):
        """Create word clouds for each topic"""
        
        for method in ['LDA', 'NMF']:
            if method not in self.topic_themes:
                continue
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, (topic_id, topic_info) in enumerate(self.topic_themes[method].items()):
                if i >= 8:  # Limit to 8 topics for visualization
                    break
                
                # Create word frequency dict
                words = topic_info['top_words']
                scores = topic_info['top_scores']
                word_freq = dict(zip(words, scores))
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(word_freq)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f"{topic_info['theme'].title()} (Topic {i})")
                axes[i].axis('off')
            
            # Hide unused subplots
            for j in range(i+1, 8):
                axes[j].axis('off')
            
            plt.suptitle(f'{method} Topic Word Clouds')
            plt.tight_layout()
            plt.savefig(f'{save_path}{method.lower()}_topic_wordclouds.png', dpi=300, bbox_inches='tight')
            plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Restaurant Topic Modeler...")
    
    # Sample restaurant reviews
    sample_reviews = [
        "The food was absolutely delicious and the service was outstanding. Our waiter was very attentive.",
        "Great atmosphere for a romantic dinner. The music was perfect and not too loud.",
        "Overpriced for what you get. The pasta was mediocre and cost way too much.",
        "Fast service and good value for money. Perfect for a quick lunch break.",
        "The restaurant is in a great location downtown with easy parking.",
        "Terrible experience. The staff was rude and the food was cold when it arrived.",
        "Beautiful ambiance with dim lighting and cozy seating. Perfect for date night.",
        "Fresh ingredients and amazing flavors. The chef really knows what they're doing.",
        "Loud and crowded. Hard to have a conversation during dinner.",
        "Great value for families. Large portions and reasonable prices."
    ]
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'review_text': sample_reviews,
        'rating': [5, 5, 2, 4, 4, 1, 5, 5, 2, 4]
    })
    
    # Initialize and run topic modeling
    modeler = RestaurantTopicModeler(n_topics=5)
    
    # Analyze topics
    result_df, summary = modeler.analyze_dataset_topics(test_df)
    
    print(f"\n📊 Topic Analysis Results:")
    print(f"   Total reviews: {summary['total_reviews']}")
    print(f"   LDA themes: {summary['lda_topic_distribution']}")
    print(f"   Average confidence: LDA={summary['average_topic_confidence']['lda']:.3f}")
    
    # Test individual review analysis
    test_review = "The food was incredible and our server was amazing, but it was quite expensive."
    insights = modeler.get_review_insights(test_review)
    
    print(f"\n🔍 Individual Review Analysis:")
    print(f"   Review: '{insights['text']}'")
    print(f"   Dominant topics:")
    for topic in insights['dominant_topics'][:3]:
        print(f"     {topic['theme']}: {topic['score']:.3f} ({', '.join(topic['top_words'])})")
    
    print("\n✅ Topic Modeler test complete!")
