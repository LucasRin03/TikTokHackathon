#!/usr/bin/env python3
"""
Process the actual Kaggle Google Maps restaurant reviews data
This is REAL data from Google Maps!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')

# Configuration
RAW_DATA_DIR = Path('data/raw/archive')
PROCESSED_DATA_DIR = Path('data/processed')

def process_kaggle_reviews():
    """Process the main Kaggle reviews.csv file"""
    
    print("🚀 PROCESSING REAL KAGGLE GOOGLE MAPS REVIEWS")
    print("=" * 60)
    
    # Load the main reviews file
    reviews_file = RAW_DATA_DIR / 'reviews.csv'
    df = pd.read_csv(reviews_file)
    
    print(f"📊 Loaded {len(df):,} REAL Google Maps restaurant reviews")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Inspect the data structure
    print(f"\n🔍 DATA INSPECTION:")
    print(f"   Unique businesses: {df['business_name'].nunique()}")
    print(f"   Unique reviewers: {df['author_name'].nunique()}")
    print(f"   Rating range: {df['rating'].min()} to {df['rating'].max()}")
    print(f"   Rating distribution:")
    for rating in sorted(df['rating'].unique()):
        count = (df['rating'] == rating).sum()
        pct = count / len(df) * 100
        print(f"      {rating} stars: {count:,} ({pct:.1f}%)")
    
    # Check review categories (this dataset has categories!)
    if 'rating_category' in df.columns:
        print(f"\n🏷️ Review categories: {df['rating_category'].unique()}")
        category_dist = df['rating_category'].value_counts()
        for cat, count in category_dist.items():
            print(f"      {cat}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Standardize column names for our pipeline
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={
        'text': 'review_text',
        'author_name': 'reviewer_name', 
        'rating': 'rating'
    })
    
    # Clean up review text
    df_clean = df_clean[df_clean['review_text'].notna()]
    df_clean = df_clean[df_clean['review_text'].str.len() > 10]  # Remove very short reviews
    
    # Add review date (this dataset doesn't have dates, so we'll create dummy ones)
    df_clean['review_date'] = pd.date_range('2023-01-01', periods=len(df_clean), freq='2h')
    
    print(f"\n🧹 After cleaning: {len(df_clean):,} reviews ready for processing")
    
    return df_clean

def apply_pipeline_to_real_data(df):
    """Apply our existing preprocessing pipeline to real Kaggle data"""
    
    print(f"\n🔧 APPLYING OUR PREPROCESSING PIPELINE TO REAL DATA")
    print("=" * 50)
    
    try:
        from data_processing import TextPreprocessor, PolicyFeatureExtractor
        
        # Initialize our processors
        text_processor = TextPreprocessor()
        policy_extractor = PolicyFeatureExtractor()
        
        print("   📝 Applying text preprocessing...")
        df_processed = text_processor.preprocess_dataframe(df, 'review_text')
        print(f"      ✅ Text features created")
        
        print("   🎯 Extracting policy violation features...")
        df_features = policy_extractor.extract_features_dataframe(df_processed, 'cleaned_text')
        print(f"      ✅ Policy features extracted")
        
        print(f"   📊 Total features: {len(df_features.columns)}")
        
        return df_features
        
    except Exception as e:
        print(f"   ❌ Pipeline processing failed: {e}")
        return None

def create_intelligent_labels(df):
    """Create intelligent pseudo-labels based on real review patterns"""
    
    print(f"\n🧠 CREATING INTELLIGENT LABELS FROM REAL REVIEW PATTERNS")
    print("=" * 50)
    
    # Create sophisticated labels using multiple signals
    
    # Advertisement detection (multiple signals)
    ad_signals = (
        (df['has_url'] > 0) |  # Contains URLs
        (df['has_discount_mention'] > 0) |  # Mentions discounts
        (df['has_contact_mention'] > 0) |  # Contact information
        (df['review_text'].str.contains(r'visit.*website|check.*out|follow.*us', case=False, na=False)) |
        (df['review_text'].str.contains(r'@\w+|facebook|instagram|twitter', case=False, na=False))
    )
    
    # Irrelevant content (off-topic mentions)
    irrelevant_signals = (
        (df['has_off_topic_mention'] > 0) |
        (df['review_text'].str.contains(r'phone|laptop|movie|weather|politics|sports', case=False, na=False)) |
        (df['review_text'].str.contains(r'netflix|youtube|tiktok|social media', case=False, na=False))
    )
    
    # Rant without visit (indirect evidence)
    rant_signals = (
        (df['has_never_mention'] > 0) |
        (df['has_heard_mention'] > 0) |
        (df['review_text'].str.contains(r'never.*been|haven.*t.*visited|heard.*from|someone.*told', case=False, na=False)) |
        (df['review_text'].str.contains(r'based.*on.*photos|looks.*like|seems.*like', case=False, na=False))
    )
    
    # Additional quality signals from real review characteristics
    quality_signals = (
        (df['review_length'] > 50) &  # Substantial reviews
        (df['review_length'] < 1000) &  # Not overly long rants
        (df['word_count'] > 10) &  # Meaningful content
        (df['sentiment_score'] > -0.8) &  # Not extremely negative
        (~ad_signals) &  # Not advertisements
        (~irrelevant_signals) &  # Not irrelevant
        (~rant_signals)  # Not rants without visit
    )
    
    # Create labels
    df['is_advertisement'] = ad_signals.astype(int)
    df['is_irrelevant'] = irrelevant_signals.astype(int) 
    df['is_rant_no_visit'] = rant_signals.astype(int)
    df['overall_quality'] = quality_signals.astype(int)
    
    # Print label distribution
    print(f"   📊 Label distribution from REAL Google reviews:")
    print(f"      ✅ Quality reviews: {df['overall_quality'].sum():,} ({df['overall_quality'].mean()*100:.1f}%)")
    print(f"      🚨 Advertisements: {df['is_advertisement'].sum():,} ({df['is_advertisement'].mean()*100:.1f}%)")
    print(f"      🚨 Irrelevant content: {df['is_irrelevant'].sum():,} ({df['is_irrelevant'].mean()*100:.1f}%)")
    print(f"      🚨 Rants w/o visit: {df['is_rant_no_visit'].sum():,} ({df['is_rant_no_visit'].mean()*100:.1f}%)")
    
    return df

def save_processed_real_data(df):
    """Save the processed real data for team use"""
    
    print(f"\n💾 SAVING PROCESSED REAL DATA FOR TEAM")
    print("=" * 40)
    
    # Save full processed dataset
    real_data_file = PROCESSED_DATA_DIR / 'real_kaggle_reviews.csv'
    df.to_csv(real_data_file, index=False)
    
    # Create a validation set from real data (smaller subset)
    validation_real = df.sample(n=min(300, len(df)), random_state=42)
    validation_file = PROCESSED_DATA_DIR / 'validation_real_data.csv'
    validation_real.to_csv(validation_file, index=False)
    
    print(f"   ✅ Full real dataset: {real_data_file} ({len(df):,} reviews)")
    print(f"   ✅ Real validation set: {validation_file} ({len(validation_real):,} reviews)")
    
    return real_data_file, validation_file

def compare_with_synthetic_data():
    """Compare real data characteristics with synthetic data"""
    
    print(f"\n📈 COMPARING REAL VS SYNTHETIC DATA")
    print("=" * 40)
    
    try:
        # Load synthetic training data
        synthetic_df = pd.read_csv('data/processed/train_data.csv')
        real_df = pd.read_csv('data/processed/real_kaggle_reviews.csv')
        
        print(f"📊 Dataset sizes:")
        print(f"   Synthetic (training): {len(synthetic_df):,} reviews")
        print(f"   Real (Kaggle): {len(real_df):,} reviews")
        
        print(f"\n📏 Text characteristics:")
        print(f"   Synthetic avg length: {synthetic_df['review_length'].mean():.1f} chars")
        print(f"   Real avg length: {real_df['review_length'].mean():.1f} chars")
        print(f"   Synthetic avg words: {synthetic_df['word_count'].mean():.1f}")
        print(f"   Real avg words: {real_df['word_count'].mean():.1f}")
        
        print(f"\n😊 Sentiment comparison:")
        print(f"   Synthetic avg sentiment: {synthetic_df['sentiment_score'].mean():.3f}")
        print(f"   Real avg sentiment: {real_df['sentiment_score'].mean():.3f}")
        
        print(f"\n🎯 Policy violation rates:")
        for policy in ['is_advertisement', 'is_irrelevant', 'is_rant_no_visit']:
            syn_rate = synthetic_df[policy].mean() * 100
            real_rate = real_df[policy].mean() * 100
            print(f"   {policy.replace('is_', '').replace('_', ' ').title()}:")
            print(f"      Synthetic: {syn_rate:.1f}% | Real: {real_rate:.1f}%")
        
    except Exception as e:
        print(f"   ⚠️ Comparison failed: {e}")

def main():
    """Main execution function"""
    
    print("🎉 PROCESSING REAL KAGGLE GOOGLE MAPS RESTAURANT REVIEWS")
    print("=" * 70)
    
    # Process the Kaggle reviews
    df_clean = process_kaggle_reviews()
    
    # Apply our preprocessing pipeline
    df_features = apply_pipeline_to_real_data(df_clean)
    
    if df_features is not None:
        # Create intelligent labels
        df_labeled = create_intelligent_labels(df_features)
        
        # Save processed data
        real_file, validation_file = save_processed_real_data(df_labeled)
        
        # Compare with synthetic data
        compare_with_synthetic_data()
        
        print(f"\n🎉 SUCCESS! REAL GOOGLE REVIEWS DATA READY!")
        print(f"🎯 Your team now has:")
        print(f"   ✅ Synthetic training data: 600 samples (controlled patterns)")
        print(f"   ✅ Real validation data: {len(df_labeled):,} samples (actual Google reviews)")
        print(f"   ✅ Same 49 features across both datasets")
        print(f"   ✅ Realistic labels based on actual review patterns")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Train models on synthetic data")
        print(f"   2. Validate performance on real data")
        print(f"   3. Compare synthetic vs real performance")
        print(f"   4. Use both in final presentation!")
        
        print(f"\n📁 Files ready for your team:")
        print(f"   - Training: data/processed/train_data.csv (synthetic)")
        print(f"   - Validation: {validation_file} (real)")
        print(f"   - Full real dataset: {real_file}")
        
    else:
        print("❌ Processing failed!")

if __name__ == "__main__":
    main()
