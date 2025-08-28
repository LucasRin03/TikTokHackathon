#!/usr/bin/env python3
"""
Quick data verification script for team members
Run this to ensure your data is ready for development
"""

import pandas as pd
import numpy as np
from pathlib import Path

def verify_datasets():
    """Verify all datasets are ready and properly formatted"""
    
    print("🔍 VERIFYING HACKATHON DATASETS")
    print("=" * 50)
    
    # Check file existence
    required_files = [
        'data/processed/train_data.csv',
        'data/processed/validation_data.csv',
        'data/processed/test_data.csv',
        'data/processed/processed_synthetic_1000.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ MISSING FILES:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    
    # Load and verify datasets
    try:
        train_df = pd.read_csv('data/processed/train_data.csv')
        val_df = pd.read_csv('data/processed/validation_data.csv') 
        test_df = pd.read_csv('data/processed/test_data.csv')
        
        print(f"\n📊 DATASET SIZES:")
        print(f"   Training: {len(train_df):,} samples")
        print(f"   Validation: {len(val_df):,} samples")
        print(f"   Test: {len(test_df):,} samples")
        print(f"   Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
        
        # Verify required columns
        required_columns = [
            'review_text', 'cleaned_text', 'rating',
            'is_advertisement', 'is_irrelevant', 'is_rant_no_visit', 'overall_quality',
            'sentiment_score', 'review_length', 'word_count'
        ]
        
        missing_cols = [col for col in required_columns if col not in train_df.columns]
        if missing_cols:
            print(f"\n❌ MISSING COLUMNS: {missing_cols}")
            return False
        
        print(f"\n✅ All required columns present ({len(train_df.columns)} total features)")
        
        # Verify class distribution
        print(f"\n🎯 CLASS DISTRIBUTION (Training):")
        print(f"   Quality Reviews: {train_df['overall_quality'].sum():,} ({train_df['overall_quality'].mean()*100:.1f}%)")
        print(f"   Advertisements: {train_df['is_advertisement'].sum():,} ({train_df['is_advertisement'].mean()*100:.1f}%)")
        print(f"   Irrelevant: {train_df['is_irrelevant'].sum():,} ({train_df['is_irrelevant'].mean()*100:.1f}%)")
        print(f"   Rants: {train_df['is_rant_no_visit'].sum():,} ({train_df['is_rant_no_visit'].mean()*100:.1f}%)")
        
        # Verify data quality
        print(f"\n🔧 DATA QUALITY:")
        print(f"   Missing values: {train_df.isnull().sum().sum()}")
        print(f"   Sentiment range: {train_df['sentiment_score'].min():.3f} to {train_df['sentiment_score'].max():.3f}")
        print(f"   Review length range: {train_df['review_length'].min()} to {train_df['review_length'].max()} chars")
        
        # Test a sample
        print(f"\n🧪 SAMPLE DATA TEST:")
        sample = train_df.iloc[0]
        print(f"   Sample review: '{sample['review_text'][:100]}...'")
        print(f"   Rating: {sample['rating']}")
        print(f"   Policy violations: Ads={sample['is_advertisement']}, Irrelevant={sample['is_irrelevant']}, Rant={sample['is_rant_no_visit']}")
        print(f"   Sentiment: {sample['sentiment_score']:.3f}")
        
        print(f"\n✅ ALL VERIFICATIONS PASSED!")
        print(f"🚀 Your data is ready for model development!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR loading datasets: {e}")
        return False

def test_preprocessing_pipeline():
    """Test that the preprocessing pipeline works"""
    
    print(f"\n🔧 TESTING PREPROCESSING PIPELINE")
    print("=" * 40)
    
    try:
        import sys
        sys.path.append('src')
        from data_processing import TextPreprocessor, PolicyFeatureExtractor
        
        # Test sample text
        test_text = "Amazing food! Visit our website www.example.com for 20% off!"
        
        # Initialize processors
        text_processor = TextPreprocessor()
        policy_extractor = PolicyFeatureExtractor()
        
        # Test text processing
        cleaned = text_processor.clean_text(test_text)
        sentiment = text_processor.get_sentiment_score(cleaned)
        
        print(f"   Original: {test_text}")
        print(f"   Cleaned: {cleaned}")
        print(f"   Sentiment: {sentiment:.3f}")
        
        # Test policy features
        features = policy_extractor.extract_all_features(test_text)
        violation_count = sum(features.values())
        
        print(f"   Policy violations detected: {violation_count}")
        print(f"   Has URL: {features.get('has_url', 0)}")
        print(f"   Has discount: {features.get('has_discount_mention', 0)}")
        
        print(f"\n✅ PREPROCESSING PIPELINE WORKING!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Cannot import preprocessing modules: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Preprocessing test failed: {e}")
        return False

def main():
    """Main verification function"""
    
    print("🎯 TIKTOK HACKATHON - DATA VERIFICATION")
    print("=" * 60)
    
    # Run verifications
    data_ok = verify_datasets()
    pipeline_ok = test_preprocessing_pipeline()
    
    print(f"\n" + "=" * 60)
    if data_ok and pipeline_ok:
        print("🎉 ALL SYSTEMS GO! YOU'RE READY FOR DEVELOPMENT!")
        print("\n📋 NEXT STEPS:")
        print("   1. Read HANDOFF_PACKAGE.md for your role-specific instructions")
        print("   2. Start building your components")
        print("   3. Coordinate with team members")
        print("   4. Build something amazing! 🚀")
    else:
        print("❌ SETUP ISSUES DETECTED")
        print("   Please resolve the above issues before continuing")
        print("   Contact the Data Engineer if you need help")

if __name__ == "__main__":
    main()
