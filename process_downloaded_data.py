#!/usr/bin/env python3
"""
Process downloaded real Google Reviews data
Run this AFTER you download real data to data/raw/
"""

import pandas as pd
import json
import gzip
from pathlib import Path

# Configuration  
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')

def detect_downloaded_files():
    """Detect what real data files have been downloaded"""
    
    print("🔍 DETECTING DOWNLOADED REAL DATA FILES")
    print("=" * 50)
    
    # Common file patterns from real datasets
    patterns = {
        'csv': ['*.csv', '*reviews*.csv', '*restaurant*.csv', '*google*.csv'],
        'json': ['*.json', '*.json.gz'],
        'zip': ['*.zip']
    }
    
    found_files = []
    
    for file_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            files = list(RAW_DATA_DIR.glob(pattern))
            for file in files:
                if 'synthetic' not in file.name and 'sample' not in file.name:  # Skip our created files
                    found_files.append((file, file_type))
    
    if found_files:
        print("✅ Found real data files:")
        for file_path, file_type in found_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   📄 {file_path.name} ({file_type.upper()}, {size_mb:.1f} MB)")
        return found_files
    else:
        print("❌ No real data files found in data/raw/")
        print("📥 Please download data first:")
        print("   Kaggle: https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews")
        print("   UCSD: https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/")
        return []

def process_csv_file(file_path):
    """Process CSV format real data"""
    
    print(f"\n📊 PROCESSING CSV FILE: {file_path.name}")
    print("-" * 40)
    
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        print(f"   📈 Loaded {len(df):,} reviews")
        
        # Inspect columns
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Try to standardize column names
        column_mapping = {
            # Common variations in real datasets
            'review': 'review_text',
            'text': 'review_text', 
            'review_text': 'review_text',
            'content': 'review_text',
            'comment': 'review_text',
            'rating': 'rating',
            'stars': 'rating',
            'score': 'rating',
            'user': 'reviewer_name',
            'username': 'reviewer_name',
            'reviewer': 'reviewer_name',
            'date': 'review_date',
            'timestamp': 'review_date',
            'time': 'review_date'
        }
        
        # Rename columns
        df_renamed = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df_renamed = df_renamed.rename(columns={old_name: new_name})
                print(f"   🔄 Renamed '{old_name}' → '{new_name}'")
        
        # Check for required columns
        if 'review_text' not in df_renamed.columns:
            print("   ❌ No review text column found!")
            return None
        
        # Sample the data to manageable size for hackathon
        if len(df_renamed) > 5000:
            df_sample = df_renamed.sample(n=5000, random_state=42)
            print(f"   📊 Sampled {len(df_sample):,} reviews for hackathon use")
        else:
            df_sample = df_renamed
        
        # Clean up review text
        df_sample = df_sample[df_sample['review_text'].notna()]
        df_sample = df_sample[df_sample['review_text'].str.len() > 10]  # Remove very short reviews
        
        print(f"   🧹 After cleaning: {len(df_sample):,} reviews")
        
        return df_sample
        
    except Exception as e:
        print(f"   ❌ Error processing CSV: {e}")
        return None

def process_json_file(file_path):
    """Process JSON format real data (UCSD format)"""
    
    print(f"\n📊 PROCESSING JSON FILE: {file_path.name}")
    print("-" * 40)
    
    try:
        reviews = []
        
        # Handle gzipped files
        if file_path.name.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                for line_num, line in enumerate(f):
                    if line_num >= 5000:  # Limit for hackathon
                        break
                    try:
                        review = json.loads(line.strip())
                        reviews.append(review)
                    except:
                        continue
        else:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num >= 5000:  # Limit for hackathon
                        break
                    try:
                        review = json.loads(line.strip())
                        reviews.append(review)
                    except:
                        continue
        
        print(f"   📈 Loaded {len(reviews):,} reviews from JSON")
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Standardize UCSD format columns
        ucsd_mapping = {
            'reviewText': 'review_text',
            'overall': 'rating',
            'reviewerID': 'reviewer_name',
            'asin': 'business_id',
            'reviewTime': 'review_date',
            'summary': 'review_summary'
        }
        
        for old_name, new_name in ucsd_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"   🔄 Renamed '{old_name}' → '{new_name}'")
        
        # Clean up
        if 'review_text' in df.columns:
            df = df[df['review_text'].notna()]
            df = df[df['review_text'].str.len() > 10]
            print(f"   🧹 After cleaning: {len(df):,} reviews")
            return df
        else:
            print("   ❌ No review text found in JSON!")
            return None
        
    except Exception as e:
        print(f"   ❌ Error processing JSON: {e}")
        return None

def apply_pipeline_to_real_data(df):
    """Apply existing preprocessing pipeline to real data"""
    
    print(f"\n🔧 APPLYING PREPROCESSING PIPELINE TO REAL DATA")
    print("=" * 50)
    
    try:
        import sys
        sys.path.append('src')
        from data_processing import TextPreprocessor, PolicyFeatureExtractor
        
        # Initialize processors
        text_processor = TextPreprocessor()
        policy_extractor = PolicyFeatureExtractor()
        
        print("   📝 Applying text preprocessing...")
        df_processed = text_processor.preprocess_dataframe(df, 'review_text')
        
        print("   🎯 Extracting policy features...")
        df_features = policy_extractor.extract_features_dataframe(df_processed, 'cleaned_text')
        
        # Since real data won't have ground truth labels, we'll create pseudo-labels
        print("   🏷️ Creating pseudo-labels for policy violations...")
        
        # Use our feature extractors to create labels
        df_features['is_advertisement'] = (
            (df_features['has_url'] > 0) | 
            (df_features['has_discount_mention'] > 0) |
            (df_features['has_contact_mention'] > 0)
        ).astype(int)
        
        df_features['is_irrelevant'] = (
            df_features['has_off_topic_mention'] > 0
        ).astype(int)
        
        df_features['is_rant_no_visit'] = (
            (df_features['has_never_mention'] > 0) | 
            (df_features['has_heard_mention'] > 0)
        ).astype(int)
        
        # Overall quality (no violations)
        df_features['overall_quality'] = (
            (df_features['is_advertisement'] == 0) & 
            (df_features['is_irrelevant'] == 0) & 
            (df_features['is_rant_no_visit'] == 0)
        ).astype(int)
        
        # Save processed real data
        real_processed_file = PROCESSED_DATA_DIR / 'real_data_processed.csv'
        df_features.to_csv(real_processed_file, index=False)
        
        print(f"   ✅ Processed real data saved: {real_processed_file}")
        print(f"   📊 Total features: {len(df_features.columns)}")
        print(f"   🎯 Pseudo-labels created:")
        print(f"      - Quality reviews: {df_features['overall_quality'].sum():,} ({df_features['overall_quality'].mean()*100:.1f}%)")
        print(f"      - Advertisements: {df_features['is_advertisement'].sum():,} ({df_features['is_advertisement'].mean()*100:.1f}%)")
        print(f"      - Irrelevant: {df_features['is_irrelevant'].sum():,} ({df_features['is_irrelevant'].mean()*100:.1f}%)")
        print(f"      - Rants: {df_features['is_rant_no_visit'].sum():,} ({df_features['is_rant_no_visit'].mean()*100:.1f}%)")
        
        return df_features, real_processed_file
        
    except Exception as e:
        print(f"   ❌ Pipeline processing failed: {e}")
        return None, None

def main():
    """Main processing function - run after downloading real data"""
    
    print("🚀 PROCESSING DOWNLOADED REAL GOOGLE REVIEWS DATA")
    print("=" * 60)
    
    # Detect downloaded files
    found_files = detect_downloaded_files()
    
    if not found_files:
        print("\n📥 DOWNLOAD INSTRUCTIONS:")
        print("1. Go to: https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews")
        print("2. Download the dataset")
        print("3. Save to data/raw/ folder")
        print("4. Run this script again")
        return
    
    # Process each found file
    processed_dfs = []
    
    for file_path, file_type in found_files:
        if file_type == 'csv':
            df = process_csv_file(file_path)
        elif file_type == 'json':
            df = process_json_file(file_path)
        else:
            print(f"⚠️ Skipping {file_type} file: {file_path.name}")
            continue
        
        if df is not None:
            processed_dfs.append(df)
    
    if processed_dfs:
        # Combine all processed dataframes
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        print(f"\n📊 COMBINED REAL DATA: {len(combined_df):,} reviews")
        
        # Apply preprocessing pipeline
        final_df, processed_file = apply_pipeline_to_real_data(combined_df)
        
        if final_df is not None:
            print(f"\n✅ SUCCESS! REAL DATA READY FOR VALIDATION")
            print(f"🎯 Use this data to:")
            print(f"   - Validate your trained models")
            print(f"   - Compare performance vs synthetic data")
            print(f"   - Demonstrate real-world applicability")
            print(f"   - Impress judges with actual Google reviews")
            
            print(f"\n📁 Files ready:")
            print(f"   - Training (synthetic): data/processed/train_data.csv")
            print(f"   - Validation (real): {processed_file}")
            print(f"   - Test (synthetic): data/processed/test_data.csv")

if __name__ == "__main__":
    main()
