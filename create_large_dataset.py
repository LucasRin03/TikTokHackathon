#!/usr/bin/env python3
"""
Create a larger synthetic dataset for team development
This ensures your team can work with substantial data while real datasets are being acquired
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Configuration
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')

def create_expanded_dataset(target_size=1000):
    """Create a larger, more realistic synthetic dataset"""
    
    print(f"🏭 Creating expanded dataset with {target_size} reviews...")
    
    # Template patterns for different types of reviews
    quality_templates = [
        "Great {food_type} and excellent service! The {dish} was {quality_adj} and staff was {service_adj}.",
        "Visited for {occasion}. Food was {quality_adj}, {specific_comment}.",
        "The {atmosphere_aspect} was {quality_adj}. {food_comment} {service_comment}",
        "{overall_sentiment} experience. {specific_detail} Would {recommendation}.",
        "Ordered the {dish}. It was {quality_adj}. {price_comment} {return_comment}"
    ]
    
    advertisement_templates = [
        "Amazing {food_type}! Visit our website {url} for {discount}% off your next order!",
        "Best {food_type} in town! Check out {url} for special deals!",
        "Call {phone} now for reservations! Special promotion this week only!",
        "Follow us on Instagram @{handle} for daily specials and {discount}% off!",
        "Visit {url} today! Use code {promo_code} for {discount}% discount!"
    ]
    
    irrelevant_templates = [
        "I love my new {device} but this place is too noisy for phone calls.",
        "The weather was {weather} today, but my {item} worked great here.",
        "Just watched {movie} on Netflix. Oh, and I ate here too, food was okay.",
        "My {device} battery died here, but they have good {tech_feature}.",
        "Talking about {random_topic}, this restaurant was average I guess."
    ]
    
    rant_templates = [
        "Never been here but heard from {source} that it's {negative_adj}. Avoid!",
        "Haven't visited yet but based on {evidence}, looks {negative_adj} and overpriced.",
        "Someone told me the {aspect} is {negative_adj} here. Will never go.",
        "According to {source}, this place is {negative_adj}. Don't waste your money.",
        "My {relation} said this place is {negative_adj}. Terrible reputation."
    ]
    
    # Word banks for filling templates
    word_banks = {
        'food_type': ['pizza', 'sushi', 'burgers', 'pasta', 'tacos', 'seafood', 'steaks', 'sandwiches'],
        'dish': ['carbonara', 'margherita pizza', 'salmon roll', 'caesar salad', 'ribeye steak', 'fish tacos'],
        'quality_adj': ['amazing', 'delicious', 'outstanding', 'decent', 'okay', 'mediocre', 'terrible', 'awful'],
        'service_adj': ['friendly', 'attentive', 'professional', 'slow', 'rude', 'helpful', 'courteous'],
        'atmosphere_aspect': ['ambiance', 'music', 'lighting', 'decor', 'seating', 'atmosphere'],
        'occasion': ['lunch', 'dinner', 'date night', 'business meeting', 'family gathering', 'birthday'],
        'url': ['www.restaurant-deals.com', 'foodie-specials.net', 'bestfood-promo.com', 'restaurant-offers.org'],
        'phone': ['(555) 123-4567', '1-800-FOOD-NOW', '(555) 987-6543', '(555) 246-8135'],
        'handle': ['bestfoodever', 'amazingdeals', 'foodie_central', 'restaurant_specials'],
        'promo_code': ['SAVE20', 'HUNGRY50', 'FOODIE25', 'SPECIAL30'],
        'discount': ['20', '25', '30', '50', '15', '40'],
        'device': ['iPhone', 'laptop', 'tablet', 'Android phone', 'MacBook', 'iPad'],
        'weather': ['terrible', 'rainy', 'sunny', 'cold', 'hot', 'windy'],
        'item': ['car', 'bike', 'umbrella', 'jacket', 'headphones', 'watch'],
        'movie': ['Avengers', 'Inception', 'Titanic', 'Avatar', 'Star Wars', 'Batman'],
        'tech_feature': ['WiFi', 'charging stations', 'outlets', 'good cell reception'],
        'random_topic': ['politics', 'sports', 'movies', 'work', 'school', 'travel'],
        'source': ['my friend', 'online reviews', 'Yelp', 'Google reviews', 'my cousin'],
        'evidence': ['the photos', 'reviews online', 'what I heard', 'social media posts'],
        'negative_adj': ['terrible', 'awful', 'disgusting', 'overpriced', 'dirty', 'slow'],
        'aspect': ['service', 'food', 'cleanliness', 'atmosphere', 'management'],
        'relation': ['friend', 'coworker', 'family member', 'neighbor', 'colleague']
    }
    
    def fill_template(template, word_banks):
        """Fill a template with random words from word banks"""
        result = template
        for category, words in word_banks.items():
            placeholder = '{' + category + '}'
            if placeholder in result:
                result = result.replace(placeholder, random.choice(words))
        return result
    
    # Generate reviews
    reviews = []
    
    # Calculate distribution
    quality_count = int(target_size * 0.70)  # 70% quality reviews
    ad_count = int(target_size * 0.10)       # 10% advertisements
    irrelevant_count = int(target_size * 0.10)  # 10% irrelevant
    rant_count = target_size - quality_count - ad_count - irrelevant_count  # Remaining rants
    
    # Generate quality reviews
    for i in range(quality_count):
        review_text = fill_template(random.choice(quality_templates), word_banks)
        reviews.append({
            'review_id': f'quality_{i}',
            'review_text': review_text,
            'rating': random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0],
            'is_advertisement': 0,
            'is_irrelevant': 0,
            'is_rant_no_visit': 0,
            'overall_quality': 1,
            'review_type': 'quality'
        })
    
    # Generate advertisement reviews
    for i in range(ad_count):
        review_text = fill_template(random.choice(advertisement_templates), word_banks)
        reviews.append({
            'review_id': f'ad_{i}',
            'review_text': review_text,
            'rating': random.choices([1, 2, 5], weights=[0.3, 0.2, 0.5])[0],  # Fake high ratings or complaints
            'is_advertisement': 1,
            'is_irrelevant': 0,
            'is_rant_no_visit': 0,
            'overall_quality': 0,
            'review_type': 'advertisement'
        })
    
    # Generate irrelevant reviews
    for i in range(irrelevant_count):
        review_text = fill_template(random.choice(irrelevant_templates), word_banks)
        reviews.append({
            'review_id': f'irrelevant_{i}',
            'review_text': review_text,
            'rating': random.choices([2, 3, 4], weights=[0.4, 0.4, 0.2])[0],
            'is_advertisement': 0,
            'is_irrelevant': 1,
            'is_rant_no_visit': 0,
            'overall_quality': 0,
            'review_type': 'irrelevant'
        })
    
    # Generate rant reviews
    for i in range(rant_count):
        review_text = fill_template(random.choice(rant_templates), word_banks)
        reviews.append({
            'review_id': f'rant_{i}',
            'review_text': review_text,
            'rating': random.choices([1, 2], weights=[0.7, 0.3])[0],
            'is_advertisement': 0,
            'is_irrelevant': 0,
            'is_rant_no_visit': 1,
            'overall_quality': 0,
            'review_type': 'rant_no_visit'
        })
    
    # Create DataFrame
    df = pd.DataFrame(reviews)
    
    # Add metadata
    df['reviewer_name'] = [f'User_{i}' for i in range(len(df))]
    df['review_date'] = pd.date_range('2023-01-01', periods=len(df), freq='2H')
    df['business_name'] = np.random.choice(['Pizza Palace', 'Sushi Central', 'Burger Hub', 'Pasta Corner', 'Taco Town'], len(df))
    df['business_category'] = 'Restaurant'
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    df['review_id'] = range(len(df))
    
    # Save to files
    large_sample_file = RAW_DATA_DIR / f'synthetic_reviews_{target_size}.csv'
    df.to_csv(large_sample_file, index=False)
    
    print(f"✅ Created synthetic dataset: {large_sample_file}")
    print(f"   📊 {len(df)} total reviews")
    print(f"   🎯 Policy violations breakdown:")
    print(f"      - Quality reviews: {quality_count} ({quality_count/target_size*100:.1f}%)")
    print(f"      - Advertisements: {ad_count} ({ad_count/target_size*100:.1f}%)")
    print(f"      - Irrelevant content: {irrelevant_count} ({irrelevant_count/target_size*100:.1f}%)")
    print(f"      - Rants without visit: {rant_count} ({rant_count/target_size*100:.1f}%)")
    
    return df, large_sample_file

def process_large_dataset(df):
    """Process the large dataset with existing pipeline"""
    print("\n🔧 Processing large dataset with existing pipeline...")
    
    try:
        import sys
        sys.path.append('src')
        from data_processing import TextPreprocessor, PolicyFeatureExtractor
        
        # Initialize processors
        text_processor = TextPreprocessor()
        policy_extractor = PolicyFeatureExtractor()
        
        # Apply preprocessing
        print("   📝 Applying text preprocessing...")
        df_processed = text_processor.preprocess_dataframe(df, 'review_text')
        
        print("   🎯 Extracting policy features...")
        df_features = policy_extractor.extract_features_dataframe(df_processed, 'cleaned_text')
        
        # Save processed data
        processed_file = PROCESSED_DATA_DIR / f'processed_synthetic_{len(df)}.csv'
        df_features.to_csv(processed_file, index=False)
        
        print(f"   ✅ Processed data saved: {processed_file}")
        print(f"   📊 Total features: {len(df_features.columns)}")
        
        return df_features, processed_file
        
    except ImportError as e:
        print(f"   ⚠️ Processing failed: {e}")
        return df, None

def create_train_test_splits(df):
    """Create train/validation/test splits for ML team"""
    print("\n🎲 Creating train/validation/test splits...")
    
    from sklearn.model_selection import train_test_split
    
    # Stratify by overall_quality to maintain class balance
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['overall_quality'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['overall_quality'], random_state=42)
    
    # Save splits
    train_file = PROCESSED_DATA_DIR / 'train_data.csv'
    val_file = PROCESSED_DATA_DIR / 'validation_data.csv'
    test_file = PROCESSED_DATA_DIR / 'test_data.csv'
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"   📊 Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   📊 Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   📊 Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   📁 Files saved: {train_file}, {val_file}, {test_file}")
    
    return train_df, val_df, test_df

def main():
    """Main execution function"""
    print("🚀 Creating large synthetic dataset for team development...")
    
    # Create expanded dataset
    df, large_file = create_expanded_dataset(target_size=1000)
    
    # Process with existing pipeline
    processed_df, processed_file = process_large_dataset(df)
    
    # Create train/test splits
    if processed_df is not None:
        train_df, val_df, test_df = create_train_test_splits(processed_df)
    
    print("\n✅ DATASET CREATION COMPLETE!")
    print("🎯 Your team now has:")
    print("   - 1000 labeled synthetic reviews")
    print("   - Processed features ready for ML")
    print("   - Train/validation/test splits")
    print("   - Clear baseline for model development")
    print("\n🚀 Signal your team to start advanced development!")

if __name__ == "__main__":
    main()
