#!/usr/bin/env python3
"""
Team monitoring script for Data Engineer
Tracks team progress and identifies blockers
"""

import os
import time
from datetime import datetime
from pathlib import Path

def check_team_progress():
    """Monitor team file creation and progress"""
    
    print(f"🕐 {datetime.now().strftime('%H:%M:%S')} - TEAM PROGRESS CHECK")
    print("=" * 50)
    
    # Expected deliverables from each team member
    expected_files = {
        'NLP Engineer': [
            'src/advanced_features.py',
            'notebooks/feature_engineering.ipynb',
            'data/processed/enhanced_features.csv'
        ],
        'ML Engineer': [
            'src/models.py', 
            'src/evaluation.py',
            'notebooks/model_development.ipynb',
            'models/baseline_model.pkl'
        ],
        'Full-Stack Developer': [
            'app.py',
            'templates/index.html',
            'api/predict.py'
        ]
    }
    
    # Check file existence
    for role, files in expected_files.items():
        print(f"\n👤 {role}:")
        for file_path in files:
            if Path(file_path).exists():
                mod_time = os.path.getmtime(file_path)
                mod_str = datetime.fromtimestamp(mod_time).strftime('%H:%M')
                print(f"   ✅ {file_path} (modified {mod_str})")
            else:
                print(f"   ⏳ {file_path} (not started)")
    
    # Check for new processed data
    processed_dir = Path('data/processed')
    recent_files = []
    if processed_dir.exists():
        for file in processed_dir.glob('*.csv'):
            mod_time = os.path.getmtime(file)
            if time.time() - mod_time < 3600:  # Modified in last hour
                recent_files.append(file.name)
    
    if recent_files:
        print(f"\n📊 Recent data updates: {', '.join(recent_files)}")
    
    print("\n" + "="*50)

def create_support_resources():
    """Create helpful resources for team members"""
    
    # Quick reference guide
    quick_ref = """
# 🔧 QUICK REFERENCE FOR TEAM

## Loading Data:
```python
import pandas as pd
train_df = pd.read_csv('data/processed/train_data.csv')
val_df = pd.read_csv('data/processed/validation_data.csv')
test_df = pd.read_csv('data/processed/test_data.csv')
```

## Features Available:
- Text: review_text, cleaned_text, sentiment_score
- Labels: is_advertisement, is_irrelevant, is_rant_no_visit, overall_quality
- Policy: ad_pattern_*, irrelevant_pattern_*, rant_pattern_*
- Stats: review_length, word_count, token_count

## Preprocessing Pipeline:
```python
from src.data_processing import TextPreprocessor, PolicyFeatureExtractor
processor = TextPreprocessor()
extractor = PolicyFeatureExtractor()
```

## Need Help? Contact Data Engineer!
"""
    
    with open('QUICK_REFERENCE.md', 'w') as f:
        f.write(quick_ref)
    
    print("📋 Created QUICK_REFERENCE.md for team")

if __name__ == "__main__":
    create_support_resources()
    while True:
        check_team_progress()
        time.sleep(1800)  # Check every 30 minutes
