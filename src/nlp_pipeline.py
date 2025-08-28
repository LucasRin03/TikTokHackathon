#!/usr/bin/env python3
"""
Complete NLP Pipeline for TikTok Hackathon
Integrates advanced features + policy detection + ML training
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import our custom modules
from advanced_nlp_features import AdvancedNLPFeatureExtractor
from policy_detection_system import PolicyViolationDetector
from data_processing import TextPreprocessor, PolicyFeatureExtractor

class ComprehensiveNLPPipeline:
    """
    Complete NLP pipeline that combines:
    1. Basic preprocessing
    2. Advanced NLP features  
    3. Policy violation detection
    4. ML model training
    5. Ensemble predictions
    """
    
    def __init__(self):
        print("🚀 Initializing Comprehensive NLP Pipeline...")
        
        # Initialize all components
        self.basic_processor = TextPreprocessor()
        self.basic_policy_extractor = PolicyFeatureExtractor()
        self.advanced_extractor = AdvancedNLPFeatureExtractor()
        self.policy_detector = PolicyViolationDetector()
        
        self.is_trained = False
        self.feature_columns = []
        
        print("✅ NLP Pipeline ready!")
    
    def process_training_data(self, df, text_column='review_text'):
        """Complete processing pipeline for training data"""
        
        print(f"🏭 PROCESSING TRAINING DATA: {len(df)} reviews")
        print("=" * 60)
        
        # Step 1: Basic preprocessing
        print("📝 Step 1: Basic text preprocessing...")
        df_basic = self.basic_processor.preprocess_dataframe(df, text_column)
        
        # Step 2: Basic policy features
        print("🎯 Step 2: Basic policy feature extraction...")
        df_policy = self.basic_policy_extractor.extract_features_dataframe(df_basic, 'cleaned_text')
        
        # Step 3: Advanced NLP features (sample for speed during development)
        print("🧠 Step 3: Advanced NLP feature extraction...")
        if len(df) > 100:
            print(f"   Processing sample of 100 reviews for development speed...")
            sample_indices = np.random.choice(df.index, size=100, replace=False)
            df_sample = df_policy.loc[sample_indices].copy()
            df_advanced = self.advanced_extractor.process_dataframe(df_sample, text_column)
            
            # For non-sample rows, fill with zeros (in production, you'd process all)
            advanced_cols = [col for col in df_advanced.columns if col not in df_policy.columns]
            for col in advanced_cols:
                df_policy[col] = 0.0
            df_policy.loc[sample_indices, advanced_cols] = df_advanced[advanced_cols]
            df_features = df_policy
        else:
            # Process all if small dataset
            df_features = self.advanced_extractor.process_dataframe(df_policy, text_column)
        
        # Step 4: Policy violation detection
        print("🛡️ Step 4: Policy violation detection...")
        df_with_rules = self.policy_detector.detect_violations_rules(df_features)
        
        # Step 5: Train ML models
        print("🤖 Step 5: Training ML models...")
        self.feature_columns = self.policy_detector.train_ml_models(df_with_rules, df_with_rules.columns)
        
        # Step 6: Get ML predictions
        print("🔮 Step 6: ML predictions...")
        df_with_ml = self.policy_detector.predict_ml_violations(df_with_rules, self.feature_columns)
        
        # Step 7: Create ensemble predictions
        print("🔗 Step 7: Ensemble predictions...")
        df_final = self.policy_detector.create_ensemble_predictions(df_with_ml)
        
        self.is_trained = True
        
        print(f"✅ Training pipeline complete!")
        print(f"   📊 Final features: {len(df_final.columns)}")
        print(f"   🎯 Available predictors: rules, ML, ensemble")
        
        return df_final
    
    def process_validation_data(self, df, text_column='review_text'):
        """Process validation/test data using trained pipeline"""
        
        if not self.is_trained:
            raise ValueError("Pipeline must be trained first!")
        
        print(f"🧪 PROCESSING VALIDATION DATA: {len(df)} reviews")
        print("=" * 50)
        
        # Step 1: Basic preprocessing
        print("📝 Basic preprocessing...")
        df_basic = self.basic_processor.preprocess_dataframe(df, text_column)
        
        # Step 2: Basic policy features  
        print("🎯 Basic policy features...")
        df_policy = self.basic_policy_extractor.extract_features_dataframe(df_basic, 'cleaned_text')
        
        # Step 3: Advanced features (sample for speed)
        print("🧠 Advanced features...")
        if len(df) > 50:
            # Process sample for demo
            sample_size = min(50, len(df))
            sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
            df_sample = df_policy.loc[sample_indices].copy()
            df_advanced = self.advanced_extractor.process_dataframe(df_sample, text_column)
            
            # Fill non-sample with zeros
            advanced_cols = [col for col in df_advanced.columns if col not in df_policy.columns]
            for col in advanced_cols:
                df_policy[col] = 0.0
            df_policy.loc[sample_indices, advanced_cols] = df_advanced[advanced_cols]
            df_features = df_policy
        else:
            df_features = self.advanced_extractor.process_dataframe(df_policy, text_column)
        
        # Step 4: Rule-based detection
        print("🛡️ Rule-based detection...")
        df_with_rules = self.policy_detector.detect_violations_rules(df_features)
        
        # Step 5: ML predictions
        print("🔮 ML predictions...")
        df_with_ml = self.policy_detector.predict_ml_violations(df_with_rules, self.feature_columns)
        
        # Step 6: Ensemble predictions
        print("🔗 Ensemble predictions...")
        df_final = self.policy_detector.create_ensemble_predictions(df_with_ml)
        
        print("✅ Validation pipeline complete!")
        
        return df_final
    
    def evaluate_performance(self, df_results, true_labels_prefix='is_'):
        """Evaluate pipeline performance on labeled data"""
        
        print("📊 EVALUATING PIPELINE PERFORMANCE")
        print("=" * 40)
        
        policy_types = ['advertisement', 'irrelevant', 'rant_no_visit']
        
        for policy_type in policy_types:
            true_col = f'{true_labels_prefix}{policy_type}'
            
            if true_col not in df_results.columns:
                print(f"⚠️ True labels for {policy_type} not found")
                continue
            
            print(f"\n🎯 {policy_type.upper()} DETECTION:")
            
            y_true = df_results[true_col]
            
            # Evaluate each approach
            for approach in ['rule', 'ml', 'ensemble']:
                pred_col = f'{approach}_{policy_type}'
                
                if pred_col in df_results.columns:
                    y_pred = df_results[pred_col]
                    
                    # Calculate metrics
                    tp = ((y_true == 1) & (y_pred == 1)).sum()
                    fp = ((y_true == 0) & (y_pred == 1)).sum()
                    fn = ((y_true == 1) & (y_pred == 0)).sum()
                    tn = ((y_true == 0) & (y_pred == 0)).sum()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"   {approach.upper()}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Overall quality evaluation
        if f'{true_labels_prefix}overall_quality' in df_results.columns or 'overall_quality' in df_results.columns:
            true_quality_col = f'{true_labels_prefix}overall_quality' if f'{true_labels_prefix}overall_quality' in df_results.columns else 'overall_quality'
            
            print(f"\n🎯 OVERALL QUALITY:")
            y_true_quality = df_results[true_quality_col]
            
            for approach in ['rule', 'ensemble']:
                pred_col = f'{approach}_overall_quality'
                
                if pred_col in df_results.columns:
                    y_pred_quality = df_results[pred_col]
                    
                    tp = ((y_true_quality == 1) & (y_pred_quality == 1)).sum()
                    fp = ((y_true_quality == 0) & (y_pred_quality == 1)).sum()
                    fn = ((y_true_quality == 1) & (y_pred_quality == 0)).sum()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"   {approach.upper()}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    def predict_single_review(self, text):
        """Predict policy violations for a single review"""
        
        if not self.is_trained:
            raise ValueError("Pipeline must be trained first!")
        
        # Create temporary dataframe
        temp_df = pd.DataFrame({'review_text': [text]})
        
        # Process through pipeline (simplified for single review)
        result = self.process_validation_data(temp_df)
        
        # Extract predictions
        predictions = {}
        for policy_type in ['advertisement', 'irrelevant', 'rant_no_visit']:
            predictions[policy_type] = {
                'rule_prediction': result[f'rule_{policy_type}'].iloc[0],
                'rule_confidence': result[f'rule_{policy_type}_confidence'].iloc[0],
                'ensemble_prediction': result[f'ensemble_{policy_type}'].iloc[0] if f'ensemble_{policy_type}' in result.columns else None,
                'ensemble_confidence': result[f'ensemble_{policy_type}_confidence'].iloc[0] if f'ensemble_{policy_type}_confidence' in result.columns else None
            }
        
        predictions['overall_quality'] = result['ensemble_overall_quality'].iloc[0] if 'ensemble_overall_quality' in result.columns else result['rule_overall_quality'].iloc[0]
        
        return predictions
    
    def save_pipeline(self, filepath):
        """Save trained pipeline"""
        import joblib
        
        pipeline_data = {
            'policy_detector': self.policy_detector,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"💾 Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath):
        """Load trained pipeline"""
        import joblib
        
        pipeline_data = joblib.load(filepath)
        self.policy_detector = pipeline_data['policy_detector']
        self.feature_columns = pipeline_data['feature_columns']
        self.is_trained = pipeline_data['is_trained']
        
        print(f"📂 Pipeline loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Comprehensive NLP Pipeline...")
    
    # Test with sample data
    test_data = {
        'review_text': [
            "Great food and excellent service! Will definitely return.",
            "Visit our website www.promo.com for 50% off your next order!",
            "I love my new iPhone but this place is too noisy for calls.",
            "Never been here but heard from friends it's terrible.",
            "Amazing pasta and friendly staff. Highly recommend!",
            "Check out our Instagram @foodiecentral for daily specials!"
        ],
        'is_advertisement': [0, 1, 0, 0, 0, 1],
        'is_irrelevant': [0, 0, 1, 0, 0, 0],
        'is_rant_no_visit': [0, 0, 0, 1, 0, 0],
        'overall_quality': [1, 0, 0, 0, 1, 0]
    }
    
    test_df = pd.DataFrame(test_data)
    
    # Initialize and train pipeline
    pipeline = ComprehensiveNLPPipeline()
    
    print("\n🏭 Training pipeline...")
    results = pipeline.process_training_data(test_df)
    
    print("\n📊 Evaluating performance...")
    pipeline.evaluate_performance(results)
    
    print("\n🧪 Testing single review prediction...")
    test_review = "Visit our website for amazing deals! 50% off everything!"
    prediction = pipeline.predict_single_review(test_review)
    
    print(f"Review: '{test_review}'")
    print(f"Predictions: {prediction}")
    
    print("\n✅ NLP Pipeline test complete!")
