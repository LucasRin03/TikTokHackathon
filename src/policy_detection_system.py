#!/usr/bin/env python3
"""
Sophisticated Policy Violation Detection System
Multi-layered approach: Rules + ML + Ensemble + Confidence Scoring
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
from collections import Counter

class PolicyViolationDetector:
    """
    Advanced policy violation detection using multiple approaches:
    1. Rule-based detection (high precision)
    2. ML-based scoring (pattern learning)
    3. Ensemble combination
    4. Confidence scoring
    """
    
    def __init__(self):
        print("🛡️ Initializing Policy Violation Detection System...")
        
        # Rule-based detectors
        self.rule_detectors = {
            'advertisement': AdvertisementRuleDetector(),
            'irrelevant': IrrelevantContentDetector(), 
            'rant_no_visit': RantNoVisitDetector()
        }
        
        # ML models (will be trained)
        self.ml_models = {
            'advertisement': None,
            'irrelevant': None,
            'rant_no_visit': None
        }
        
        # Ensemble models
        self.ensemble_models = {
            'advertisement': None,
            'irrelevant': None,
            'rant_no_visit': None
        }
        
        print("✅ Policy Detection System ready!")
    
    def detect_violations_rules(self, df):
        """Apply rule-based detection to all reviews"""
        
        print("🔍 Applying rule-based policy detection...")
        
        results = df.copy()
        
        for policy_type, detector in self.rule_detectors.items():
            print(f"   Detecting {policy_type} violations...")
            results[f'rule_{policy_type}'] = results['review_text'].apply(detector.detect)
            results[f'rule_{policy_type}_confidence'] = results['review_text'].apply(detector.get_confidence)
        
        # Overall rule-based quality
        rule_violations = results[['rule_advertisement', 'rule_irrelevant', 'rule_rant_no_visit']].any(axis=1)
        results['rule_overall_quality'] = (~rule_violations).astype(int)
        
        return results
    
    def train_ml_models(self, train_df, feature_columns):
        """Train ML models for each policy type"""
        
        print("🤖 Training ML models for policy detection...")
        
        # Prepare features (exclude text and label columns)
        exclude_cols = ['review_text', 'cleaned_text', 'reviewer_name', 'review_date', 'business_name', 'business_category', 'review_type', 'tokens', 'entities']
        label_cols = ['is_advertisement', 'is_irrelevant', 'is_rant_no_visit', 'overall_quality']
        
        # Only include numeric columns
        numeric_cols = []
        for col in feature_columns:
            if col not in exclude_cols + label_cols:
                try:
                    if train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        numeric_cols.append(col)
                except:
                    continue
        
        available_feature_cols = numeric_cols
        
        if len(available_feature_cols) < 10:
            print(f"⚠️ Warning: Only {len(available_feature_cols)} features available for training")
        
        X = train_df[available_feature_cols].fillna(0)
        
        # Train models for each policy type
        for policy_type in ['advertisement', 'irrelevant', 'rant_no_visit']:
            target_col = f'is_{policy_type}'
            
            if target_col not in train_df.columns:
                print(f"⚠️ Warning: {target_col} not found in training data")
                continue
            
            y = train_df[target_col]
            
            print(f"   Training {policy_type} detector...")
            print(f"      Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")
            
            # Create ensemble of models
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            
            # Train individual models
            rf_model.fit(X, y)
            lr_model.fit(X, y)
            
            # Create voting ensemble
            ensemble = VotingClassifier([
                ('rf', rf_model),
                ('lr', lr_model)
            ], voting='soft')
            
            ensemble.fit(X, y)
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(ensemble, X, y, cv=3, scoring='f1')
            print(f"      Cross-validation F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Store models
            self.ml_models[policy_type] = {
                'model': ensemble,
                'features': available_feature_cols,
                'performance': cv_scores.mean()
            }
        
        print("✅ ML model training complete!")
        return available_feature_cols
    
    def predict_ml_violations(self, df, feature_columns):
        """Predict violations using trained ML models"""
        
        print("🔮 Predicting violations with ML models...")
        
        results = df.copy()
        
        for policy_type, model_info in self.ml_models.items():
            if model_info is None:
                continue
            
            model = model_info['model']
            features = model_info['features']
            
            # Prepare features
            X = df[features].fillna(0)
            
            # Get predictions and probabilities
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]  # Probability of violation
            
            results[f'ml_{policy_type}'] = predictions
            results[f'ml_{policy_type}_confidence'] = probabilities
        
        return results
    
    def create_ensemble_predictions(self, df):
        """Combine rule-based and ML predictions"""
        
        print("🔗 Creating ensemble predictions...")
        
        results = df.copy()
        
        for policy_type in ['advertisement', 'irrelevant', 'rant_no_visit']:
            rule_col = f'rule_{policy_type}'
            ml_col = f'ml_{policy_type}'
            rule_conf_col = f'rule_{policy_type}_confidence'
            ml_conf_col = f'ml_{policy_type}_confidence'
            
            if rule_col in df.columns and ml_col in df.columns:
                # Weighted ensemble based on confidence
                rule_preds = df[rule_col].fillna(0)
                ml_preds = df[ml_col].fillna(0)
                rule_conf = df[rule_conf_col].fillna(0)
                ml_conf = df[ml_conf_col].fillna(0)
                
                # Ensemble decision: prioritize high-confidence predictions
                ensemble_score = (rule_preds * rule_conf + ml_preds * ml_conf) / (rule_conf + ml_conf + 1e-6)
                ensemble_pred = (ensemble_score > 0.5).astype(int)
                
                results[f'ensemble_{policy_type}'] = ensemble_pred
                results[f'ensemble_{policy_type}_confidence'] = ensemble_score
            
            elif rule_col in df.columns:
                # Use only rules if ML not available
                results[f'ensemble_{policy_type}'] = df[rule_col]
                results[f'ensemble_{policy_type}_confidence'] = df[rule_conf_col]
        
        # Overall quality from ensemble
        ensemble_violations = results[['ensemble_advertisement', 'ensemble_irrelevant', 'ensemble_rant_no_visit']].any(axis=1)
        results['ensemble_overall_quality'] = (~ensemble_violations).astype(int)
        
        return results
    
    def get_violation_explanation(self, text, policy_type):
        """Get human-readable explanation for violation detection"""
        
        explanations = []
        
        # Rule-based explanations
        if policy_type in self.rule_detectors:
            rule_explanation = self.rule_detectors[policy_type].explain(text)
            if rule_explanation:
                explanations.append(f"Rule-based: {rule_explanation}")
        
        # ML-based explanations (simplified)
        if policy_type in self.ml_models and self.ml_models[policy_type] is not None:
            explanations.append("ML model indicates violation patterns")
        
        return "; ".join(explanations) if explanations else "No violations detected"


class AdvertisementRuleDetector:
    """Rule-based advertisement detection"""
    
    def __init__(self):
        self.patterns = [
            (r'https?://[^\s]+', 'Contains URL'),
            (r'www\.[^\s]+', 'Contains website reference'),
            (r'\d+%\s*off|\d+%\s*discount', 'Contains discount offer'),
            (r'call\s*now|order\s*now|visit\s*now', 'Contains urgent call-to-action'),
            (r'follow\s+us|like\s+our\s+page', 'Contains social media promotion'),
            (r'@[a-zA-Z0-9_]+', 'Contains social media handle'),
            (r'\(\d{3}\)\s*\d{3}-\d{4}|\d{3}-\d{3}-\d{4}', 'Contains phone number')
        ]
    
    def detect(self, text):
        if not text or pd.isna(text):
            return 0
        
        text = str(text).lower()
        matches = sum(1 for pattern, _ in self.patterns if re.search(pattern, text, re.IGNORECASE))
        return 1 if matches >= 1 else 0
    
    def get_confidence(self, text):
        if not text or pd.isna(text):
            return 0.0
        
        text = str(text).lower()
        matches = sum(1 for pattern, _ in self.patterns if re.search(pattern, text, re.IGNORECASE))
        return min(1.0, matches * 0.3)  # Confidence increases with more matches
    
    def explain(self, text):
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower()
        explanations = [desc for pattern, desc in self.patterns if re.search(pattern, text, re.IGNORECASE)]
        return "; ".join(explanations)


class IrrelevantContentDetector:
    """Rule-based irrelevant content detection"""
    
    def __init__(self):
        self.patterns = [
            (r'my\s+(phone|laptop|computer|tablet|iphone|android)', 'Personal device mention'),
            (r'netflix|youtube|streaming|movie|film', 'Entertainment content mention'),
            (r'weather|rain|snow|sunny|temperature', 'Weather discussion'),
            (r'politics|government|election|president', 'Political content'),
            (r'sports|football|basketball|soccer', 'Sports discussion'),
            (r'work|job|office|meeting|business(?!.*restaurant)', 'Work-related content')
        ]
    
    def detect(self, text):
        if not text or pd.isna(text):
            return 0
        
        text = str(text).lower()
        matches = sum(1 for pattern, _ in self.patterns if re.search(pattern, text, re.IGNORECASE))
        return 1 if matches >= 1 else 0
    
    def get_confidence(self, text):
        if not text or pd.isna(text):
            return 0.0
        
        text = str(text).lower()
        matches = sum(1 for pattern, _ in self.patterns if re.search(pattern, text, re.IGNORECASE))
        return min(1.0, matches * 0.4)
    
    def explain(self, text):
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower()
        explanations = [desc for pattern, desc in self.patterns if re.search(pattern, text, re.IGNORECASE)]
        return "; ".join(explanations)


class RantNoVisitDetector:
    """Rule-based rant without visit detection"""
    
    def __init__(self):
        self.patterns = [
            (r'never\s+(been|visited|went|tried)', 'Admits never visiting'),
            (r'haven\'t\s+(been|visited|gone)', 'Admits not visiting'),
            (r'heard\s+(from|that|it)', 'Based on hearsay'),
            (r'someone\s+(told|said)', 'Second-hand information'),
            (r'according\s+to|based\s+on\s+reviews', 'Based on other reviews'),
            (r'looks\s+like|seems\s+like', 'Speculation based on appearance')
        ]
    
    def detect(self, text):
        if not text or pd.isna(text):
            return 0
        
        text = str(text).lower()
        matches = sum(1 for pattern, _ in self.patterns if re.search(pattern, text, re.IGNORECASE))
        return 1 if matches >= 1 else 0
    
    def get_confidence(self, text):
        if not text or pd.isna(text):
            return 0.0
        
        text = str(text).lower()
        matches = sum(1 for pattern, _ in self.patterns if re.search(pattern, text, re.IGNORECASE))
        return min(1.0, matches * 0.5)
    
    def explain(self, text):
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower()
        explanations = [desc for pattern, desc in self.patterns if re.search(pattern, text, re.IGNORECASE)]
        return "; ".join(explanations)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Policy Violation Detection System...")
    
    # Test with sample data
    test_data = {
        'review_text': [
            "Great food and excellent service!",
            "Visit our website www.promo.com for 50% off!",
            "I love my new iPhone but this place is noisy.",
            "Never been here but heard it's terrible."
        ],
        'is_advertisement': [0, 1, 0, 0],
        'is_irrelevant': [0, 0, 1, 0],
        'is_rant_no_visit': [0, 0, 0, 1]
    }
    
    test_df = pd.DataFrame(test_data)
    
    # Initialize detector
    detector = PolicyViolationDetector()
    
    # Test rule-based detection
    results = detector.detect_violations_rules(test_df)
    
    print("\n📊 Rule-based detection results:")
    for i, text in enumerate(test_data['review_text']):
        print(f"\nReview: '{text[:50]}...'")
        print(f"  Advertisement: {results.iloc[i]['rule_advertisement']} (conf: {results.iloc[i]['rule_advertisement_confidence']:.2f})")
        print(f"  Irrelevant: {results.iloc[i]['rule_irrelevant']} (conf: {results.iloc[i]['rule_irrelevant_confidence']:.2f})")
        print(f"  Rant: {results.iloc[i]['rule_rant_no_visit']} (conf: {results.iloc[i]['rule_rant_no_visit_confidence']:.2f})")
    
    print("\n✅ Policy Detection System test complete!")
