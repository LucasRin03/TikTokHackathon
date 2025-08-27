import os
import keras_nlp
import keras

class GemmaClassifier:
    def __init__(self, model_path="gemma_1.1_instruct_2b_en"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the Gemma model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model directory {self.model_path} not found")
        
        print("Loading Gemma model...")
        self.model = keras_nlp.models.GemmaCausalLM.from_preset(self.model_path)
        print("Model loaded successfully!")
    
    def classify_review(self, review_text, policy_type):
        """Classify a review based on policy type"""
        prompts = {
            "advertisement": f"Classify this review as either 'advertisement' or 'legitimate review': {review_text}",
            "irrelevant": f"Classify this review as either 'irrelevant content' or 'relevant content': {review_text}",
            "rant": f"Classify this review as either 'rant without visit' or 'legitimate complaint': {review_text}"
        }
        
        if policy_type not in prompts:
            raise ValueError("Policy type must be 'advertisement', 'irrelevant', or 'rant'")
        
        # Generate classification
        response = self.model.generate(prompts[policy_type], max_length=100)
        return response
    
    def batch_classify(self, reviews, policy_type):
        """Classify multiple reviews"""
        results = []
        for review in reviews:
            result = self.classify_review(review, policy_type)
            results.append(result)
        return results

# Example usage
if __name__ == "__main__":
    classifier = GemmaClassifier()
    
    # Test with sample reviews
    sample_reviews = [
        "Best pizza! Visit www.pizzapromo.com for discounts!",
        "The food was amazing and service was excellent.",
        "I love my new phone, but this place is too noisy."
    ]
    
    print("Testing advertisement detection:")
    for review in sample_reviews:
        result = classifier.classify_review(review, "advertisement")
        print(f"Review: {review}")
        print(f"Classification: {result}")
        print("---")