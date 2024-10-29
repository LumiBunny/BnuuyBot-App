import torch
import re
from enum import Enum
from typing import List, Dict, Optional
from transformers import pipeline  # If you're already using this elsewhere

class PreferenceClassifier:
    def __init__(self, models):
        self.models = models
        self.categories = [
            "food", "music", "hobbies", "video_games",
            "projects", "streaming", "vtubing"
        ]
        
        # Category keywords for basic classification
        self.category_keywords = {
            'food': [
                r'\b(eat|food|drink|taste|flavor|dish|meal|snack|cuisine)\b',
                r'\b(cook|bake|recipe|restaurant|hungry|delicious)\b'
            ],
            # Add patterns for other categories...
        }
        
        # Compile patterns once during initialization
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in patterns]
            for category, patterns in self.category_keywords.items()
        }
        
        # Initialize the zero-shot classification pipeline
        self.classifier = self.models.get_classifier()

    def classify_preference(self, text: str) -> dict:
        # Use the zero-shot classifier
        result = self.classifier(
            text,
            candidate_labels=self.categories,
            hypothesis_template="This text is about {}."
        )
        
        # Return the highest scoring category and its confidence
        return {
            'category': result['labels'][0],
            'confidence': result['scores'][0]
        }

class PreferenceExtractor:
    def __init__(self, category: str):
        self.category = category
        
        # Define extraction patterns for each category
        self.extraction_patterns = {
            'food': {
                'items': [
                    r'(?:like|love|enjoy|favorite)\s+([^,.!?]+(?:food|dish|meal))',
                    r'(?:eating|tasting|trying)\s+([^,.!?]+)',
                    r'(?:cook|make|prepare)\s+([^,.!?]+)'
                ],
                'context': [
                    r'(?:at the restaurant|for dinner|for lunch|for breakfast)',
                    r'(?:cuisine|recipe|dish|meal)'
                ]
            },
            # Add patterns for other categories...
        }
        
        # Compile patterns
        self.compiled_patterns = {
            type_: [re.compile(pattern, re.IGNORECASE) 
                   for pattern in patterns]
            for type_, patterns in self.extraction_patterns.get(category, {}).items()
        }

    async def extract_items(self, text: str) -> List[str]:
        extracted_items = set()
        
        # Use patterns to extract items
        if self.category in self.extraction_patterns:
            for pattern in self.compiled_patterns['items']:
                matches = pattern.finditer(text)
                for match in matches:
                    # Clean and normalize extracted item
                    item = self.clean_extracted_item(match.group(1))
                    if item:
                        extracted_items.add(item)
        
        return list(extracted_items)

    def clean_extracted_item(self, text: str) -> Optional[str]:
        """Clean and normalize extracted text"""
        if text:
            # Remove common articles and determiners
            text = re.sub(r'\b(a|an|the)\b', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove common punctuation
            text = text.strip('.,!?')
            return text.strip()
        return None

class PreferenceProcessor:
    def __init__(self, models, chat_history):
        self.chat_history = chat_history
        self.classifier = PreferenceClassifier(models)
        self.extractors = {
            category: PreferenceExtractor(category)
            for category in self.classifier.categories
        }

    async def process_text(self, chat_history: List[dict], user_id: str) -> str:
        best_result = None
        best_score = 0

        for msg in self.chat_history:
            # Process each message individually
            text = f"{msg['role']}: {msg['content']}"
            print(f"Input text: {text}")

            # Classify the text
            classification = self.classifier.classify_preference(text)
            category = classification['category']
            confidence = classification['confidence']
            print(f"Classified category: {category} (confidence: {confidence})")

            if confidence > best_score and confidence > 0.5:
                # Extract items
                extractor = self.extractors.get(category)
                if extractor:
                    items = await extractor.extract_items(text)
                    print(f"Extracted items: {items}")

                    # Format the best result
                    for item in items:
                        formatted_item = f"{user_id} {self.get_sentiment_word(item)} {item}"
                        print(f"Formatted result: {formatted_item}")
                        best_result = formatted_item
                        best_score = confidence

        return best_result or ""

    def get_sentiment_word(self, text: str) -> str:
        """Simple sentiment analysis for preference statements"""
        text = text.lower()
        
        if any(word in text for word in ['love', 'adore', 'favorite']):
            return 'loves'
        elif any(word in text for word in ['hate', 'despise']):
            return 'hates'
        elif any(word in text for word in ['dislike', "don't like"]):
            return 'dislikes'
        elif any(word in text for word in ['like', 'enjoy']):
            return 'likes'
        
        return 'mentions'
    
#processor = PreferenceProcessor()
#results = await processor.process_text("I really love playing Minecraft", "Lumi")
# ['Lumi loves Minecraft']

#results = await processor.process_text("Pizza is my favorite food", "Lumi")
# ['Lumi loves pizza']