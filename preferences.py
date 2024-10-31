import re
from typing import List, Optional
from messages import SentimentAnalyzer

class PreferenceClassifier:
    def __init__(self, models):
        self.models = models
        self.categories = [
            "food", "music", "hobbies", "video_games",
            "projects", "streaming", "vtubing"
        ]

        # Initialize the zero-shot classification pipeline
        self.classifier = self.models.get_classifier()

    async def classify_sentence(self, sentence: str) -> Optional[str]:
        """Classify a single sentence and return the category (if any)."""
        try:
            result = self.classifier(
                sentence,
                candidate_labels=self.categories,
                hypothesis_template="This text is about {}."
            )
            
            # Get the category with the highest confidence score
            category = result["labels"][0]
            confidence = result["scores"][0]
            
            if confidence > 0.5:
                return category
            else:
                return None
        except:
            return None

class ItemExtractor:
    def __init__(self, models):
        self.models = models

    async def extract_food_items(self, text: str) -> List[str]:
        return await self.models.use_openai_functions(text) # Calls OpenAI function for extracting food items
        
class PreferenceProcessor:
        def __init__(self, models):
            self.models = models
            self.classifier = PreferenceClassifier(models)
            self.sentiment_analyzer = SentimentAnalyzer()
            self.item_extractor = ItemExtractor(models)

        async def process_text(self, chat_history: List[dict], user_id: str) -> str:
            recent_messages = chat_history.get_recent_messages(1)
            last_sentiment = None
            last_category = None

            for msg in recent_messages:
                sentence = f"{msg['role']}: {msg['content']}"
                
                # Step 3: Classify the sentence
                category = await self.classifier.classify_sentence(sentence)
                if not category:
                    continue  # Step 7: Skip to next sentence if no category found
                
                last_category = category  # Remember the last found category
                
                # Step 4: Get sentiment
                sentiment_result = await self.sentiment_analyzer.get_sentiment(sentence)
                sentiment = sentiment_result['word']
                if sentiment == "has mentioned":
                    continue  # Step 8: Skip to next sentence if no significant sentiment found
                
                last_sentiment = sentiment  # Remember the last found sentiment
                
                # Step 5: Extract items
                items = await self.item_extractor.extract_food_items(sentence)
                if not items:
                    continue  # Step 9: Skip to next sentence if no items found
                
                # Step 6: If we have all three (category, sentiment, and item), format and return result
                if sentiment == "favourite":
                    return f"{user_id}'s favourite {category} is {items[0]}."
                if category and sentiment != "has mentioned" and items:
                    return f"{user_id} {sentiment} {items[0]}"

            # Step 10: If we've processed all sentences without returning
            if not last_sentiment:
                last_sentiment = "has mentioned"
            if not last_category:
                last_category = "something"
            
            return f"{user_id} {last_sentiment} {last_category}"
        
        async def get_sentiment_word(self, text: str) -> Optional[str]:
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
            else:
                return None
    
#processor = PreferenceProcessor()
#results = await processor.process_text("I really love playing Minecraft", "Lumi")
# ['Lumi loves Minecraft']

#results = await processor.process_text("Pizza is my favorite food", "Lumi")
# ['Lumi loves pizza']