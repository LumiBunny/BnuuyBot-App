'''
Module for handling user preferences.
'''
from typing import List, Optional
from messages import SentimentAnalyzer, TextFormatting

class PreferenceClassifier:
    """Class for handling zero-shot classification of user preferences."""
    def __init__(self, models):
        self.models = models
        self.categories = [
            "food", "music", "hobbies", "video_games",
            "projects", "streaming", "vtubing", "cooking",
            "coding", "live2d_rigging", "drawing"
        ]

        # Initialize the pipeline
        self.classifier = self.models.get_classifier()

    async def classify_sentence(self, sentence: str) -> Optional[str]:
        """Classify a single sentence and return the category(ies) if any."""
        try:
            result = self.classifier(
                sentence,
                candidate_labels=self.categories,
                hypothesis_template="This text is about {}."
            )
            
            # Get all categories with confidence > 0.5
            categories = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]
            
            return categories
        except:
            return []
        
class ItemExtractor:
    """Class for handling calling OpenAI functions for each category."""
    def __init__(self, models):
        self.models = models

    async def extract_food_items(self, text: str) -> List[str]:
        items = await self.models.use_openai_functions(text, "extract_food_items")
        return [item["item"] for item in items if item["item"] != "None"]

    async def extract_hobbies(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_hobbies")
    
    async def extract_music(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_music")
    
    async def extract_video_games(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_video_games")

    async def extract_streaming(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_streaming")

    async def extract_vtubing(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_vtubing")
    
    async def extract_live2d_rigging(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_live2d_rigging")

    async def extract_drawing(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_drawing")

    async def extract_coding(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_coding")

    async def extract_cooking(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_cooking")

    async def extract_projects(self, text: str) -> List[dict]:
        return await self.models.use_openai_functions(text, "extract_projects")
           
class PreferenceProcessor:
        def __init__(self, models, chat_history):
            self.models = models
            self.text = TextFormatting(chat_history, self.models)
            self.classifier = PreferenceClassifier(models)
            self.sentiment_analyzer = SentimentAnalyzer()
            self.item_extractor = ItemExtractor(models)

        async def process_text(self, chat_history: List[dict], user_id: str) -> List[str]:
            """Returns a list of suggested user preferences from the chat history."""
            if not chat_history:
                return [f"{user_id} has mentioned something"]
        
            most_recent_msg = chat_history[-1]
            sentence = f"{most_recent_msg['role']}: {most_recent_msg['content']}"
        
            categories = await self.classifier.classify_sentence(sentence)
            sentiment_result = await self.sentiment_analyzer.get_sentiment(sentence)
            sentiment = sentiment_result['word']
        
            results = []
        
            for category in categories:
                result = await self.process_category(category, sentence, user_id, sentiment)
                if result:
                    results.append(result)
        
            if not results:
                if sentiment == "favourite":
                    results.append(f"{user_id}'s favourite thing is something.")
                else:
                    results.append(f"{user_id} {sentiment} something.")
        
            return results
        
        async def process_category(self, category, sentence, user_id, sentiment):
            category_mapping = {
                "music": ("extract_music", "music item"),
                "food": ("extract_food_items", "food"),
                "hobby": ("extract_hobbies", "hobby"),
                "video_games": ("extract_video_games", "video game"),
                "streaming": ("extract_streaming", "streaming item"),
                "vtubing": ("extract_vtubing", "vtubing item"),
                "live2d_rigging": ("extract_live2d_rigging", "Live2D rigging item"),
                "drawing": ("extract_drawing", "drawing item"),
                "coding": ("extract_coding", "coding item"),
                "cooking": ("extract_cooking", "cooking item"),
                "projects": ("extract_projects", "project")
            }
        
            if category in category_mapping:
                extractor_method, category_name = category_mapping[category]
                items = await getattr(self.item_extractor, extractor_method)(sentence)
                return self.format_category_result(items, category_name, user_id, sentiment)
            return None
        
        def format_category_result(self, items, category_name, user_id, sentiment):
            if items:
                values = [item['value'] for item in items]
                actions = [item['action'] for item in items if item['action']]
                if sentiment == "favourite":
                    return f"{user_id}'s favourite {category_name}s are {self.format_list(values)}."
                if actions and all(action == actions[0] for action in actions):
                    return f"{user_id} {sentiment} {actions[0]} {self.format_list(values)}."
                return f"{user_id} {sentiment} {self.format_list(values)}."
            return None
        
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