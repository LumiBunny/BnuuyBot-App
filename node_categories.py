from typing import List
import spacy
import torch
from transformers import pipeline
from messages import SentimentAnalyzer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PreferenceClassifier:
    def __init__(self, models):
        self.models = models
        self.classifier = models.get_classifier()
        self.categories = ["food", "music", "hobbies"]  # Will add more later as needed

    def classify_preference(self, text: str):
        result = self.classifier(
            text,
            candidate_labels=self.categories,
            hypothesis_template="This text is about {}."
        )
        return {
            'category': result['labels'][0],
            'confidence': result['scores'][0]
        }

class PreferenceExtractor:
    def __init__(self, category: str):  # Note the category parameter
        self.nlp = spacy.load("en_core_web_sm")
        self.category = category
        self.category_indicators = {
            'food': {
                'indicators': {'eat', 'ate', 'food', 'dish', 'meal', 'snack', 'drink', 'taste'},
                'verbs': {'eat', 'drink', 'cook', 'bake', 'taste'},
                'contexts': {'restaurant', 'kitchen', 'dinner', 'lunch', 'breakfast'}
            },
            'hobbies': {
                'indicators': {'hobby', 'pastime', 'activity', 'interest', 'leisure'},
                'verbs': {'play', 'enjoy', 'practice', 'do', 'make'},
                'contexts': {'weekend', 'free time', 'spare time', 'club', 'class'}
            },
            'music': {
                'indicators': {'music', 'song', 'band', 'artist', 'genre', 'album'},
                'verbs': {'listen', 'play', 'sing', 'dance', 'perform'},
                'contexts': {'concert', 'playlist', 'radio', 'spotify', 'headphones'}
            }
        }

    async def extract_items(self, text: str) -> List[str]:
        # Combine multiple extraction methods
        llm_items = await self.llm_extraction(text)
        spacy_items = self.spacy_extraction(text)
        return self.combine_and_clean(llm_items, spacy_items)

    # I will likely just use OpenAI API for this one here to not slow down "main" LLM.
    # Will set this up in llm_models.py later to implement here.
    async def llm_extraction(self, text: str) -> List[str]:
        prompt = f"""
        Extract all {self.category} items from this text. 
        Return them as a simple comma-separated list.
        Only include the actual {self.category} items, no descriptions or sentiments.
        Text: {text}
        {self.category.capitalize()} items:
        """
        response = await self.models.get_completion(prompt)
        return [item.strip() for item in response.split(',')]

    def spacy_extraction(self, text: str) -> List[str]:
        doc = self.nlp(text)
        items = []
        for chunk in doc.noun_chunks:
            if self.is_category_related(chunk):
                items.append(chunk.text)
        return items

    def is_category_related(self, span) -> bool:
        indicators = self.category_indicators[self.category]
        return any([
            any(token.text.lower() in indicators['indicators'] for token in span),
            any(token.lemma_ in indicators['verbs'] for token in span)
        ])

    def combine_and_clean(self, *item_lists: List[str]) -> List[str]:
        combined = set()
        for item_list in item_lists:
            combined.update(item_list)
        return [item.strip().lower() for item in combined if item.strip()]

class PreferenceProcessor:
    def __init__(self, models):
        self.models = models
        self.classifier = PreferenceClassifier(self.models)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.extractors = {
            'food': PreferenceExtractor('food'),
            'hobbies': PreferenceExtractor('hobbies'),
            'music': PreferenceExtractor('music')
        }

    async def process_text(self, text: str, user_id: str) -> List[str]:
        classification = self.classifier.classify_preference(text)
        category = classification['category']
        
        if classification['confidence'] > 0.7 and category in self.extractors:
            sentiment = self.sentiment_analyzer.get_sentiment(text)
            extractor = self.extractors[category]
            items = await extractor.extract_items(text)
            
            return [
                f"{user_id} {sentiment['word']} {item}"
                for item in items
            ]
        return []

# Usage example:
# async def main():
    # processor = PreferenceProcessor()
    # results = await processor.process_text("I really love pizza and pasta!", "Lumi")
    # print(results)  # ['Lumi loves pizza', 'Lumi loves pasta']