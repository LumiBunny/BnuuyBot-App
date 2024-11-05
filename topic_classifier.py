from typing import List, Optional
from llm_models import LLMModels
from messages import ChatHistory
from category_mappings import CategoryMappings

class ClassifyTopic:
    def __init__(self, models: LLMModels, chat_history: ChatHistory):
        self.chat_history = chat_history
        self.mappings = CategoryMappings()
        self.models = models
        self.categories = None
        self.classification = None
        self.extraction = None

    async def initialize(self):
        self.categories = await self.mappings.get_categories()
        self.classification = await self.mappings.get_classification_mapping()
        self.extraction = await self.mappings.get_extraction_mapping()

    async def classify_sentence(self, sentence: str) -> Optional[str]:
        if self.categories is None:
            await self.initialize()
        try:
            response = await self.models.openai_classifier(sentence, self.categories, self.classification)
            if not response:
                print("openai_classifier returned an empty list")
            return response
        except Exception as e:
            print(f"An error occurred while classifying: {e}")
            return []

    async def classify_list(self, chat_history: List[dict], num_messages: int = 3) -> Optional[str]:
        if self.categories is None:
            await self.initialize()
        recent = chat_history.get_recent_messages(num_messages)
        message_texts = [f"{msg['user_id']}: {msg['content']}" for msg in recent]
        paragraph = " ".join(message_texts)
        try:
            response = await self.models.openai_classifier(paragraph, self.categories, self.classification)
            if not response:
                print("openai_classifier returned an empty list")
            return response
        except Exception as e:
            print(f"An error occurred while classifying: {e}")
            return []
