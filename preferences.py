'''
Module for handling user preferences.
'''
from typing import List
from topic_classifier import ClassifyTopic
from tests.sentiments import SentimentAnalyzer
from messages import TextFormatting
           
class UserPreferences:
        def __init__(self, models, chat_history):
            self.models = models
            self.text = TextFormatting(chat_history, self.models)
            self.sentiment_analyzer = SentimentAnalyzer()
            self.classifier = ClassifyTopic(models, chat_history)

        async def process_text(self, chat_history, num_messages: int = 3):
            processed_results = []

            sentences = chat_history.get_recent_messages(num_messages)

            for sentence in sentences:
                try:
                    user_id = sentence["user_id"]
                    content = sentence["content"]

                    category = await self.classifier.classify_sentence(content)
                    if not category:
                        print(f"Could not classify category for message: {content}")
                        continue

                    absa_results = await self.models.openai_absa(user_id, content, category)
                    
                    for result in absa_results:
                        subject = result['user_id']
                        aspect = result['aspect']
                        sentiment = result['sentiment']
                        action = result.get('action', '')
                        preposition = result.get('preposition', '')
                        adverb = self.filter_adverb(result.get('adverb', ''))

                        if sentiment == 'favourite':
                            processed_sentence = f"{subject}'s favourite {category} is {aspect}"
                        else:
                            # Check if sentiment and action are similar
                            if self.are_sentiment_and_action_similar(sentiment, action):
                                components = [subject, adverb, sentiment]
                            else:
                                components = [subject, adverb, sentiment, action]
                            
                            # Handle preposition and aspect
                            if preposition and preposition.lower() != 'because':
                                components.extend([preposition, aspect])
                            else:
                                components.append(aspect)
                            
                            # Filter out None, null, empty string, and specific unwanted values
                            components = [str(comp) for comp in components if comp not in [None, '', 'null', 'None', 'none', 'it']]
                            
                            processed_sentence = ' '.join(components).strip()

                            # Check if the category is implied in the aspect
                            category_implied = await self.models.is_category_implied(aspect, category)
                            if not category_implied and category.lower() not in processed_sentence.lower():
                                processed_sentence += f" of {category}"

                        processed_results.append(processed_sentence)

                except Exception as e:
                    print(f"Error processing sentence: {sentence}")
                    print(f"Error details: {str(e)}")

            return processed_results
        
        def filter_adverb(self, adverb: str) -> str:
            # List of adverbs to remove
            adverbs_to_remove = ['also']
            
            if adverb.lower() in adverbs_to_remove:
                return ''
            return adverb

        def are_sentiment_and_action_similar(self, sentiment: str, action: str) -> bool:
            if not sentiment or not action:
                return False
            
            # Convert both to lowercase for comparison
            sentiment = sentiment.lower()
            action = action.lower()
            
            # Check if they're exactly the same
            if sentiment == action:
                return True
            
            # Check if one is a conjugated form of the other
            if sentiment.startswith(action) or action.startswith(sentiment):
                return True
            
            # You could add more specific checks here if needed
            # For example, checking for synonyms or related words
            
            return False