"""
Description: A class for initializing LLM models and loading them to GPU.
Also includes functions and pipelines for getting LLM responses.
"""

import torch
import json
from openai import AsyncOpenAI, OpenAI
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import os
import re
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLMModels:
        def __init__(self):
                self.openai_key = os.getenv("OPENAI_API_KEY")
                self.openai = AsyncOpenAI(api_key=self.openai_key)

                self.embedder = SentenceTransformer('all-MiniLM-L6-v2', model_kwargs={"torch_dtype": "float16"}, device=device)
                self.summarizer = pipeline("summarization", 
                                           model="sshleifer/distilbart-cnn-6-6", 
                                           device=device)
                self.emotion_pipe = pipeline(task="text-classification", 
                                             model="SamLowe/roberta-base-go_emotions", 
                                             top_k=None, 
                                             device=device
                                             )
                self.classifier = pipeline(task="zero-shot-classification", 
                                       model="facebook/bart-large-mnli",
                                       device=device
                                       )
                self.sentiment_pipe = pipeline(task="sentiment-analysis", 
                                               model="sachin19566/distilbert_Yes_No_Other_Intent",
                                               device=device)
                # Set up LM Studio for chatting LLM
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                self.model = "Lewdiculous/Eris-Daturamix-7b-v2-GGUF-IQ-Imatrix"

        def cleanup(self):
            """ Clean up resources and free GPU memory."""
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Delete large objects and models
            if hasattr(self, 'embedder'):
                del self.embedder
            if hasattr(self, 'summarizer'):
                del self.summarizer
            if hasattr(self, 'emotion_pipe'):
                del self.emotion_pipe
            if hasattr(self, 'classifier'):
                del self.classifier
            if hasattr(self, 'sentiment_pipe'):
                del self.sentiment_pipe
            self.client = None
            self.model = None
            self.openai = None

        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cleanup()

        def get_embedder(self):
                return self.embedder
        
        def get_summarizer(self):
                return self.summarizer
        
        def get_classifier(self):
                return self.classifier
        
        def get_emotion(self, text):
                emotion = self.emotion_pipe(text)
                feeling = emotion[0]
                if feeling[0]['score'] > 0.68:
                    return feeling[0]['label']
                
        def get_intent(self, text):
                labels = ["question", "statement", "command", "remember that"]
                intent = self.classifier(text, labels)
                if intent['scores'][0] > 0.5:
                    return intent['labels'][0]
                
        def get_decision(self, text):
            try:
                result = self.sentiment_pipe(text)[0]  # Get first result
                return result['label']  # This will return 'Yes', 'No', or 'Other'
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                return 'Other'  # Default fallback
        
        def lm_studio_client(self):
                return self.client
        
        def get_llm(self):
                return self.model
        
        async def openai_classifier(self, text: str, category: list[str], category_mapping: dict = None) -> list[str]:
            functions = [
                {
                    "name": "classify",
                    "description": "Classifies a given text into one or more categories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": category,
                                "description": "List of categories to classify the text into"
                            }
                        },
                        "required": ["category"]
                    }
                }
            ]

            system_message = "You are a helpful assistant that classifies a given text into predefined categories."
            if category_mapping:
                 system_message += "Here are some keywords associated with each category:\n"
                 for category, keywords in category_mapping.items():
                      system_message += f"{category}: {', '.join(keywords[:10])}\n"

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Please classify the following text into one of the predefined categories: {text}"}
            ]

            try:
                completion = await self.openai.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                    functions=functions,
                    function_call={"name": "classify"}
                )

                # Extract the category from the completion response
                function_args = json.loads(completion.choices[0].message.function_call.arguments)
                categories = function_args.get('category', [])
                
                if not categories:
                    print("No categories were returned by the classifier")
                
                return categories

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return []
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return []
            
        async def openai_absa(self, user_id: str, text: str, category: str) -> list[dict[str, str]]:
            # Split the input text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            results = []
            
            for sentence in sentences:
                functions = [
                    {
                        "name": "extract_aspect_sentiments",
                        "description": "Extracts aspects and their associated actions, adverbs and prepositions from the given sentence, identifying the correct user for each action",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "aspects": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "user_id": {"type": "string", "description": "The ID of the user who performed this action. Use the provided user_id for first-person references, or the name mentioned for other people."},
                                            "aspect": {"type": "string", "description": "The specific aspect or feature mentioned"},
                                            "sentiment": {
                                                "type": "string", 
                                                "enum": ["favourite", "loves", "likes", "enjoys", "prefers","dislikes", "hates"], 
                                                "description": "The sentiment associated with the aspect, with 'favourite' being the strongest positive. If sentiment is not listed here, find closest matching sentiment. If no sentiment is found, use 'has mentioned'."
                                            },
                                            "adverb": {"type": "string", "description": "The adverb used to describe the sentiment, if any. If there is no sentiment, do not make one up, leave it empty."},
                                            "action": {"type": "string", "description": "The action associated with the aspect, if any. If there is no action, do not make one up, leave it empty."},
                                            "preposition": {"type": "string", "description": "The preposition used with the action, if any. If there is no preposition, do not make one up, leave it empty."}
                                        },
                                        "required": ["user_id", "aspect", "sentiment"]
                                    }
                                }
                            },
                            "required": ["aspects"]
                        }
                    }
                ]

                messages = [
                    {"role": "system", "content": f"""You are an expert in aspect-based analysis for the {category} domain. Extract aspects, actions, adverbs, and prepositions from the given sentence, focusing on {category}-related aspects.

                    Important rules:
                    1. The user '{user_id}' is the default speaker. Any first-person references ('I', 'me', 'my', etc.) refer to {user_id} unless explicitly stated otherwise.
                    2. If other names are mentioned, attribute the sentiment/action to them instead of {user_id}.
                    3. Each aspect should be a separate item in the result, even if multiple aspects are mentioned for the same user.
                    4. If an aspect is implied but not explicitly mentioned, use a general term (e.g., 'food' for pizza, hamburgers, sushi).
                    """},
                    {"role": "user", "content": sentence}
                ]

                try:
                    completion = await self.openai.chat.completions.create(
                        model="gpt-4-0613",
                        messages=messages,
                        functions=functions,
                        function_call={"name": "extract_aspect_sentiments"}
                    )

                    response = json.loads(completion.choices[0].message.function_call.arguments)
                    results.extend(response.get('aspects', []))

                except Exception as e:
                    print(f"Error in openai_absa: {e}")

            return results
            
        async def is_category_implied(self, aspect: str, category: str) -> bool:
            prompt = f"""
            Given the aspect "{aspect}" and the category "{category}", determine if the category is implied in the aspect.
            Return True if the category is implied, and False otherwise.
            
            Examples:
            Aspect: "pepperoni", Category: "food" -> True (pepperoni implies food)
            Aspect: "red", Category: "color" -> False (red doesn't necessarily imply color in all contexts)
            """
            
            messages = [
                {"role": "system", "content": "You are an AI assistant that determines if a category is implied in an aspect."},
                {"role": "user", "content": prompt}
            ]

            try:
                completion = await self.openai.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                    max_tokens=5
                )
                result = completion.choices[0].message.content.strip().lower()
                return result == "true"
            except Exception as e:
                print(f"Error in is_category_implied: {e}")
                return False