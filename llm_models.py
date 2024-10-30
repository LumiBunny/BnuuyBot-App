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
        
        async def use_openai_functions(self, text: str) -> list[str]:
            """
            Calls functions in the via OpenAI API. 
            Example: to extract food items from a given text.
            """
            functions = [
                {
                    "name": "extract_food_items",
                    "description": "Extracts food items from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "food_items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of food items extracted from the text"
                            }
                        },
                        "required": ["food_items"]
                    }
                }
            ]
            try:
                completion = await self.openai.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts food items from text."},
                        {"role": "user", "content": f"Extract all food items from the following text: {text}"}
                    ],
                        functions=functions,
                        function_call={"name": "extract_food_items"}
                )

                function_call = completion.choices[0].message.function_call
                if function_call and function_call.name == "extract_food_items":
                    food_items = json.loads(function_call.arguments)["food_items"]
                    return food_items if food_items else ["None"]
                else:
                    print("Function was not called as expected")
                    return ["None"]
            except Exception as e:
                print(f"An error occurred: {e}")
                return ["None"]