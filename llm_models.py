"""
Description: A class for getting the required LLM.
"""

import torch
from openai import OpenAI
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLMModels:
        def __init__(self):
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2', model_kwargs={"torch_dtype": "float16"}, device=device)
                self.summarizer = pipeline("summarization", 
                                           model="sshleifer/distilbart-cnn-6-6", 
                                           device=device)
                self.emotion_pipe = pipeline(task="text-classification", 
                                             model="SamLowe/roberta-base-go_emotions", 
                                             top_k=None, 
                                             device=device
                                             )
                self.intent_pipe = pipeline(task="zero-shot-classification", 
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
        
        def get_emotion(self, text):
                emotion = self.emotion_pipe(text)
                feeling = emotion[0]
                if feeling[0]['score'] > 0.68:
                    return feeling[0]['label']
                
        def get_intent(self, text):
                labels = ["question", "statement", "command", "remember that"]
                intent = self.intent_pipe(text, labels)
                if intent['scores'][0] > 0.5:
                    return intent['labels'][0]
                
        def get_sentiment(self, text):
            try:
                result = self.sentiment_pipe(text)[0]  # Get first result
                return result['label']  # This will return 'Yes', 'No', or 'Other'
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                return 'Other'  # Default fallback
            
        def get_alternative_sentiment(self, text):
            try:
                results = self.sentiment_pipe(text)  # Get all results
            
                # If no results, return Other
                if not results:
                    return 'Other'
            
                primary_label = results[0]['label']
            
                 # If primary result isn't 'Other', return it
                if primary_label != 'Other':
                    return primary_label
            
                # If we have multiple results and primary is 'Other',
                # return the next highest confidence label
                if len(results) > 1:
                    return results[1]['label']
                
                return 'Other'  # Default if no alternative found
            
            except Exception as e:
                print(f"Error in alternative sentiment analysis: {e}")
                return 'Other'
        
        def lm_studio_client(self):
                return self.client
        
        def get_llm(self):
                return self.model