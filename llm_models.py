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
                self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)
                self.emotion_pipe = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)
                
                # Set up LM Studio for chatting LLM
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                self.model = "LumiBunny/Eris-Daturamix-Neural-7b-slerp-IQ4_NL-GGUF"

        def get_embedder(self):
                return self.embedder
        
        def get_summarizer(self):
                return self.summarizer
        
        def get_emotion(self, text):
                emotion = self.emotion_pipe(text)
                feeling = emotion[0]
                if feeling[0]['score'] > 0.68:
                    return feeling[0]['label']
        
        def lm_studio_client(self):
                return self.client
        
        def get_llm(self):
                return self.model