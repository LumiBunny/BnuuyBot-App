"""
Description: A class for getting the required LLM.
"""

import torch
from openai import OpenAI
from transformers import pipeline
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLMModels:
        def __init__(self):
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                self.classifier = pipeline("zero-shot-classification", model="FacebookAI/roberta-large-mnli", device=device)
                # Set up LM Studio for chatting LLM
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                self.model = "Lewdiculous/Eris-Daturamix-7b-v2-GGUF-IQ-Imatrix"

        def get_embedder(self):
                return self.embedder
        
        def get_classifier(self):
                return self.classifier
        
        def lm_studio_client(self):
                return self.client
        
        def get_llm(self):
                return self.model