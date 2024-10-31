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
        
        async def use_openai_functions(self, text: str, function_name: str) -> list[str]:
            """
            Calls functions in the via OpenAI API. 
            Example: to extract food items from a given text.
            """
            #region OpenAI Functions
            functions = [
                #region Food Items
                {
                    "name": "extract_food_items",
                    "description": "Extracts food items and associated actions from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "food_items": {
                                "type": "array",
                                "items": {"type": "object",
                                        "properties": {
                                            "item": {"type": "string"},
                                            "action": {"type": "string"}
                                            }
                                        },
                                "description": "List of food items and associated actions extracted from the text"
                            }
                        },
                        "required": ["food_items"]
                    }
                },
                #endregion

                #region Hobbies
                {
                    "name": "extract_hobbies",
                    "description": "Extracts hobbies and associated actions from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "hobbies": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "hobby": {"type": "string"},
                                        "action": {"type": "string"}
                                        }
                                    },
                                "description": "List of hobbies extracted from the text"
                            }
                        },
                        "required": ["hobbies"]
                    }
                },
                #endregion

                #region Music
                {
                    "name": "extract_music",
                    "description": "Extracts music-related preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "music_preferences": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["genre", "song", "artist", "instrument", "other"]},
                                        "value": {"type": "string"},
                                        "action": {"type": "string", "description": "Associated action or sentiment, if any"}
                                    },
                                    "required": ["type", "value"]
                                },
                                "description": "List of music-related preferences extracted from the text"
                            }
                        },
                        "required": ["music_preferences"]
                    }
                },
                #endregion

                #region Video Games
                {
                    "name": "extract_video_games",
                    "description": "Extracts video game preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "video_games": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "game": {"type": "string"},
                                        "action": {"type": "string"}
                                    },
                                    "required": ["game"]
                                },
                                "description": "List of video games extracted from the text"
                            }
                        },
                        "required": ["video_games"]
                    }
                },
                #endregion

                #region Streaming
                {
                    "name": "extract_streaming",
                    "description": "Extracts streaming-related preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "streaming_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item": {"type": "string"},
                                        "action": {"type": "string"}
                                    },
                                    "required": ["item"]
                                },
                                "description": "List of streaming-related items extracted from the text"
                            }
                        },
                        "required": ["streaming_items"]
                    }
                },
                #endregion

                #region VTubing
                {
                    "name": "extract_vtubing",
                    "description": "Extracts VTubing-related preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "vtubing_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item": {"type": "string"},
                                        "action": {"type": "string"}
                                    },
                                    "required": ["item"]
                                },
                                "description": "List of VTubing-related items extracted from the text"
                            }
                        },
                        "required": ["vtubing_items"]
                    }
                },
                #endregion

                #region Live2D Rigging
                {
                    "name": "extract_live2d_rigging",
                    "description": "Extracts Live2D rigging-related preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "live2d_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item": {"type": "string"},
                                        "action": {"type": "string"}
                                    },
                                    "required": ["item"]
                                },
                                "description": "List of Live2D rigging-related items extracted from the text"
                            }
                        },
                        "required": ["live2d_items"]
                    }
                },
                #endregion

                #region Drawing
                {
                    "name": "extract_drawing",
                    "description": "Extracts drawing-related preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "drawing_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item": {"type": "string"},
                                        "action": {"type": "string"}
                                    },
                                    "required": ["item"]
                                },
                                "description": "List of drawing-related items extracted from the text"
                            }
                        },
                        "required": ["drawing_items"]
                    }
                },
                #endregion

                #region Coding
                {
                    "name": "extract_coding",
                    "description": "Extracts coding-related preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "coding_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item": {"type": "string"},
                                        "action": {"type": "string"}
                                    },
                                    "required": ["item"]
                                },
                                "description": "List of coding-related items extracted from the text"
                            }
                        },
                        "required": ["coding_items"]
                    }
                },
                #endregion

                #region Cooking
                {
                    "name": "extract_cooking",
                    "description": "Extracts cooking-related preferences from a given text",
                    "parameters": {
                        "type": "object",
                        "properties": {    
                            "cooking_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item": {"type": "string"},
                                        "action": {"type": "string"}
                                    },
                                    "required": ["item"]
                                },
                                "description": "List of cooking-related items extracted from the text"
                            }
                        },
                        "required": ["cooking_items"]
                    }
                },
                #endregion

                #region Projects
                    {
                        "name": "extract_projects",
                        "description": "Extracts project-related preferences from a given text",
                        "parameters": {
                            "type": "object",
                            "properties": {    
                                "project_items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "item": {"type": "string"},
                                            "action": {"type": "string"}
                                        },
                                        "required": ["item"]
                                    },
                                    "description": "List of project-related items extracted from the text"
                                }
                            },
                            "required": ["project_items"]
                        }
                    }
                    #endregion
                ]
                #endregion OpenAI Functions
    
            # System messages for each function
            system_messages = {
                "extract_food_items": "You are a helpful assistant that extracts food items from text.",
                "extract_hobbies": "You are a helpful assistant that extracts hobbies and associated actions from text.",
                "extract_music": "You are a helpful assistant that extracts music-related preferences from text.",
                "extract_video_games": "You are a helpful assistant that extracts video game preferences from text.",
                "extract_streaming": "You are a helpful assistant that extracts streaming-related preferences from text.",
                "extract_vtubing": "You are a helpful assistant that extracts VTubing-related preferences from text.",
                "extract_live2d_rigging": "You are a helpful assistant that extracts Live2D rigging-related preferences from text.",
                "extract_drawing": "You are a helpful assistant that extracts drawing-related preferences from text.",
                "extract_coding": "You are a helpful assistant that extracts coding-related preferences from text.",
                "extract_cooking": "You are a helpful assistant that extracts cooking-related preferences from text.",
                "extract_projects": "You are a helpful assistant that extracts project-related preferences from text."
            }

            try:
                completion = await self.openai.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": system_messages[function_name]},
                        {"role": "user", "content": f"Extract all food items from the following text: {text}"}
                    ],
                        functions=functions,
                        function_call={"name": "extract_food_items"}
                )

                function_call = completion.choices[0].message.function_call
                if function_call and function_call.name == function_name:
                    result = json.loads(function_call.arguments)
                    if function_name == "extract_food_items":
                        return [{"item": item} for item in result["food_items"]] if result["food_items"] else [{"item": "None"}]
                    elif function_name == "extract_hobbies":
                        return result["hobbies"] if result["hobbies"] else [{"hobby": "None", "action": "None"}]
                    elif function_name == "extract_music":
                        return result["music_preferences"] if result["music_preferences"] else [{"type": "None", "value": "None", "action": "None"}]
                else:
                    print("Function was not called as expected")
                    return [{"item": "None"}] if function_name == "extract_food_items" else [{"hobby": "None", "action": "None"}]
            except Exception as e:
                print(f"An error occurred: {e}")
                return [{"item": "None"}] if function_name == "extract_food_items" else [{"hobby": "None", "action": "None"}]