"""
Classes and functions for managing messages, chat history, self prompting and text transformations.
"""

import datetime
import os
import re
import json
import torch
import torch.nn.functional as F

class ChatHistory:
# A class for managing the chat history between users and LLM.
    def __init__(self):
        self.history = []

    def get_length(self):
        return len(self.history)
    
    def get_most_recent(self):
        return self.history[-1] if self.history else None
    
    def get_recent_messages(self, num):
        """Return the last n messages from the chat history."""
        recent_messages = self.history[-num:] if num <= len(self.history) else self.history
        # Ensure each message has the correct format
        return [{"role": msg["role"], "content": msg["content"]} for msg in recent_messages]

    def get_content(self):
        recent = self.get_most_recent()
        return recent.get("content") if recent else None
    
    def get_user(self):
        recent = self.get_most_recent()
        return recent.get("role") if recent else None
    
    def delete_most_recent(self):
        self.history.pop()

    def add(self, user, user_id, content):
        self.history.append({"role": user, "user_id": user_id, "content": content})

class SelfPrompt:
    def __init__(self, history):
        self.transcription_processing = True
        if isinstance(history, list):
            self.history = ChatHistory()
            for item in history:
                self.history.add(item.get("role"), item.get("user_id"), item.get("content"))
        elif isinstance(history, ChatHistory):
            self.history = history
        else:
            raise TypeError("history must be a ChatHistory object or a list")

    def self_prompt(self):
        last = self.history.get_content()
        if self.history.get_user() == 'assistant':
            question = self.is_question(last)
            len = self.history.get_length()
            if len >= 2:    
                if (len%5) == 0:
                    # Shake it up.
                    print("MULTIPLE OF 5, CHANGE TOPIC")
                    self.change_topic()  
                    return True
                
                elif question:
                    print("IS_QUESTION = TRUE")
                    print(question)
                    self.answer_own_question()  
                    return True

                else:
                    print("CONTINUE THOUGHTS")
                    self.continue_thoughts()   
                    return True

            elif question:
                print("IS_QUESTION = TRUE")    
                print(question)    
                self.answer_own_question()  
                return True
            else:
                print("LUMI HASN'T SPOKEN YET, START A TOPIC")    
                self.start_a_topic()  
                return True
        elif self.history.get_user() == 'system':
            print("GREET LUMI")
            self.greet_lumi()
            return True
        else:
            print("Unknown role in the most recent message.")
            return False

    def is_question(self, paragraph):
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        
        for sentence in sentences:
            stripped_sentence = sentence.strip()       
            if stripped_sentence.endswith('?') or any (stripped_sentence.lower().startswith(word) for word in ["who", "what", "where", "when", "why", "how"]):
                return stripped_sentence  # Return the first question found
        
        return None  # Return None if no question is found
    
    # The prompts
    def greet_lumi(self):
        self.history.add("system", "System", "Lumi hasn't said anything yet, greet her.")
    
    def start_a_topic(self):
        self.history.add("system", "System", "Lumi hasn't said anything yet. Talk about something you love to do. Speak as though you prompted this yourself and this was not a message from Lumi.")
    
    def change_topic(self):
        self.history.add("system","System","Change the topic of the conversation. Speak as though you prompted this yourself and this was not a message from Lumi.")

    def answer_own_question(self):
        self.history.add("user", "System", "Answer the question you asked in your previous message. Do not ask a question in your response. Answer as though you prompted this yourself and this was not a message from Lumi.")
    
    def continue_thoughts(self):
        self.history.add("system", "System", "Continue your thoughts on the previous message.  Speak as though you prompted this yourself and this was not a message from Lumi.")

class TextFormatting:
    def __init__(self, chat_history, models):
        self.chat_history = chat_history
        self.models = models
        self.summarizer = models.get_summarizer()

    # Summarize the context of the message or conversation history
    def get_context(self, num):
        # Step 1: Get recent messages
        context = self.chat_history.get_recent_messages(num)

        # Step 2: Check if context is a single string, if not, convert/combine it
        if not isinstance(context, str):
            if isinstance(context, list):
                context = " ".join([f"{getattr(msg, 'role', 'Unknown')}: {getattr(msg, 'content', str(msg))}" for msg in context])
            else:
                context = str(context)

        # Step 3: Summarization pipeline
        max_length = min(len(context.split()) // 2, 130)  # Aim for half the original length, max 130 tokens
        min_length = max(30, max_length // 2)  # At least 30 tokens, or half of max_length

        summary = self.summarizer(
            context,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        # Step 4: Return result as a STRING
        return summary[0]['summary_text']
    
    def strip_emoji(self, text):
        # Remove emojis from text for better tts.
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        return RE_EMOJI.sub(r'', text)

    # TTS is wonky, added filter to change words so that the TTS can pronounce things correctly, or remove unnecessary words/text.
    def bnuuybot_reply_filter(self, text):
        if isinstance(text, list):
            # If text is a list of dictionaries, extract the 'content' from each dictionary
            text = ' '.join(item['content'] for item in text if 'content' in item)
        elif isinstance(text, dict):
            # If text is a single dictionary, extract the 'content'
            text = text.get('content', '')
        
        # Now proceed with the replacements
        text = text.replace("Live2D", "live 2D")
        text = text.replace("✨", "")
        text = text.replace("⛰", "")
        return text

    # Split text into sentences to allow for shorter gen times when generating TTS.
    def split_into_sentences(self, paragraph):
            # Split by '.', '!', and '?' and filter out empty strings
            sentences = []
            for sentence in re.split(r'[.!?]', paragraph):
                    stripped_sentence = sentence.strip()
                    if stripped_sentence:  # Check if the string is not empty
                            sentences.append(stripped_sentence)
            return sentences

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
class ChatLog:
    def __init__(self):
          self.chat_log = []
          self.filename = f'.logs/chat_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    def update_chat_log(self, prompt, reply):
        self.filename  
    
        entry = [
             {"role": "user", "prompt": prompt},
             {"role": "assistant", "response": reply}
        ]
        
        # Check if the file exists and load existing data
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)

        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=4)
            print(f"Appended to {self.filename}")  # For debugging