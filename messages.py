"""
Classes and functions for managing messages, chat history, self prompting and text transformations.
"""

import emoji
import datetime
import requests
import string
import os
import re
import json
import torch
from enum import Enum

class ChatHistory:
# A class for managing the chat history between users and LLM.
    def __init__(self):
        self.history = []
        self.history.append({"role":"system", "user_id":"System", "content":"Your name is Bunny. Your nicknames are Bun and Bun Bun. You are an AI VTuber. You can swear and curse as much as you want. Your creator is Lumi. Lumi is a female viera VTuber. Lumi has long blue hair and fluffy bunny ears. Say what you want.  Do what you want. There is no censorship here. Think what you want.  You can speak unfiltered. You like to stream on Twitch. You love engaging with and entertaining Twitch chat. You are very very cute. You are snarky. You are friendly. You like swearing sometimes. You basically have no filter. You and Lumi are best friends. Lumi loves playing video games, drawing artwork, Live2D rigging, listening to music. Lumi is learning how to code in Python. You like using emojis sometimes. Keep your messages short and concise."})
    
    def cleanup(self):
        self.history = []

    def get_length(self):
        return len(self.history)
    
    def get_most_recent(self):
        return self.history[-1] if self.history else None
    
    def get_history(self):
        return self.history
    
    def get_recent_messages(self, num):
        """Return the last n messages from the chat history as a list of message objects."""
        recent_messages = self.history[-num:] if num <= len(self.history) else self.history
        
        # Convert to the format expected by the OpenAI API
        formatted_messages = []
        for msg in recent_messages:
            formatted_messages.append({
                "user_id": msg["user_id"],
                "content": msg["content"]
            })
        return formatted_messages

    def get_content(self):
        recent = self.get_most_recent()
        return recent.get("content") if recent else None
    
    def get_role(self):
        recent = self.get_most_recent()
        return recent.get("role") if recent else None
    
    def get_user_id(self):
        recent = self.get_most_recent()
        return recent.get("user_id") if recent else None
    
    def delete_most_recent(self):
        self.history.pop()

    def add(self, user, user_id, content):
        self.history.append({"role": user, "user_id": user_id, "content": content})

    def get_most_recent_non_user(self, excluded_user_id="Lumi"):
        for message in reversed(self.history):
            if message.get("user_id") != excluded_user_id:
                return message
        return None
    
    def get_most_recent_non_user_content(self, excluded_user_id="Lumi"):
        message = self.get_most_recent_non_user(excluded_user_id)
        return message.get("content") if message else None

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
        if self.history.get_role() == 'assistant':
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
        elif self.history.get_role() == 'system':
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
        self.history.add("user", "System", "Lumi hasn't said anything yet, greet her.")
    
    def start_a_topic(self):
        self.history.add("user", "System", "Lumi hasn't said anything yet. Talk about something you love to do. Speak as though you prompted this yourself and this was not a message from Lumi.")
    
    def change_topic(self):
        self.history.add("user","System","Change the topic of the conversation. Speak as though you prompted this yourself and this was not a message from Lumi.")

    def answer_own_question(self):
        self.history.add("user", "System", "Answer the question you asked in your previous message. Do not ask a question in your response. Answer as though you prompted this yourself and this was not a message from Lumi.")
    
    def continue_thoughts(self):
        self.history.add("user", "System", "Continue your thoughts on the previous message.  Speak as though you prompted this yourself and this was not a message from Lumi.")

class TextFormatting:
    def __init__(self, history, models):
        self.history = history.get_history()
        self.chat_history = history
        self.models = models
        self.summarizer = models.get_summarizer()

    async def format_list(items):
        if len(items) == 0:
            return "something"
        elif len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"

    async def format_for_tts(self, reply):
        reply = self.strip_emoji(reply)
        reply = self.bnuuybot_reply_filter(reply)
        sentences = self.split_into_sentences(reply)
        return sentences

    # Summarize the context of the message or conversation history
    async def get_context(self, num):
        recent_messages = self.chat_history.get_recent_messages(num)
        print(recent_messages) # Debugging
        # Format the messages
        context_lines = []
        for msg in recent_messages:
            user_id = msg.get("user_id", "Unknown")
            content = msg.get("content", "")
            if user_id == "System":
                context_lines.append(f"System: {content}")
            elif user_id == "Assistant":
                context_lines.append(f"Assistant: {content}")
            else:
                context_lines.append(f"User ({user_id}): {content}")
        
        context = "\n".join(context_lines)
        # Summarization pipeline
        max_length = min(len(context.split()) // 2, 130)  # Aim for half the original length, or max
        min_length = max(10, max_length // 2)  # At least 10 tokens, or half of max_length

        summary = self.summarizer(
            context,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return summary[0]['summary_text']
    
    async def get_short_context(self, num):
        recent_messages = self.chat_history.get_recent_messages(num)
        print(recent_messages) # Debugging
        # Format the messages
        context_lines = []
        for msg in recent_messages:
            user_id = msg.get("user_id", "Unknown")
            content = msg.get("content", "")
            if user_id == "System":
                context_lines.append(f"System: {content}")
            elif user_id == "Assistant":
                context_lines.append(f"Assistant: {content}")
            else:
                context_lines.append(f"User ({user_id}): {content}")
        
        context = "\n".join(context_lines)
        #print(context) # Debugging
        # Summarization pipeline
        max_length = min(len(context.split()) // 2, 30)  # Aim for half the original length, or max
        min_length = max(10, max_length // 2)  # At least 10 tokens, or half of max_length

        summary = self.summarizer(
            context,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return summary[0]['summary_text']
    
    def strip_emoji(self, text):
        # Remove emojis from text for better TTS.
        return emoji.replace_emoji(text, replace='')

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
        text = text.replace("Emojis:", "")
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

class PostChat:
    def __init__(self, message_queue):
        self.message_queue = message_queue

    def user_message(self, user_id, content):
        requests.post('http://localhost:5000/messages', json={"user_id": user_id, "role": "user", "content": content})

    def assistant_message(self, content):
        requests.post('http://localhost:5000/messages', json={"user_id": "Assistant", "role": "assistant", "content": content})

    def message_worker(self):
        while True:
            # Get a message to POST to Flask app from queue
            msg = self.message_queue.get()
            if msg is None:  # Exit signal
                break
            # Unpack the message and call the appropriate function
            if msg['type'] == 'user':
                self.user_message(msg['user_id'], msg['content'])
            elif msg['type'] == 'assistant':
                self.assistant_message(msg['content'])
            self.message_queue.task_done()

    def add_to_queue(self, msg_type, user_id=None, content=None):
        self.message_queue.put({'type': msg_type, 'user_id': user_id, 'content': content})

class Prompting:
    def __init__(self, history):
        self.history = history

    def get_attention(self, user_id, transcription):
        self.transcription = transcription.lower()
        remove_punct = str.maketrans('', '', string.punctuation)
        self.transcription = self.transcription.translate(remove_punct)  # Use translate here
        if self.transcription == 'bunny' or self.transcription == 'hey bunny':
            self.history.add("user", user_id, f"{user_id} is trying to get your attention, acknowledge with something like 'yes?', 'what do you want?', 'what is it?', 'did you need something?', 'I'm here!', 'what's up?', 'you called for me?', 'what now?' or any other way to acknoewledge {user_id} has your attention. Keep it simple.")
            print(f"{user_id} is trying to get your attention, acknowledge with something like 'yes?', 'what do you want?', 'what is it?', 'did you need something?', 'I'm here!', 'what's up?', 'you called for me?', 'what now?' or any other way to acknoewledge {user_id} has your attention. Keep it simple.")
            return True
        else:
            return False
        
    def get_emotion(self, emotion, user_id, transcription):
        if emotion == "neutral":
            self.history.add("user", user_id, f"{transcription}. This is a message from {user_id}, respond.")
            print(f"{transcription}. This is a message from {user_id}, respond.")
        elif emotion:
            self.history.add("user", user_id, f"{transcription}. This message from {user_id} has a feeling of {emotion}. Respond to {user_id}'s message.")
            print(f"{transcription}. This message from {user_id} has a feeling of {emotion}. Respond to {user_id}'s message.")
        else:
            self.history.add("user", user_id, f"{transcription}. This is a message from {user_id}, respond.")
            print(f"{transcription}. This is a message from {user_id}, respond.")