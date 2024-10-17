"""
Description: A class for handling the chat completions with the local LLM.
"""

import requests
from messages import TextFormatting

class Completions:
    def __init__(self, chat_history, models):
        self.models = models
        self.client = self.models.lm_studio_client()
        self.model = self.models.get_llm()
        self.chat_history = chat_history
        self.text_formatting = TextFormatting(chat_history, models)

    def bnuuybot_completion(self):
        completed_generation = False
        
        while not completed_generation:
            try:
                messages = self.chat_history.get_recent_messages(20)  # Adjust the number as needed

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.8,
                    stream=True,
                )

                new_message = {"role": "assistant", "content": ""}

                print("Bnuuy Bot: ")
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                        new_message["content"] += chunk.choices[0].delta.content

                tts_reply = new_message["content"]

                # Check if tts_reply is empty before proceeding
                if not tts_reply.strip():  # If the reply is empty or contains only whitespace
                    print("Warning: Context reply is empty. Skipping TTS.")
                    print("Ready!")
                    return
                
                completed_generation = True

                requests.post('http://localhost:5000/messages', json={"user_id": "Assistant", "role": "assistant", "content": new_message["content"]})

                tts_reply = self.text_formatting.strip_emoji(tts_reply)
                tts_reply = self.text_formatting.bnuuybot_reply_filter(tts_reply)
                sentence_groups = self.text_formatting.split_into_sentences(tts_reply)

                # Delete system message
                self.chat_history.delete_most_recent()
                # Add the assistant's response to the chat history
                self.chat_history.add("assistant", "Assistant", new_message["content"])

                return sentence_groups
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                # Log the error, maybe retry, or handle it appropriately