"""
Description: A class for handling the chat completions with the local LLM.
"""

import asyncio
from messages import TextFormatting

class Completions:
    def __init__(self, chat_history, models, chat_log, post):
        self.models = models
        self.client = self.models.lm_studio_client()
        self.model = self.models.get_llm()
        self.chat_history = chat_history
        self.chat_log = chat_log
        self.post = post
        self.text_formatting = TextFormatting(chat_history, models)

    async def bnuuybot_completion(self):
        completed_generation = False
        
        while not completed_generation:
            try:
                messages = self.chat_history.get_recent_messages(20)  # Adjust the number as needed
                user_input = self.chat_history.get_content()

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

                reply = new_message["content"]
                tts_reply = asyncio.create_task(self.text_formatting.format_for_tts(reply))

                # Check if tts_reply is empty before proceeding
                if not reply.strip():  # If the reply is empty or contains only whitespace
                    print("Warning: Context reply is empty. Skipping TTS.")
                    print("Ready!")
                    return
                
                completed_generation = True

                self.post.add_to_queue("assistant", "Assistant", reply)
                # Delete system message
                self.chat_history.delete_most_recent()
                # Add the assistant's response to the chat history
                self.chat_history.add("assistant", "Assistant", new_message["content"])
                self.chat_log.update_chat_log(user_input, new_message["content"])
                sentence_groups = await tts_reply
                return sentence_groups
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                # Log the error, maybe retry, or handle it appropriately