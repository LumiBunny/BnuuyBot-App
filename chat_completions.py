"""
Description: A class for handling the chat completions with the local LLM.
"""

from messages import TextFormatting

class Completions:
    def __init__(self,
                 chat_history,
                 models):
        self.models = models
        self.client = self.models.lm_studio_client()
        self.model = self.models.get_llm()
        self.chat_history = chat_history
        #self.chat_log = chat_log
        self.text_formatting = TextFormatting(chat_history, models)

    def bnuuybot_completion(self):
        try:
            reply = ""
            messages = self.chat_history.get_recent_messages(20)  # Adjust the number as needed

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.35
            )
            reply = completion.choices[0].message.content
            return reply
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            # Log the error, maybe retry, or handle it appropriately