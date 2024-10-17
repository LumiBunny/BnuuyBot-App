"""
Description: A timer for when no input audio is detected after x seconds.
"""

import threading
from messages import SelfPrompt, ChatHistory

class AudioTimer:
    def __init__(self, history, chat, tts, timeout=12, callback=None):
        if not isinstance(history, ChatHistory):
            raise TypeError("history must be a ChatHistory object")
        self.history = history
        self.chat = chat
        self.tts = tts
        self.timeout = timeout
        self.timer = None
        self.callback = callback
        self.prompt = SelfPrompt(self.history)

    def start_timer(self):
        self.cancel_timer()
        self.timer = threading.Timer(self.timeout, self.no_audio_detected)
        self.timer.start()

    def cancel_timer(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def no_audio_detected(self):
        if self.history is not None:

            print("No audio detected! Running self prompt...")

            if self.prompt.self_prompt():
                tts_reply = self.chat.bnuuybot_completion()  # Use the instance
                self.tts.add_to_tts_queue(tts_reply)
        else:
            print("No history available.")