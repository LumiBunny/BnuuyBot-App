"""
Description: A timer for when no input audio is detected after x seconds.
"""

import threading
from messages import SelfPrompt, ChatHistory

class AudioTimer:
    def __init__(self, history, chat, tts, timeout=15, callback=None):
        if not isinstance(history, ChatHistory):
            raise TypeError("history must be a ChatHistory object")
        self.history = history
        self.chat = chat
        self.tts = tts
        self.timeout = timeout
        self.timer = None
        self.callback = callback
        self.prompt = SelfPrompt(self.history)
        self.lock = threading.Lock()
        self.is_timer_active = False  # New flag to track if timer is active

    def start_timer(self):
        with self.lock:
            if not self.is_timer_active:
                print("Starting new audio timer...")
                self.timer = threading.Timer(self.timeout, self._timer_completed)
                self.timer.start()
                self.is_timer_active = True
            else:
                print("Timer already running, not starting a new one.")

    def cancel_timer(self):
        with self.lock:
            if self.is_timer_active:
                print("Cancelling audio timer...")
                self.timer.cancel()
                self.timer = None
                self.is_timer_active = False
            else:
                print("No active timer to cancel.")

    def _timer_completed(self):
        with self.lock:
            self.is_timer_active = False
            self.timer = None
        self.no_audio_detected()

    def no_audio_detected(self):
        if self.history is not None:
            if self.tts.tts_queue.empty():
                print("No audio detected! Running self prompt...")
                if self.prompt.self_prompt():
                    tts_reply = self.chat.bnuuybot_completion()
                    self.tts.add_to_tts_queue(tts_reply)
            else:
                self.tts.cancel_audio_timer()
        else:
            print("No history available.")