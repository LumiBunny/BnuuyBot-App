"""
Classes for STT and TTS.
"""

import queue
import threading
from audio_timer import AudioTimer
from azure_ai import Azure_AI

"""
A class for handling STT and audio transcriptions work queues.
"""

class STT:
    def __init__(self, audio_timeout: int = 15, history=None, chat=None, tts=None):
        self.tts = tts
        self.chat = chat
        self.history = history
        self.lock = threading.Lock()

        self.transcription = ['']
        self.last_transcription = ""
        self.is_listening = True

        # Initialize Azure TTS
        self.azure_ai = Azure_AI()

        # Initialize AudioTimer
        self.audio_timer = AudioTimer(history=self.history, chat=self.chat, tts=self.tts, timeout=audio_timeout)

        print("Ready!\n")
        print("Starting continuous listening...")
        self.azure_ai.start_continuous_listening(self.handle_transcription)

        # Start the initial timer
        self.audio_timer.start_timer()

    def handle_transcription(self, text):
        if text:
            with self.lock:
                self.transcription.append(text)
                self.last_transcription = text
            self.audio_timer.cancel_timer()
            print("Audio detected! Cancelling timer!")

    def stop(self):
        print("Stopping...")
        print(f"Transcription:\n {self.transcription}")
        self.is_listening = False
        self.azure_ai.stop_continuous_listening()
        self.audio_timer.cancel_timer()

    def get_last_transcription(self):
        with self.lock:
            text = self.last_transcription
            self.last_transcription = ""
        return text

class TTS:
    def __init__(self, tts_queue, azure_tts, history, chat):
        self.tts_queue = tts_queue
        self.azure_tts = azure_tts
        self.history = history
        self.chat = chat
        self.audio_timer = AudioTimer(history=self.history, chat=self.chat, tts=self)
        
    def tts_worker(self):
        processed_groups = []

        while True:
            try:
                tts_reply = self.tts_queue.get(timeout=1)
                if tts_reply is None:
                    break
                
                for group in tts_reply:
                    if group not in processed_groups:
                        self.azure_tts.azure_tts(group)
                        processed_groups.append(group)
                
                self.tts_queue.task_done()
                
                # Start the audio timer only if the queue is empty
                if self.tts_queue.empty():
                    print("Ready!!")
                    self.audio_timer.start_timer()

            except queue.Empty:
                continue

    def stop_tts_worker(self):
        self.tts_queue.put(None)

    def add_to_tts_queue(self, tts_reply):
        self.audio_timer.cancel_timer()
        for group in tts_reply:
            self.tts_queue.put(tts_reply)