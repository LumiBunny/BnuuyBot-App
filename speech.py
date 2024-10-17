"""
Classes for STT and TTS.
"""

import queue
import random
import threading
import logging
import time
import io
from audio_timer import AudioTimer
import speech_recognition as sr
from faster_whisper import WhisperModel
import pyaudio

"""
A class for handling STT and audio transcriptions work queues.
"""

class STT:

    #Real-time Speech to Text class using Faster WhisperModel and speech_recognition.

    # Initialize STT model
    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16",
                 language: str = "en", logging_level: str = None, audio_timeout: int = 5, history=None, chat=None, tts=None):
        self.tts = tts
        self.chat=chat
        self.history = history
        self.audio_timeout = audio_timeout
        self.lock = threading.Lock()
        self.audio_timer = AudioTimer(history=self.history,chat=self.chat, tts=self.tts)  # Initialize the audio timer

        #Initialize the STT object.
        self.recorder = sr.Recognizer()
        self.data_queue = queue.Queue()
        self.transcription = ['']
        self.last_transcription = ""
        self.is_listening = True

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.default_mic = self.setup_mic()

        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        if logging_level:
            self.configure_logging(level=logging_level)

        self.thread = threading.Thread(target=self.transcribe)
        self.thread.setDaemon(True)
        self.thread.start()

        print("Ready!\n")

        print("Starting timer!")
        self.audio_timer.start_timer()  # Start the timer immediately

    def transcribe(self):
        """Transcribe the audio data from the queue."""
        while self.is_listening:
            audio_data = self.data_queue.get()

            if audio_data == 'PLATYPUS':
                break

            segments, info = self.model.transcribe(audio_data, beam_size=5, language=self.language, vad_filter=True)
            for segment in segments:
                text = segment.text.strip()
                logging.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, text))
                with self.lock:
                    self.transcription.append(text)
                    self.last_transcription = text

            self.data_queue.task_done()
            time.sleep(0.25)

    def recorder_callback(self, _, audio_data):
        """Callback function for the recorder."""
        self.audio_timer.cancel_timer()  # Cancel the timer if audio is detected to avoid prompting overlap and broken convos.
        print("Audio detected! Cancelling timer!")
        audio = io.BytesIO(audio_data.get_wav_data())
        self.data_queue.put(audio)

    def listen(self):
        """Start listening to the microphone."""
        with sr.Microphone(device_index=self.default_mic) as source:
            self.recorder.adjust_for_ambient_noise(source)

        self.recorder.listen_in_background(source=source, callback=self.recorder_callback)

    def stop(self):
        """Stop the transcription process."""
        logging.info("Stopping...")
        logging.info(f"Transcription:\n {self.transcription}")
        self.is_listening = False
        self.data_queue.put("PLATYPUS")

    def get_last_transcription(self):
        """Get the last transcription and clear it."""
        with self.lock:
            text = self.last_transcription
            self.last_transcription = ""
        return text
    
    @staticmethod
    def setup_mic():
        """Set up the microphone."""
        p = pyaudio.PyAudio()
        default_device_index = None
        try:
            default_input = p.get_default_input_device_info()
            default_device_index = default_input["index"]
        except (IOError, OSError):
            logging.error("Default input device not found. Printing all input devices:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    logging.info(f"Device index: {i}, Device name: {info['name']}")
                    if default_device_index is None:
                        default_device_index = i

        if default_device_index is None:
            raise Exception("No input devices found.")

        return default_device_index

    @staticmethod
    def configure_logging(level: str = "INFO"):
        """
        Configure the logging level for the whole application.
        :param level: The desired logging level. Should be one of the following:
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        """
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logging.basicConfig(level=levels.get(level.upper(), logging.INFO))

"""
A class for handling TTS functions and work queue.
"""

class TTS:
    def __init__(self, tts_queue, azure_tts, history, chat):
        self.tts_queue = tts_queue
        self.azure_tts = azure_tts
        self.history = history
        self.chat = chat
        self.audio_timer = AudioTimer(history=self.history, chat=self.chat, tts=self)
        
    def start_audio_timer(self):
        self.audio_timer.start_timer()  # Start the audio timer

    @staticmethod
    def get_random_interval():
        return random.randint(3, 12)

    def tts_worker(self):
        """Worker function to process TTS replies from the queue."""
        processed_groups = []  # List to store processed groups

        while True:
            try:
                tts_reply = self.tts_queue.get(timeout=1)
                if tts_reply is None:
                    break
                
                for group in tts_reply:
                    if group in processed_groups:
                        continue
                    
                    self.azure_tts.azure_tts(group)
                    processed_groups.append(group)  
                
                self.tts_queue.task_done()  # Mark the processed item as done
                
                # Start the audio timer only if the queue is empty
                if self.tts_queue.empty():
                    print("Ready!!")
                    self.start_audio_timer()                    
                    print("Audio timer started.")

            except queue.Empty:
                continue  # If the queue is empty, continue checking for new items

    def stop_tts_worker(self):
        self.tts_queue.put(None)

    # Add tts_reply to work queue
    def add_to_tts_queue(self, tts_reply):
        for group in tts_reply:
            self.tts_queue.put(tts_reply)  # Add the TTS reply to the queue