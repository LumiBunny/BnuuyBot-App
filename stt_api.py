import asyncio
import pyaudio
import threading
import os
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

class Deepgram_STT:
    def __init__(self):
        self.deepgram = DeepgramClient(os.environ.get('DEEPGRAM_KEY'))
        self.is_listening = False
        self.listen_thread = None
        
        # PyAudio setup
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

    def start_continuous_listening(self, callback):
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._continuous_listen, args=(callback,))
        self.listen_thread.start()

    def _continuous_listen(self, callback):
        asyncio.run(self._async_continuous_listen(callback))

    async def _async_continuous_listen(self, callback):
        dg_connection = self.deepgram.listen.websocket.v("1")

        async def on_message(result):
            if result.is_final:
                sentence = result.channel.alternatives[0].transcript
                if sentence:
                    callback(sentence)

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            encoding="linear16",
            sample_rate=self.RATE,
            punctuate=True,
            endpointing=300
        )

        await dg_connection.start(options)

        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        while self.is_listening:
            data = stream.read(self.CHUNK)
            await dg_connection.send(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        await dg_connection.finish()

    def stop_continuous_listening(self):
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join()