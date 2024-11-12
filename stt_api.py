import asyncio
import pyaudio
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

# Replace URL with PyAudio setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def main():
    try:
        # use default config
        deepgram: DeepgramClient = DeepgramClient(api_key="36ec06c5ff54ead1d67747c0381837267eba95fe")

        # Create a websocket connection to Deepgram
        dg_connection = deepgram.listen.websocket.v("1")

        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            print(f"You: {sentence}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # connect to websocket
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            encoding="linear16",
            sample_rate=RATE,
            profanity_filter=False,
            punctuate=True,
            smart_format=True,
            diarize=True,
            filler_words=True,
            # Time in milliseconds of silence to wait for before finalizing speech
            endpointing=300
            )

        print("\n\nPress Ctrl+C to stop recording...\n\n")
        
        if not dg_connection.start(options):
            print("Failed to start connection")
            return

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Listening...")

        try:
            while True:
                data = stream.read(CHUNK)
                dg_connection.send(data)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            dg_connection.finish()

        print("Finished")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

if __name__ == "__main__":
    main()