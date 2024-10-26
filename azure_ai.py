"""
Description: A class that handles the Azure TTS API call function and parameters.
"""

import azure.cognitiveservices.speech as speechsdk
import os
import threading

class Azure_AI():
        def __init__(self, audio_timer):
                # Setup of Azure TTS voice settings, can preview these online.
                self.voice_name = "en-US-AvaMultilingualNeural"
                self.pitch_percentage = "+14%"
                self.rate_percentage = "+20%"

                # This class requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
                self.speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
                self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

                # The neural multilingual voice can speak different languages based on the input text.
                self.speech_config.speech_synthesis_voice_name=self.voice_name
                self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=self.audio_config)

                # Setup of Azure STT settings
                self.speech_config.speech_recognition_language="en-US"
                self.speech_config.set_profanity(speechsdk.ProfanityOption.Removed)
                self.audio_input_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
                self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "600000")
                self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_input_config)

                self.is_listening = False
                self.listen_thread = None

                self.audio_timer = audio_timer
        
        def azure_tts(self, text):
                # Create the SSML with pitch adjustment
                ssml = f"""
                <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
                <voice name='{self.voice_name}'>
                        <prosody pitch='{self.pitch_percentage}'>
                        {text}
                        </prosody>
                </voice>
                </speak>
                """

                speech_synthesis_result = self.speech_synthesizer.speak_ssml_async(ssml).get()

                if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        print("Speech synthesized for text [{}]".format(text))
                elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
                        cancellation_details = speech_synthesis_result.cancellation_details
                        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
                        if cancellation_details.reason == speechsdk.CancellationReason.Error:
                                if cancellation_details.error_details:
                                        print("Error details: {}".format(cancellation_details.error_details))
                                        print("Did you set the speech resource key and region values?")

        def start_continuous_listening(self, callback):
                def recognize_cb(evt):
                    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                        callback(evt.result.text)

                self.speech_recognizer.recognized.connect(recognize_cb)
                self.speech_recognizer.session_started.connect(lambda evt: print('Azure STT: Session started'))
                self.speech_recognizer.session_stopped.connect(lambda evt: print('Azure STT: Session stopped'))

                self.is_listening = True
                self.listen_thread = threading.Thread(target=self._continuous_listen)
                self.listen_thread.start()

        def _continuous_listen(self):
                self.speech_recognizer.start_continuous_recognition()
                while self.is_listening:
                    threading.Event().wait(1)  # Wait for 1 second
                self.speech_recognizer.stop_continuous_recognition()

        def stop_continuous_listening(self):
                self.is_listening = False
                if self.listen_thread:
                    self.listen_thread.join()