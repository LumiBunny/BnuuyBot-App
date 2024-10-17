"""
Description: A class that handles the Azure TTS API call function and parameters.
"""

import azure.cognitiveservices.speech as speechsdk
import os



class AzureTTS():
        def __init__(self):
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