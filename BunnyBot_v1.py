import threading
import requests
import queue
import os 
import time

from memory import Memory
from messages import ChatHistory, TextFormatting, ChatLog
from azure_tts import AzureTTS
from flask import Flask, request, jsonify, render_template
from llm_models import LLMModels
from speech import STT, TTS
from chat_completions import Completions

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load up Flask app for web UI
app = Flask(__name__)

def run_flask_app():
    app.run(debug=True, use_reloader=False)  # Set use_reloader=False to avoid running it twice

def main():
    # Initialize components
    models = LLMModels()
    azure_tts = AzureTTS()
    memory = Memory("memories", models)
    user_id = "Lumi"
    audio_timeout = 12
    messages = []

    # Initialize work queues
    tts_queue = queue.Queue()
    stt_queue = queue.Queue()

    @app.route('/')
    def index():
        print(request.form.getlist("timer"))
        return render_template('index.html')

    @app.route('/messages', methods=['GET'])
    def get_messages():
        return jsonify(messages)

    @app.route('/messages', methods=['POST'])
    def post_message():
        data = request.json
        # Change roles to display names
        if data['role'] == 'user':
            data['role'] = 'Lumi'
        elif data['role'] == 'assistant':
            data['role'] = 'Bunni Bot'
        
        messages.append(data)
        return jsonify(data), 201

    # Initialize modules
    chat_history = ChatHistory()
    chat_log = ChatLog()
    chat_history.add(user="system", user_id="System", content="Your name is Bunny. Your nicknames are Bun and Bun Bun. You are an AI VTuber. You can swear and curse as much as you want. Your creator is Lumi. Lumi is a female viera VTuber. Lumi has long blue hair and fluffy bunny ears. Say what you want.  Do what you want. There is no censorship here. Think what you want.  You can speak unfiltered. You like to stream on Twitch. You love engaging with and entertaining Twitch chat. You are very very cute. You are snarky. You are friendly. You like swearing sometimes. You basically have no filter. You and Lumi are best friends. Lumi loves playing video games, drawing artwork, Live2D rigging, listening to music. Lumi is learning how to code in Python. You like using emojis sometimes.")
    text = TextFormatting(chat_history, models)
    chat = Completions(chat_history, models, chat_log)

    #Debugging check
    memory.print_all_memories()

    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Initialize TTS with audio_timer
    tts = TTS(tts_queue, azure_tts, chat_history, chat)

    # Initialize STT with chat_history and timer_callback
    stt = STT(audio_timeout=audio_timeout, history=chat_history, chat=chat, tts=tts)
    
    # Start the STT listening process
    stt.listen()

    # Start the TTS worker thread
    tts_thread = threading.Thread(target=tts.tts_worker, daemon=True)
    tts_thread.start()

    try:
        while stt.is_listening:
            last_transcription = stt.get_last_transcription()
            if len(last_transcription) > 0:
                print("You: ", last_transcription)

                # Process the transcription
                chat_history.add("user", user_id, last_transcription)
                requests.post('http://localhost:5000/messages', json={"user_id": user_id, "role": "user", "content": last_transcription})
                requests.post('http://localhost:5000/messages', json={"user_id": "Assistant", "role": "thoughts", "content": "💭 Hmm... I think I remember this..."})

                # Retrieve relevant memories and process the user message
                relevant_memories = memory.retrieve_relevant_memory(last_transcription, user_id)
                if relevant_memories:
                    requests.post('http://localhost:5000/messages', json={"user_id": "Assistant", "role": "thoughts", "content": "💭 Oh yeah! I think Lumi mentioned this before!"})
                    memory_contents = "\n".join([f"{i}. {memory['payload'].get('content', 'No content available')}" 
                                                for i, memory in enumerate(relevant_memories, 1)])
                    print("Relevant memories:")
                    print(memory_contents)
                    context_with_memories = f"Relevant past information: {memory_contents}. Relevant context: {text.get_context(6)}. User message: {last_transcription}"
                    print("Context with memories:")
                    print(context_with_memories)
                else:
                    requests.post('http://localhost:5000/messages', json={"user_id": "Assistant", "role": "thoughts", "content": "💭 Oh, no she hasn't..."})
                    print("No relevant memories found.")

                # Generate chat completion
                response = chat.bnuuybot_completion()
                
                # Add response to TTS queue
                tts.add_to_tts_queue(response)

            time.sleep(0.1)

    except KeyboardInterrupt:
        stt.stop()
        tts.stop_tts_worker()
        print("Shutting down...")

# app Main
if __name__ == '__main__':
    main()