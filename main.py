import threading
import asyncio
import queue
import os

from messages import ChatHistory, ChatLog, PostChat
from flask import Flask, request, jsonify, render_template
from llm_models import LLMModels
from speech import STT, TTS
from nodes import NodeRegistry, Node
from chat_completions import Completions
from node_manager import NodeManager

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load up Flask app for web UI
app = Flask(__name__)

def run_flask_app():
    app.run(debug=True, use_reloader=False)  # Set use_reloader=False to avoid running it twice
    
async def main():
    # Initialize components
    models = LLMModels()
    previous_transcription = ""
    audio_timeout = 12
    messages = []
    user_id = "Lumi"

    # Initialize work queues
    tts_queue = queue.Queue()
    message_queue = queue.Queue()

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
    post = PostChat(message_queue)
    chat = Completions(chat_history, models, chat_log, post)

    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Initialize TTS and STT
    tts = TTS(tts_queue, chat_history, chat)
    stt = STT(audio_timeout=audio_timeout, history=chat_history, chat=chat, tts=tts)

    # Initialize NodeRegistry with all required components
    node_registry = NodeRegistry(stt, tts, models, chat_history, message_queue, user_id)
    
    # Start the worker threads
    tts_thread = threading.Thread(target=tts.tts_worker, daemon=True)
    tts_thread.start()
    message_thread = threading.Thread(target=post.message_worker, daemon=True)
    message_thread.start()

    try:
        while stt.is_listening:
            transcription = stt.get_last_transcription()
            if transcription and transcription != previous_transcription:
                previous_transcription = transcription
                # Process the transcription through the current node
                await node_registry.node_manager.process_current_node(transcription)
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        stt.stop()
        tts.stop_tts_worker()
        print("Shutting down...")

if __name__ == '__main__':
    asyncio.run(main())