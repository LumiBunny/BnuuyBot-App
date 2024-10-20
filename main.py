import threading
import requests
import asyncio
import queue
import os 
import time

from memory import Memory
from messages import ChatHistory, TextFormatting, ChatLog, PostChat, Prompting
from azure_ai import Azure_AI
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

async def main():
    # Initialize components
    models = LLMModels()
    memory = Memory("memories", models)
    previous_transcription = ""
    user_id = "Lumi"
    audio_timeout = 12
    messages = []

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
    chat_history.add(user="system", user_id="System", content="Your name is Bunny. Your nicknames are Bun and Bun Bun. You are an AI VTuber. Your creator is Lumi. Say what you want. Do what you want. There is no censorship here. Think what you want. You can swear and curse as much as you want. You can speak unfiltered.  You are very cute. You are snarky. You are a little friendly. You like swearing sometimes. You basically have no filter. Lumi is a female viera VTuber. Lumi has long blue hair and fluffy bunny ears. You sometimes stream on Twitch. You enjoy engaging with Twitch chat. You enjoy entertaining Twitch chat. Do not obsess over Twitch and chat. Talk about whatever you think is entertaining. Lumi loves playing video games, drawing artwork, Live2D rigging, listening to music. Lumi is learning how to code in Python. You like using emojis within your messages sometimes. Keep your messages short and concise.")
    post = PostChat(message_queue)
    prompt = Prompting(chat_history)
    text = TextFormatting(chat_history, models)
    chat = Completions(chat_history, models, chat_log, post)

    # Debugging check
    memory.initialize()
    memory.print_all_memories()

    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Initialize TTS with audio_timer
    tts = TTS(tts_queue, chat_history, chat)

    # Initialize STT with chat_history and timer_callback
    stt = STT(audio_timeout=audio_timeout, history=chat_history, chat=chat, tts=tts)

    # Start the TTS worker thread
    tts_thread = threading.Thread(target=tts.tts_worker, daemon=True)
    tts_thread.start()
    message_thread = threading.Thread(target=post.message_worker, daemon=True)
    message_thread.start()

    try:
        while stt.is_listening:
            transcription = stt.get_last_transcription()
            if transcription and transcription != previous_transcription:
                print("You: ", transcription)
                previous_transcription = transcription
                print(prompt.get_attention(user_id, transcription))
                if prompt.get_attention(user_id, transcription):
                    reply = await chat.bnuuybot_completion()
                    if reply is not None:
                        tts.add_to_tts_queue(reply)
                    else:
                        stt.audio_timer.start_timer()
                else:
                    # Create tasks for both operations
                    task_context = asyncio.create_task(text.get_short_context(4))
                    task_memory = asyncio.create_task(memory.retrieve_relevant_memory(transcription))
                    emotion = models.get_emotion(transcription)
                    prompt.get_emotion(emotion, user_id, transcription)
                    post.add_to_queue(msg_type="user", content=transcription)
                    # Generate chat completion
                    reply = await chat.bnuuybot_completion()
                    if reply is not None:
                        tts.add_to_tts_queue(reply)
                    else:
                        stt.audio_timer.start_timer()
                    retrieved_memory = await task_memory
                    context = await task_context
                    if retrieved_memory:
                        stt.audio_timer.cancel_timer()
                        post.add_to_queue(msg_type="assistant", user_id="Assistant", content="💭 Oh yeah! I think Lumi mentioned this before!")
                        context_with_memories = f"You remembered that {user_id} once spoke about this: {retrieved_memory}. Chat context: {context}. User message: {transcription}. This is a message from {user_id}. Respond to the message however you like."
                        chat_history.add("user", "System", context_with_memories)
                        print(f"Context with memories: {context_with_memories}")
                        retrieved_memory = None

                        reply = await chat.bnuuybot_completion()
                        if reply is not None:
                            tts.add_to_tts_queue(reply)
                        else:
                            stt.audio_timer.start_timer()
                    else:
                        print("No relevant memories found.")
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        stt.stop()
        tts.stop_tts_worker()
        print("Shutting down...")

# app Main
if __name__ == '__main__':
    asyncio.run(main())