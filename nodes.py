import asyncio
from messages import TextFormatting, Prompting, PostChat, ChatLog, SentimentAnalyzer
from chat_completions import Completions
from preferences import PreferenceClassifier, PreferenceProcessor
from node_manager import NodeManager
from memory import Memory

previous_transcription = ""
user_id = "Lumi"

class Node: 
    def __init__(self, name, handler=None): 
        self.name = name
        self.handler = handler  # Single handler function instead of a list

    async def process(self, *args):
        if self.handler:
            return await self.handler(*args)

# nodes.py
class NodeRegistry:
    def __init__(self, stt, tts, models, chat_history, message_queue, user_id):
        self.nodes = {}
        self.stt = stt
        self.tts = tts
        self.models = models
        self.chat_history = chat_history
        self.user_id = user_id
        self.remember = ""
        self.analyzer = SentimentAnalyzer()
        self.prompt = Prompting(chat_history)
        self.chat_log = ChatLog()
        self.post = PostChat(message_queue)
        self.memory = Memory("memories", self.models)
        self.chat = Completions(chat_history, models, self.chat_log, self.post)
        self.text = TextFormatting(chat_history, models)
        self.node_manager = NodeManager(self)  # Add Node Modules
        self.preference_classifier = PreferenceClassifier(models)
        self.processor = PreferenceProcessor(models, chat_history)
        
        # Initialize nodes
        self.setup_nodes()
        self.node_manager.set_initial_node("start")  # Set initial node

        self.memory.print_all_memories()

    def setup_nodes(self):
        # Create nodes with their handlers
        self.nodes["start"] = Node("start", self.starting_node_handler)
        self.nodes["remember this"] = Node("remember this", self.remember_this)

    # ALL of my node functions go here. I can probably recycle some of these.

    async def starting_node_handler(self, transcription):
            print("You: ", transcription)
            self.chat_history.add("user", self.user_id, transcription)
            intent = self.models.get_intent(transcription)
            print("intent", intent)
            if self.prompt.get_attention(self.user_id, transcription):
                await self.get_attention()
            elif intent == "remember that":
                await self.verify_remember_this(transcription)
            else:
                await self.get_reply_plus_memory(transcription)

    async def get_attention(self):
        reply = await self.chat.bnuuybot_completion()
        if reply is not None:
            self.tts.add_to_tts_queue(reply)
        else:
            self.stt.audio_timer.start_timer()
        # Finishes here, we will loop back to starter/main node

    async def get_reply_plus_memory(self, transcription):
        # Create tasks for both operations
                task_context = asyncio.create_task(self.text.get_short_context(4))
                task_memory = asyncio.create_task(self.memory.retrieve_relevant_memory(transcription))
                emotion = self.models.get_emotion(transcription)
                self.prompt.get_emotion(emotion, self.user_id, transcription)
                self.post.add_to_queue(msg_type="user", content=transcription)
                # Generate chat completion
                reply = await self.chat.bnuuybot_completion()
                if reply is not None:
                    self.tts.add_to_tts_queue(reply)
                else:
                    self.stt.audio_timer.start_timer()
                retrieved_memory = await task_memory
                context = await task_context
                if retrieved_memory:
                    await self.talk_about_memory(transcription,retrieved_memory,context)
                else:
                    print("No relevant memories found.")
                    
    async def talk_about_memory(self, transcription, retrieved_memory, context):
        self.stt.audio_timer.cancel_timer()
        self.post.add_to_queue(msg_type="assistant", user_id="Assistant", content="💭 Oh yeah! I think Lumi mentioned this before!")
        context_with_memories = f"You remembered that {self.user_id} once spoke about this: {retrieved_memory}. Chat context: {context}. User message: {transcription}. This is a message from {self.user_id}. Respond to the message however you like."
        self.chat_history.add("user", "System", context_with_memories)
        print(f"Context with memories: {context_with_memories}")
        retrieved_memory = None

        reply = await self.chat.bnuuybot_completion()
        if reply is not None:
            self.tts.add_to_tts_queue(reply)
        else:
            self.stt.audio_timer.start_timer()

    async def verify_remember_this(self, transcription):
        recent = self.chat_history.get_recent_messages(3)
        self.remember = self.text.history_list_to_string(recent)
        print(self.remember)
        print(f"Sentiment found in context: {await self.analyze_preference(self.remember)}") # Works
        self.post.add_to_queue(msg_type="user", content=transcription)
        print(f"To remember: {self.remember}") # Pulls correct sentence with sentiment.
        self.chat_history.add("user", "System", f"Ask {self.user_id} if they want you to remember this: {self.remember}. You MUST tell them what it is they asked you to remember. You will also ask the user to if they want it remembered or not. You should address the user casually when you ask.")
        reply = await self.chat.bnuuybot_completion()
        if reply is not None:
            self.tts.add_to_tts_queue(reply)
        else:
            self.stt.audio_timer.start_timer()
        # Transition to remember this node
        self.node_manager.transition_to_node("remember this")
        
    async def remember_this(self, transcription):
        print("You: ", transcription)
        self.chat_history.add("user", self.user_id, transcription)
        sentiment = self.models.get_sentiment(transcription)
        self.post.add_to_queue(msg_type="user", content=transcription)
        print(sentiment)
        if sentiment == 'Yes':
            context = f"{self.user_id} wants me to remember this. Do I want to remember this? I will tell {self.user_id} what I have decided to do, I can make my own choices after all."
            # DEBUGGING
            print(self.user_id)
            # I am going to rework the logic before adding things because it makes for very ugly "memories"
            self.to_remember = await self.processor.process_text(self.remember, self.user_id)
            print("Stored to memory.")
            print(self.to_remember)
        elif sentiment == 'No':
            context = f"{self.user_id} doesn't want me to remember this. Do I want to remember this? I will tell {self.user_id} what I have decided to do, I can make my own choices after all."
        else:
            context = f"{self.user_id} can't seem to decide if it's worth remembering or not. Do I want to remember this anyways or not? I will tell {self.user_id} what I have decided to do, I can make my own choices after all."
    
        # Add context to chat history and get response
        self.chat_history.add("user", "Assistant", context)
        reply = await self.chat.bnuuybot_completion()
        
        if reply is not None:
            self.tts.add_to_tts_queue(reply)
        else:
            self.stt.audio_timer.start_timer()
    
        # Return to start node
        self.node_manager.transition_to_node("start")

    async def analyze_preference(self, transcription):
        sentences = self.text.split_into_sentences(transcription)
        text = ' '.join(sentences)
        result = self.analyzer.get_sentiment(text)
        return result['word']