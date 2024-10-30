import asyncio
from messages import TextFormatting, Prompting, PostChat, ChatLog, SentimentAnalyzer
from chat_completions import Completions
from preferences import PreferenceProcessor
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
        self.preference_processor = PreferenceProcessor(models)
        
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
        # Process the recent messages using the PreferenceProcessor
        self.remember = await self.preference_processor.process_text(self.chat_history, self.user_id)
        self.chat_history.add("user", "user", f"Ask {self.user_id} if they want you to remember this: {self.remember}. You MUST tell them what it is they asked you to remember. You will also ask the user to if they want it remembered or not. You should address the user casually when you ask.")
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
        sentiment = self.models.get_decision(transcription)
        self.post.add_to_queue(msg_type="user", content=transcription)
        print(sentiment)
        if sentiment == 'Yes':
            context = f"{self.user_id} wants you to remember this: {self.remember}. Do you want to remember this? You will tell {self.user_id} what you have decided to do, you can make my own choices after all."
            # DEBUGGING
            print(self.user_id)
            # I am going to rework the logic before adding things because it makes for very ugly "memories"
            print("Stored to memory.")
            print(self.remember)
        elif sentiment == 'No':
            context = f"{self.user_id} doesn't want you to remember this: {self.remember}. Do you want to remember this? You will tell {self.user_id} what you have decided to do, you can make my own choices after all."
        else:
            context = f"{self.user_id} can't seem to decide if it's worth remembering this or not: {self.remember}. Do you want to remember this anyways or not? You will tell {self.user_id} what you have decided to do, you can make my own choices after all."
    
        # Add context to chat history and get response
        self.chat_history.add("user", "user", context)
        reply = await self.chat.bnuuybot_completion()
        
        if reply is not None:
            self.tts.add_to_tts_queue(reply)
        else:
            self.stt.audio_timer.start_timer()
    
        # Return to start node
        self.node_manager.transition_to_node("start")