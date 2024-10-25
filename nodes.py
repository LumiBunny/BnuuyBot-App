import asyncio
from messages import TextFormatting, Prompting, PostChat, ChatLog
from chat_completions import Completions
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

class NodeRegistry: 
    def __init__(self, stt, tts, models, chat_history, message_queue): 
        self.nodes = {} 
        self.stt = stt
        self.tts = tts
        self.models = models
        self.prompt = Prompting(chat_history)
        self.chat_log = ChatLog()
        self.post = PostChat(message_queue)
        self.memory = Memory("memories", self.models)
        self.chat = Completions(chat_history, models, self.chat_log, self.post)
        self.text = TextFormatting(chat_history, models)

        # Initialize nodes
        self.setup_nodes()

    def setup_nodes(self):
        # Create nodes with their handlers
        self.nodes["start"] = Node("start", self.starting_node_handler)

    async def starting_node_handler(self, transcription):
            print("You: ", transcription)
            print("intent", self.models.get_intent(transcription))
            if self.prompt.get_attention(user_id, transcription):
                reply = await self.chat.bnuuybot_completion()
                if reply is not None:
                    self.tts.add_to_tts_queue(reply)
                else:
                    self.stt.audio_timer.start_timer()
            else:
                # Create tasks for both operations
                task_context = asyncio.create_task(self.text.get_short_context(4))
                task_memory = asyncio.create_task(self.memory.retrieve_relevant_memory(transcription))
                emotion = self.models.get_emotion(transcription)
                self.prompt.get_emotion(emotion, user_id, transcription)
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
                    self.stt.audio_timer.cancel_timer()
                    self.post.add_to_queue(msg_type="assistant", user_id="Assistant", content="💭 Oh yeah! I think Lumi mentioned this before!")
                    context_with_memories = f"You remembered that {user_id} once spoke about this: {retrieved_memory}. Chat context: {context}. User message: {transcription}. This is a message from {user_id}. Respond to the message however you like."
                    self.chat_history.add("user", "System", context_with_memories)
                    print(f"Context with memories: {context_with_memories}")
                    retrieved_memory = None

                    reply = await self.chat.bnuuybot_completion()
                    if reply is not None:
                        self.tts.add_to_tts_queue(reply)
                    else:
                        self.stt.audio_timer.start_timer()
                else:
                    print("No relevant memories found.")