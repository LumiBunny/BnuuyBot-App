# BunnyBot 🐰💕
Speech to speech app developed in Python for conversations with an LLM.

## Introduction ✨
BunnyBot was a way for me to get back into learning coding again by learning something new using Python. The app became a way for me to practice all kinds of things like using modules, making classes, OOP, the general piecing together of LLMs and their functions, plus much more.

I actually have no idea what I am doing most of the time, but that is part of the fun of a learning project.

### Version 0.4.3 Notes 📝 (October 19, 2024)
A lot of new changes and additions to enable smoother running and less latency in inferencing!
+ Added async to functions. Allows for concurrent processes to run and reduces latency/wait times between user prompts and AI replies.
+ Memory retrieval and context retrieval are now processed asyncronously.
+ Added Prompting class to messages.py module for creating functions to allow for more dynamic prompt engineering.
+ Added emotion detection in a get_emotion() function via text-classification LLM. Allows for more dynamic prompting/responses.
+ Added get_attention() function to use 'Hey Bunny' or 'Bunny' to get the AI's attention in a conversation.
+ Moved Flask POST requests to a queue and thread with its own class and functions, reduces time spent processing code by moving it to a background thread like this.
+ Bug fixes to self prompting functions.
+ Bug fixes to audio timer functions.
+ Bug fixes to memory retrieval functions.

### Version 0.1.0 Notes 📝 (October 17, 2024)
+ Speech to text
+ Audio transcription using [Azure AI STT](https://azure.microsoft.com/en-ca/products/ai-services/ai-speech) (previously was using Whisper)
+ Locally hosted LLM using LM Studio (can also be replaced with OpenAI API)
+ Text to speech using [Azure AI TTS](https://azure.microsoft.com/en-ca/products/ai-services/ai-speech)
+ Conversational memory storage using [Qdrant](https://qdrant.tech) vector database (base set up, in development)
+ Relevant memory retrieval using vector similarity search in Qdrant.
+ Other small LLMs to perform various tasks like embeddings and summarization.
+ Audio Timer for self prompting the LLM when no audio input is detected (allows for conversation to not run stale)
