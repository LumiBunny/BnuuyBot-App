# BunnyBot 🐰💕
Speech to speech app developed in Python for conversations with an LLM.

## Introduction ✨
BunnyBot was a way for me to get back into learning coding again by learning something new using Python. The app became a way for me to practice all kinds of things like using modules, making classes, OOP, the general piecing together of LLMs and their functions, plus much more.

I actually have no idea what I am doing most of the time, but that is part of the fun.

### Version 0.1.0 Notes 📝
+ Speech to text
+ Audio transcription using [Whisper/FasterWhisper](https://medium.com/@sumudithalanz/the-art-of-crafting-an-effective-readme-for-your-github-project-cf425a8b1580)
+ Locally hosted LLM using LM Studio (can also be replaced with OpenAI API)
+ Text to speech using [Azure AI TTS](https://azure.microsoft.com/en-ca/products/ai-services/ai-speech)
+ Conversational memory storage using [Qdrant](https://qdrant.tech) vector database (base set up, in development)
+ Relevant memory retrieval using vector similarity search in Qdrant.
+ Other small LLMs to perform various tasks like embeddings and summarization.
+ Audio Timer for self prompting the LLM when no audio input is detected (allows for conversation to not run stale)
