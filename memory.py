"""
Description: A class that handles a long term conversational and contextual memories 
via a vector store. The vector store is managed with Qdrant hosted on a local server 
using Docker.
"""

from datetime import datetime
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Memory System Class
class Memory:
    candidate_labels = [
            "likes", "dislikes", "preference", "personal", "question", "hobby", "dates", "goals"
        ]
    
    user_id = "Lumi"
    
    def __init__(self, collection_name="memories", llm_models=None):
        self.models = llm_models
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.embedder = llm_models.get_embedder()
        self.classifier = llm_models.get_classifier()

        exists = self.client.collection_exists(collection_name=collection_name)
        
        if exists:
            print(f"Collection '{collection_name}' exists: {exists}")
        else:
            print(f"Creating new collection: {collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
            self.add_memory(self.user_id,"Lumi has been learning how to use Python since the summer, she's really passionate about it.", "goals")
            self.add_memory(self.user_id,"Lumi's favourite colour is blue.", "likes")
            self.add_memory(self.user_id,"Lumi really dislikes green peas.", "dislikes")

    def add_memory(self, user_id, text, category):
        if category is None:
            category = self.classify_categories(text)
        
        embedding = self.embedder.encode(text)
        memory_id = str(uuid.uuid4())
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=memory_id,
                    vector=embedding.tolist(),
                    payload={
                        "content": text,  
                        "user_id": user_id,
                        "category": category,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )

    # Function to retrieve relevant memories, if any.
    # You can use it like this:
    """query = "What are some good programming languages to learn?"
    relevant_memories = self.retrieve_relevant_memory(query)

    if relevant_memories:
        print("These memories might be relevant to your query.")
    else:
        print("No relevant memories found. You might want to add some new information on this topic.")"""

    # user_id will be implemented later when I explore conversations with multiple/other users.
    def retrieve_relevant_memory(self, query, user_id):
        embedding = self.embedder.encode(query)
        similar_memories = self.search_similar_memories(embedding)
        matching_category = self.search_matching_categories(query)

        combined = similar_memories + matching_category

        query_category = self.classify_categories(query)
        retrieved_memory = self.pick_relevant_memory(combined, query_category)


        if retrieved_memory:
            print("Relevant memory found:")
            for i, memory in enumerate(retrieved_memory, 1):
                print(f"{i}. Memory object: {memory}")
                print(f"   ID: {memory.get('id', 'N/A')}")
                print(f"   Score: {memory.get('score', 'N/A')}")
                print(f"   Category: {memory['payload'].get('category', 'N/A')}")
                print(f"   Text: {memory['payload'].get('content', 'N/A')[:100]}...")
                print(f"   All keys in memory object: {list(memory.keys())}")
                print(f"   All keys in payload: {list(memory['payload'].keys())}")
        else:
            print("No relevant memories found.")

        return retrieved_memory

    def classify_categories(self, text):
        result = self.classifier(text, self.candidate_labels)
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        if top_score >= 0.7:
            return top_label
        else:
            return "general chat"
        
    def search_similar_memories(self, embedding, limit=10, score_threshold=0.5):
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        if not search_results:
            print("No matching semantic meanings found.")
            return []
        
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        similar_memories = [
            {"id": result.id, "payload": result.payload, "score": result.score}
            for result in sorted_results
        ]
        
        return similar_memories
    
    def search_matching_categories(self, text, limit=10):
        category = self.classify_categories(text)
        if category == "general chat":
            print("Input classified as general chat. No specific category to search for.")
            return []
        
        search_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category)
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        matching_memories = [
            {
                "id": point.id,
                "payload": point.payload,
                "score": 1.0  # Since this is a category match, we assign a default score of 1.0
            }
            for point in search_results[0]
        ]
        
        # Check if we found any matching memories
        if not matching_memories:
            print("No matching categories found in memory.")
            return []
        
        print(f"Found {len(matching_memories)} memories with matching category: {category}")
        return matching_memories
    
    def pick_relevant_memory(self, combined_results, query_category):
        matching_category_high_score = []
        high_score_only = []

        for item in combined_results:
            semantic_score = item.get('score', 0)
            item_category = item.get('category', '')

            if semantic_score >= 0.5 and item_category == query_category:
                matching_category_high_score.append(item)
            elif semantic_score >= 0.5:
                high_score_only.append(item)

        # Debugging output
        print(f"Items with matching category and high score: {len(matching_category_high_score)}")
        print(f"Items with only high score: {len(high_score_only)}")

        matching_category_high_score.sort(key=lambda x: x.get('score', 0), reverse=True)
        high_score_only.sort(key=lambda x: x.get('score', 0), reverse=True)
        organized_list = matching_category_high_score + high_score_only
        retrieved_memory = organized_list[:1]

        return retrieved_memory
    
    def delete_memory(self, collection_name, retrieved_memory):
        
        vector = retrieved_memory[0].get('vector', None)

        self.client.delete_vectors(
            collection_name=collection_name,
            vectors=vector
        )
    
    def print_all_memories(self):
        all_memories = self.client.scroll(
            collection_name="memories",
            limit=100  # Adjust as needed
        )
        print(f"Total memories retrieved: {len(all_memories[0])}")
        for i, memory in enumerate(all_memories[0]):
            print(f"Memory {i+1}:")
            print(f"  ID: {memory.id}")
            print(f"  Payload: {memory.payload}")
            if 'vector' in memory:
                print(f"  Vector: {memory.vector[:5]}... (truncated)")  # Print first 5 elements of vector
            print("---")

'''
# Multi-user support (Future planned update, vague idea, to be worked on)
user_sessions = {}

# When a user starts a conversation or joins a group chat
def start_user_session(user_id):
if user_id not in user_sessions:
user_sessions[user_id] = {"history": []}
# Add some initial memories
memory.add_memory(user_id, "This is a new user.", category="system")
memory.add_memory(user_id, "The user's preferences are not known yet.", category="system")

# When processing a message
def process_message(user_id, message):
if user_id not in user_sessions:
start_user_session(user_id)

user_sessions[user_id]["history"].append(message)

# Process the message using your existing pipeline
# Use memory_system.retrieve_relevant_memories(message, user_id) to get context
# ...

# For group chats
def process_group_message(group_id, user_id, message):
    if group_id not in user_sessions:
        user_sessions[group_id] = {"users": set(), "history": []}
    
    user_sessions[group_id]["users"].add(user_id)
    user_sessions[group_id]["history"].append((user_id, message))
    
    # Process the group message
    # You might want to retrieve memories for all users in the group
    # ...
'''