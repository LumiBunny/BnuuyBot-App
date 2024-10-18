"""
Description: A class that handles a long term conversational and contextual memories 
via a vector store. The vector store is managed with Qdrant hosted on a local server 
using Docker.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional
import uuid
from datetime import datetime

class Memory:
    def __init__(self, collection_name="memories", llm_models=None):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.embedder = llm_models.get_embedder()

    def initialize(self):
        self.create_optimized_collection()

    def create_optimized_collection(self):
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Adjust this to match your embedding size
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True
                        )
                    )
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                    max_indexing_threads=0
                )
            )

    def retrieve_relevant_memory(self, query: str) -> Optional[models.ScoredPoint]:
        embedding = self.embedder.encode(query)
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=1,
            score_threshold=0.5,
            search_params=models.SearchParams(hnsw_ef=128, exact=False)
        )
        
        return search_results[0] if search_results else None

    def add_memory(self, user_id, text):
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
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )
    
    def delete_memory(self, retrieved_memory):
        if retrieved_memory and hasattr(retrieved_memory, 'id'):
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[retrieved_memory.id])
            )
    
    def print_all_memories(self):
        all_memories = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,  # Adjust as needed
            with_vectors=True  # Ensure we're requesting vectors
        )
        print(f"Total memories retrieved: {len(all_memories[0])}")
        for i, memory in enumerate(all_memories[0]):
            print(f"Memory {i+1}:")
            print(f"  ID: {memory.id}")
            print(f"  Payload: {memory.payload}")
            if hasattr(memory, 'vector') and memory.vector is not None:
                print(f"  Vector: {memory.vector[:5]}... (truncated)")  # Print first 5 elements of vector
            else:
                print("  Vector: Not available")
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