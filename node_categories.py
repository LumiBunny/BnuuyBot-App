from nodes import Node 

class EmotionNode(Node): 
    def __init__(self, name, emotion_type, funcs=None, vector=None): 
        super().__init__(name, funcs, vector)  # Initialize the base Node class 
        self.emotion_type = emotion_type  # Specific to emotion nodes 

class TopicNode(Node): 
    def __init__(self, name, topic, funcs=None, vector=None): 
        super().__init__(name, funcs, vector)  # Initialize the base Node class 
        self.topic = topic  # Specific to topic nodes 