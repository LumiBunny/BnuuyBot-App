class Node: 
    def __init__(self, name, funcs=None, vector=None): 
        self.name = name  # Name of the node 
        self.funcs = funcs if funcs is not None else []  # List of functions associated with the node 
        self.vector = vector  # Optional vector for embeddings or other purposes 

    def run_functions(self): 
        """Execute all functions associated with this node.""" 
        for func in self.funcs: 
            if func: 
                func()  # Call each function in the list 

class NodeRegistry: 
    def __init__(self): 
        self.nodes = [] 

    def add_node(self, node): 
        self.nodes.append(node) 

    def get_all_nodes(self): 
        return self.nodes 

# Example usage 
registry = NodeRegistry() 
registry.add_node(Node("happyNode", "happy_function")) 
# Add more nodes as needed 

# Add all your functions here I guess