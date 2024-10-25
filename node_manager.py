class NodeManager: 
    def __init__(self, registry): 
        self.registry = registry 
        self.current_node = None 

    def set_initial_node(self, node_name): 
        self.current_node = self.get_node_by_name(node_name) 

    def get_node_by_name(self, name): 
        for node in self.registry.get_all_nodes(): 
            if node.name == name: 
                return node 
        return None 

    def transition_to_node(self, node_name): 
        new_node = self.get_node_by_name(node_name) 
        if new_node: 
            self.current_node = new_node 
            print(f"Transitioned to node: {self.current_node.name}") 
        else: 
            print(f"Node {node_name} not found!") 

    def handle_user_input(self, user_input): 
        # Example logic to decide next node based on user input 
        if "happy" in user_input: 
            self.transition_to_node("happyNode") 
        else: 
            print("Staying at current node.") 

    def run(self): 
        # Example of running the node manager 
        while True: 
            user_input = input("Enter your input: ") 
            self.handle_user_input(user_input) 
            # You can add more logic here to process the current node's function 