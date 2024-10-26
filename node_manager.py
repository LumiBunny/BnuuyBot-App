class NodeManager:
    def __init__(self, registry):
        self.registry = registry
        self.current_node = None

    def set_initial_node(self, node_name):
        self.current_node = self.get_node_by_name(node_name)

    def get_node_by_name(self, name):
        return self.registry.nodes.get(name)  # Updated to use registry's nodes dict directly

    def transition_to_node(self, node_name):
        new_node = self.get_node_by_name(node_name)
        if new_node:
            self.current_node = new_node
            print(f"Transitioned to node: {self.current_node.name}")
        else:
            print(f"Node {node_name} not found!")

    async def process_current_node(self, *args):
        if self.current_node:
            return await self.current_node.process(*args)