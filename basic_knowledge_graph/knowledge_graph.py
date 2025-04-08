from pyvis.network import Network

def create_knowledge_graph():
    # Create a new network
    net = Network(
        select_menu=True,
        filter_menu=True,
    )

    # Add nodes
    net.add_node("Mary", label="Mary")
    net.add_node("John", label="John")
    net.add_node("Teradata", label="Teradata")

    # Add edges to represent the working relationships
    net.add_edge("Mary", "Teradata", title="works at")
    net.add_edge("John", "Teradata", title="works at")

    # Save and show the network
    net.show("index.html", local=False, notebook=False)

