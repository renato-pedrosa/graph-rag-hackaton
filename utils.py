def extract_labels_and_relationships(data):
    node_labels = set()
    rel_types = set()

    # Extract node labels
    for node in data.get("Nodes", []):
        if len(node) > 1:
            node_labels.add(node[1])

    # Extract relationship types
    for edge in data.get("Edges", []):
        if len(edge) > 1:
            rel_types.add(edge[1])

    return {
        "node-labels": sorted(node_labels),
        "rel_types": sorted(rel_types)
    }
 