import torch
import random
import networkx as nx
from torch_geometric.utils import from_networkx


def generate_synthetic_data(num_nodes=100, num_edges=300, feature_dim=5, anomaly_ratio=0.1):
    """
    Generate synthetic data to simulate normal and anomalous nodes in a transaction network.
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
        feature_dim: Dimensionality of node features
        anomaly_ratio: Ratio of anomalous nodes
    Returns:
        graph_data: PyTorch Geometric formatted graph data
    """
    # Step 1: Create a random graph structure
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=False)

    # Step 2: Assign features and labels to nodes
    for node in G.nodes:
        # Normal node features: Random vector with mean 0.5
        G.nodes[node]["x"] = [random.uniform(0.4, 0.6) for _ in range(feature_dim)]

        # Anomalous node features: Random vector with mean 1.5
        if random.random() < anomaly_ratio:
            G.nodes[node]["x"] = [random.uniform(1.4, 1.6) for _ in range(feature_dim)]
            G.nodes[node]["y"] = 1  # Anomalous node
        else:
            G.nodes[node]["y"] = 0  # Normal node

    # Step 3: Convert to PyTorch Geometric format
    graph_data = from_networkx(G)

    # Convert node features and labels to Tensors
    graph_data.x = torch.tensor([G.nodes[n]["x"] for n in G.nodes], dtype=torch.float)
    graph_data.y = torch.tensor([G.nodes[n]["y"] for n in G.nodes], dtype=torch.long)

    return graph_data