import torch
import networkx as nx
from torch_geometric.utils import from_networkx


class GraphGenerator:
    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def generate_graph(self):
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        node_features = torch.rand((self.num_nodes, 5))  # Assuming feature_dim = 5
        return Data(x=node_features, edge_index=edge_index)


class FeatureAssigner:
    def assign_features(self, graph):
        for node in graph.nodes:
            graph.nodes[node]["x"] = torch.rand((5))  # Assuming feature_dim = 5
            graph.nodes[node]["y"] = 1 if torch.sum(graph.nodes[node]["x"]) > 2.5 else 0
        return graph