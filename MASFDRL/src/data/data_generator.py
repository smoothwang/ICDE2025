import torch
import random
import networkx as nx
from torch_geometric.utils import from_networkx
from data_models import NodeFeatureGenerator, LabelAssigner

class SyntheticDataGenerator:
    def __init__(self, num_nodes=100, num_edges=300, feature_dim=5, anomaly_ratio=0.1):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.feature_dim = feature_dim
        self.anomaly_ratio = anomaly_ratio

    def generate_graph(self):
        G = nx.gnm_random_graph(self.num_nodes, self.num_edges, directed=False)
        return G

    def assign_features_and_labels(self, G):
        feature_generator = NodeFeatureGenerator(self.feature_dim)
        label_assigner = LabelAssigner(self.anomaly_ratio)

        for node in G.nodes:
            features = feature_generator.generate_features()
            label = label_assigner.assign_label()
            G.nodes[node]["x"] = features
            G.nodes[node]["y"] = label

        return G

    def generate_synthetic_data(self):
        G = self.generate_graph()
        G = self.assign_features_and_labels(G)
        graph_data = from_networkx(G)
        graph_data.x = torch.tensor([G.nodes[n]["x"] for n in G.nodes], dtype=torch.float)
        graph_data.y = torch.tensor([G.nodes[n]["y"] for n in G.nodes], dtype=torch.long)
        return graph_data