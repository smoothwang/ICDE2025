import random

class NodeFeatureGenerator:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def generate_features(self):
        return [random.uniform(0.4, 0.6) for _ in range(self.feature_dim)]

class LabelAssigner:
    def __init__(self, anomaly_ratio):
        self.anomaly_ratio = anomaly_ratio

    def assign_label(self):
        if random.random() < self.anomaly_ratio:
            return 1  # Anomalous node
        else:
            return 0  # Normal node