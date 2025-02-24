import torch

class FeatureNormalizer:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.mean = torch.zeros(feature_dim)
        self.std = torch.ones(feature_dim)

    def compute_statistics(self, dataset):
        for data in dataset:
            self.mean += data.x.mean(dim=0)
            self.std += data.x.std(dim=0)
        self.mean /= len(dataset)
        self.std /= len(dataset)

    def normalize(self, data):
        return (data.x - self.mean) / self.std