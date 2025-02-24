import torch
from torch.distributions import Categorical


class ActionSelector:
    def __init__(self, action_probs):
        self.action_probs = action_probs
         self.action_dist = Categorical(action_probs)

    def sample(self):
        return self.action_dist.sample()

    def log_prob(self, action):
        return self.action_dist.log_prob(action)