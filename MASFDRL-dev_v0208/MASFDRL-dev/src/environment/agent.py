import torch


class Agent:
    def __init__(self, agent_type, agent_id):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.reward = 0

    def calculate_reward(self, state):
        if self.agent_type == "fraudster":
            return -torch.sum(state).item()
        elif self.agent_type == "detector":
            return 1 if torch.sum(state) > 1.5 else -0.5

    def reset_reward(self):
        self.reward = 0