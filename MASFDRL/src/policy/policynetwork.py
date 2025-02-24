import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from optimizer import OptimizerManager
from action_selector import ActionSelector

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Policy network that outputs action probability distributions.
        Args:
            input_dim: Dimension of the input state.
            output_dim: Dimension of the action space.
            hidden_dim: Dimension of the hidden layer.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        """
        Forward pass to generate action distributions.
        Args:
            state: Current state [batch_size, input_dim]
        Returns:
            Action probability distributions [batch_size, output_dim]
        """
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs


class MultiAgentPolicy:
    def __init__(self, feature_dim, num_nodes, hidden_dim=128, lr=0.001):
        """
        Manager for the policy networks of multiple agents.
        Args:
            feature_dim: Dimension of the input features (e.g., node features).
            num_nodes: Dimension of the action space (e.g., number of target nodes).
            hidden_dim: Dimension of the hidden layer in the policy network.
            lr: Learning rate for the optimizers.
        """
        self.fraudster_policy = PolicyNetwork(feature_dim, num_nodes, hidden_dim)
        self.detector_policy = PolicyNetwork(feature_dim, num_nodes, hidden_dim)

        # Optimizer manager
        self.optimizer_manager = OptimizerManager(lr)

    def get_action(self, policy, state):
        """
        Generate an action based on the policy network and current state.
        Args:
            policy: Policy network (either fraudster or detector).
            state: Current state [batch_size, feature_dim]
        Returns:
            action: Sampled action.
            log_prob: Log probability of the action.
        """
        action_probs = policy(state)
        action_selector = ActionSelector(action_probs)
        action = action_selector.sample()
        log_prob = action_selector.log_prob(action)
        return action, log_prob

    def update_policy(self, policy, log_probs, rewards, gamma=0.99):
        """
        Update the policy network using policy gradient.
        Args:
            policy: Policy network (either fraudster or detector).
            log_probs: Log probabilities of the actions.
            rewards: Rewards received for the actions.
            gamma: Discount factor.
        """
        # Calculate discounted rewards
        discounted_rewards = self.calculate_discounted_rewards(rewards, gamma)
        # Calculate policy gradient loss
        loss = -torch.stack(log_probs) * discounted_rewards
        # Update policy
        self.optimizer_manager.update(policy, loss)

    @staticmethod
    def calculate_discounted_rewards(rewards, gamma):
        """
        Calculate discounted rewards.
        Args:
            rewards: List of rewards.
            gamma: Discount factor.
        Returns:
            Tensor of discounted rewards.
        """
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        return torch.tensor(discounted_rewards)