import torch
from policy_network import MultiAgentPolicy
from multi_agent_env import MultiAgentEnvironment
from action_selector import ActionSelector
from reward_manager import RewardManager
from logger import Logger
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, env, policy_manager, num_episodes=1000, gamma=0.99, lr=0.001):
        self.env = env
         self.policy_manager = policy_manager
         self.num_episodes = num_episodes
         self.gamma = gamma
         self.lr = lr
         self.reward_manager = RewardManager()
         self.logger = Logger("training.log")
         self.action_selector = ActionSelector()

    def train(self):
        rewards_history = []
        for episode in range(self.num_episodes):
             state = self.env.reset()
            done = False
            episode_rewards = []
            while not done:
                 actions, log_probs = self._select_actions(state)
                 state, rewards, done = self.env.step(actions)
                 self._update_rewards(rewards, log_probs)
                 episode_rewards.append(sum(rewards))
            total_rewards = sum(episode_rewards)
             rewards_history.append(total_rewards)
             self.logger.log(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_rewards:.2f}")
             if (episode + 1) % 10 == 0:
                 self._log_training_progress(rewards_history)
        return rewards_history

    def _select_actions(self, state):
        fraudster_action, fraudster_log_prob = self.policy_manager.get_action(
             self.policy_manager.fraudster_policy, state)
        detector_action, detector_log_prob = self.policy_manager.get_action(
             self.policy_manager.detector_policy, state)
        return [
            {"type": "fraudster", "target": fraudster_action.item(), "amount": torch.rand(1).item()},
            {"type": "detector", "node": detector_action.item()},
        ], [fraudster_log_prob, detector_log_prob]

    def _update_rewards(self, rewards, log_probs):
         discounted_rewards = self.reward_manager.discount(rewards, self.gamma)
         self.policy_manager.update_policy(
             self.policy_manager.fraudster_optimizer,
             log_probs[0],
             discounted_rewards[0],
             self.gamma
         )
         self.policy_manager.update_policy(
             self.policy_manager.detector_optimizer,
             log_probs[1],
             discounted_rewards[1],
             self.gamma
         )

    def _log_training_progress(self, rewards_history):
         plt.plot(rewards_history)
         plt.xlabel("Episode")
         plt.ylabel("Total Reward")
         plt.title("Training Performance")
         plt.savefig("training_performance.png")

if __name__ == "__main__":
    num_nodes = 10
    num_edges = 20
    feature_dim = 5
    num_agents = 4
    max_steps = 50

    env = MultiAgentEnvironment(
        num_nodes=num_nodes,
        num_edges=num_edges,
        feature_dim=feature_dim,
        num_agents=num_agents,
        max_steps=max_steps
    )

    policy_manager = MultiAgentPolicy(
        feature_dim=feature_dim,
        num_nodes=num_nodes,
        hidden_dim=128,
        lr=0.001
    )

    trainer = Trainer(env, policy_manager, num_episodes=500)
    rewards_history = trainer.train()