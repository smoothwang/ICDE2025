class RewardManager:
    def discount(self, rewards, gamma):
        discounted_rewards = []
         R = 0
         for r in reversed(rewards):
             R = r + gamma * R
             discounted_rewards.insert(0, R)
         return torch.tensor(discounted_rewards)