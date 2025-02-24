import torch.optim as optim


class OptimizerManager:
    def __init__(self, lr):
        self.lr = lr

    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.lr)

    def update(self, model, loss):
        optimizer = self.get_optimizer(model)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()