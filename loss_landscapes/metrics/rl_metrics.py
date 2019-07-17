import torch
import torch.autograd
from loss_landscapes.metrics.metric import Metric


class ExpectedReturnMetric(Metric):
    def __init__(self, gym_environment, n_episodes):
        super().__init__()
        self.gym_environment = gym_environment
        self.n_episodes = n_episodes

    def __call__(self, agent):
        returns = []

        # compute total return for each episode
        for episode in range(self.n_episodes):
            episode_return = 0
            obs, reward, done, _ = self.gym_environment.step(
                agent(torch.from_numpy(self.gym_environment.reset()).float())
            )
            episode_return += reward

            while not done:
                obs, reward, done, info = self.gym_environment.step(
                    agent(torch.from_numpy(obs).float())
                )
                episode_return += reward
            returns.append(episode_return)

        # return average of episode returns
        return sum(returns) / len(returns)
