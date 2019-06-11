from abc import ABC, abstractmethod
import torch
import torch.autograd
from loss_landscapes.evaluators.evaluators import Evaluator


class TorchReinforcementEvaluator(Evaluator, ABC):
    @abstractmethod
    def __init__(self, gym_environment, n_episodes):
        super().__init__()
        self.env = gym_environment
        self.n_episodes = n_episodes

    @abstractmethod
    def __call__(self, agent):
        pass


class ExpectedReturnEvaluator(TorchReinforcementEvaluator):
    def __init__(self, gym_environment, n_episodes):
        super().__init__(gym_environment, n_episodes)

    def __call__(self, agent):
        returns = []

        for episode in range(self.n_episodes):
            cum_reward = 0

            obs, reward, done, _ = self.env.step(
                agent(torch.from_numpy(self.env.reset()).float())
            )
            cum_reward += reward

            while not done:
                obs, reward, done, info = self.env.step(
                    agent(torch.from_numpy(obs).float())
                )
                cum_reward += reward

            returns.append(cum_reward)

        return sum(returns) / len(returns)
