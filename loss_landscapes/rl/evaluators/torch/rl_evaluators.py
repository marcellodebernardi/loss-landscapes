"""
A library of pre-written evaluation functions for PyTorch RL agents using openai/gym RL environments.

The classes and functions in this module cover evaluating aspects of the "cumulative return landscape"
of reinforcement learning models in RL environments.
"""


from abc import ABC, abstractmethod
from loss_landscapes.common.evaluators.evaluator import Evaluator


class GymEvaluator(Evaluator, ABC):
    def __init__(self, env, n_samples=10, max_episode_length=None):
        super().__init__()
        self.env = env
        self.n_samples = n_samples
        self.max_episode_length = max_episode_length

    @abstractmethod
    def __call__(self, agent):
        pass


class CumulativeReturnEvaluator(GymEvaluator):
    """ Abstract class for PyTorch reinforcement learning cumulative reward evaluation in a gym environment. """
    def __init__(self, env, n_samples=15, max_episode_length=None):
        super().__init__(env, n_samples, max_episode_length)

    def __call__(self, agent_act_fn):
        self.env.reset()
        cumulative_return = 0

        for _ in range(self.n_samples):
            obs, reward, done, info = self.env.step(agent_act_fn(self.env.reset()))
            cumulative_return += reward
            t = 0

            while not done and t != self.max_episode_length:
                action = agent_act_fn(obs)
                obs, reward, done, info = self.env.step(action)

                cumulative_return += reward
                t += 1

        return cumulative_return / self.n_samples
