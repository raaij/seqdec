import numpy as np

from bandits._base import _BanditBase


class EpsilonGreedyBandit(_BanditBase):
    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)
    
    def choose(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(low=0, high=self.k)
        else:
            arm = np.argmax(self.q)
        return arm
    
    def update(self, arm, reward):
        self.q[arm] += (1 / (self.n[arm] + 1)) * (reward - self.q[arm])
        self.n[arm] += 1
