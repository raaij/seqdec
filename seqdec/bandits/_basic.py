import numpy as np

from seqdec.bandits._base import _BanditBase


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


class UCBBandit(_BanditBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)
        self.ucb = np.zeros(self.k)
        self.t = 0
    
    def choose(self):
        if self.t < self.k:
            arm = self.t
        else:
            arm = np.argmax(self.ucb)
        return arm
    
    def update(self, arm, reward):
        self.t += 1
        
        self.q[arm] += (1 / (self.n[arm] + 1)) * (reward - self.q[arm])
        self.n[arm] += 1

        if self.t >= self.k: # This implies all arms have been played at least once.
            for arm in range(self.k):
                self.ucb[arm] = self.q[arm] + np.sqrt(np.log(self.t) / self.n[arm])
