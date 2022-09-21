from abc import ABC
import numpy as np


class _SimpleSimulator(ABC):
    def __init__(self, seed = 12345, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(seed)
    
    def observe(self):
        return None


class BernoulliSimulator(_SimpleSimulator):
    def __init__(self, k, p = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.proba = p if p else np.random.uniform(size=k)

    def play(self, arm):
        return np.random.random() < self.proba[arm]
