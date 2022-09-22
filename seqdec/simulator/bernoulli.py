import numpy as np

from seqdec.simulator._base import _SimpleSimulator


class BernoulliSimulator(_SimpleSimulator):
    def __init__(self, k, p = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.proba = p if p else np.random.uniform(size=k)

    def play(self, arm):
        return np.random.random() < self.proba[arm]
