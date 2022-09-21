from abc import ABC, abstractmethod


class _BanditBase(ABC):
    def __init__(self, sim, iter=10_000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = sim
        self.k = self.sim.k
        self.iter = iter
    
    @abstractmethod
    def choose(self):
        raise NotImplementedError()
    
    @abstractmethod
    def update(self, arm, reward):
        raise NotImplementedError()

    def run(self):
        for _ in range(self.iter):
            arm = self.choose()
            reward = self.sim.play(arm)
            self.update(arm, reward)
