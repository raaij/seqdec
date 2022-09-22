from abc import ABC, abstractmethod

import numpy as np


class _BaseSimulator(ABC):
    def __init__(self, seed = 12345, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(seed)


class _SimpleSimulator(_BaseSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _ContextualSimulator(_BaseSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def observe(self):
        raise NotImplementedError()
