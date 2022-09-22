"""
FIXME: Only have 1 base simulator class, any simulator needs to return context
"""

from abc import ABC, abstractmethod

import numpy as np


class _BaseSimulator(ABC):
    def __init__(self, seed = 12345, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(seed)


class _SimpleSimulator(_BaseSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def play(self):
        # returns reward, context
        raise NotImplementedError()


class _ContextualSimulator(_BaseSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def play(self):
        # returns reward, context
        raise NotImplementedError()
