from abc import ABC

import numpy as np


class _SimpleSimulator(ABC):
    def __init__(self, seed = 12345, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(seed)
    
    def observe(self):
        return None
