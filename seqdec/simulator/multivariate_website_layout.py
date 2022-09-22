from re import M
from typing import Tuple

import numpy as np
import scipy.optimize
from scipy.stats import norm

from seqdec.simulator._base import _ContextualSimulator


class _Layout:
    def __init__(self, A, n):
        self.A = A
        self.n = n
        self.ohe = self._get_ohe()
            
    def __repr__(self):
        result = 'Layout:'
        for i, c in enumerate(self.A):
            result += f'\n  Widget {i+1}: variation {c+1}'
        return result
    
    def _get_ohe(self):
        result = np.zeros(sum(self.n))
        start = 0
        for i, c in enumerate(self.A):
            result[start + c] = 1
            start += self.n[i]
        
        return result.reshape(1, -1)


class MultivariateWebsiteLayout(_ContextualSimulator):
    DESIRED_VARIANCE = 0.002

    ERROR_ALPHA_LENGTH = "alpha should be a tuple with 3 floats"

    def __init__(
        self, 
        number_of_context_variables: int,
        number_of_widget_variations: Tuple[int],
        alpha: Tuple[float],
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert len(alpha) == 3, self.ERROR_ALPHA_LENGTH

        self.L = number_of_context_variables
        self.n = number_of_widget_variations
        self.k = sum(number_of_widget_variations)
        self.alpha_1, self.alpha_2, self.alpha_c = alpha
        
        self._initialize_weights()
        self._initialize_layouts()
        self._initialize_beta()
    
    def _initialize_weights(self):
        # bias
        self.W_0 = np.random.normal(size=1)
        # first order widget effect
        self.W_1 = np.random.normal(size=self.k).reshape(1, -1)
        # second order widget-widget effect
        self.W_2 = np.random.normal(size=(self.k, self.k))
        # first order context effect
        self.W_c = np.random.normal(size=self.L)
        # second order context-widget effect
        self.W_1c = np.random.normal(size=(self.L, self.k))

    def _initialize_layouts(self):
        # FIXME: Only works with 3 widgets

        import itertools 

        layouts = [
            _Layout(A=[i, j, k], n=self.n) for (i, j, k) in itertools.product(*[
                list(range(n_i)) for n_i in self.n
            ])
        ]
        self.layouts = layouts
    
    def _initialize_beta(self):
        X = self.observe() # take random context for stabilisation

        base_p = np.array([
            np.sum(
                self.W_0
                + self.alpha_1 * np.sum(self.W_1 * layout.ohe)
                + self.alpha_2 * np.sum(self.W_2 * (layout.ohe.T @ layout.ohe))
                + self.alpha_c * np.sum(self.W_c * X)
                + self.alpha_c * np.sum(self.W_1c * (X.T @ layout.ohe))
            )
            for layout in self.layouts
        ])

        res = scipy.optimize.root_scalar(
            f = lambda beta: np.var(norm.cdf((1 / beta) * base_p)) - self.DESIRED_VARIANCE,
            x0 = 1,
            x1 = 10
        )
        self.beta = res.root
    
    def observe(self):
        X =  np.random.randint(low=0, high=2, size=self.L).reshape(1, -1)
        return X
    
    def play(self, arm):
        X = self.observe()

        # Compute click probabilities 
        p = norm.cdf((1 / self.beta) * np.array([
            np.sum(
                self.W_0
                + self.alpha_1 * np.sum(self.W_1 * layout.ohe)
                + self.alpha_2 * np.sum(self.W_2 * (layout.ohe.T @ layout.ohe))
                + self.alpha_c * np.sum(self.W_c * X)
                + self.alpha_c * np.sum(self.W_1c * (X.T @ layout.ohe))
            )
            for layout in self.layouts
        ]))

        reward = int(np.random.random() < p[arm])
        return reward, X

