from abc import ABC

import pandas as pd


class HypothesisTest(ABC):
    def __init__(self, sim, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sim = sim
        self.alpha = alpha
    
    def _get_result(self) -> pd.DataFrame:
        # TODO: Some form of allocation
        result = []

        for option in range(self.sim.k):
            data = {
                'option': option,
                0: 0,
                1: 0
            }
            for _ in range(5_000):
                reward, _ = self.sim.play(option)
                data[reward] += 1

            result.append(data)
        
        result = pd.DataFrame(result).set_index('option')
        return result


class FisherExactTest(HypothesisTest):
    MAX_OPTION = 2

    ERROR_MORE_THAN_MAX_OPTION = (
        "ERROR: Only a simulator with 2 options is supported for Fisher's exact test."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.sim.k == self.MAX_OPTION, self.ERROR_MORE_THAN_MAX_OPTION

    def run(self):
        from scipy.stats import fisher_exact
        data = self._get_result()
        _, p = fisher_exact(data.values)
        print("H_0: u_A = u_B")
        print("H_1: u_A != u_B")
        print(f"p value is {round(p, 5)}")
        print(f"with significance level {self.alpha}")
        if p < self.alpha:
            print("we reject the null hypothesis")
        else:
            print("we fail to reject the null hypothesis")
