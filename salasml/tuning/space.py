import numpy as np
from scipy import stats
import warnings


class Real:
    """
    Dimensi ruang pencarian nilai real untuk RandomizedSearchCV.

    == Contoh ==
    Real(0, 1)
    Real(-3, 3, prior='log-uniform')

    == Parameter ==
    low  : batas bawah (inklusif)
    high : batas atas (inklusif)
    prior: {'uniform', 'log-uniform'}
    """
    def __init__(self, low, high, prior='uniform'):
        if low > high:
            low, high = high, low
            warnings.warn("'low' lebih besar dari 'high', otomatis ditukar")
        self.low = low
        self.high = high
        self.prior = prior

    def rvs(self, random_state):
        uniform = self._uniform_inclusive(loc=self.low, scale=self.high - self.low)
        if self.prior == 'uniform':
            return uniform.rvs(random_state=random_state)
        elif self.prior == 'log-uniform':
            return np.power(10, uniform.rvs(random_state=random_state))
        else:
            raise Exception("prior yang didukung: {'uniform', 'log-uniform'}")

    @staticmethod
    def _uniform_inclusive(loc, scale):
        return stats.uniform(loc=loc, scale=np.nextafter(scale, scale + 1.))

    def __repr__(self):
        return f"Real(low={self.low}, high={self.high}, prior='{self.prior}')"


class Integer:
    """
    Dimensi ruang pencarian nilai integer untuk RandomizedSearchCV.

    == Contoh ==
    Integer(20, 80)
    Integer(100, 200)

    == Parameter ==
    low : batas bawah (inklusif)
    high: batas atas (inklusif)
    """
    def __init__(self, low, high):
        if low > high:
            low, high = high, low
            warnings.warn("'low' lebih besar dari 'high', otomatis ditukar")
        self.low = low
        self.high = high

    def rvs(self, random_state):
        rand_int = stats.randint(low=self.low, high=self.high + 1)
        return rand_int.rvs(random_state=random_state)

    def __repr__(self):
        return f"Integer(low={self.low}, high={self.high})"
