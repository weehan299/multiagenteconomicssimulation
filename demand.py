from attr import define,field
import numpy as np
import math
from typing import Tuple


@define
class Demand:
    mu: float = field(default=0.25)
    a0: float = field(default=0)

    def get_quantity_demand(self, prices: np.array, qualities: np.array) -> np.array:
        """returns a value from 0 to 1 based on the multinomial logit demand. """

        return np.exp((qualities - prices)/self.mu)/(sum(np.exp((qualities-prices)/self.mu)) + np.exp(self.a0/self.mu))

