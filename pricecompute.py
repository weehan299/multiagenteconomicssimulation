
from attrs import define
import numpy as np
import copy
from demand import Demand,field

from scipy.optimize import fsolve, minimize

@define
class PriceCompute:
    demand: Demand = field(factory=Demand)

    def profits_of_firm_i(self, given_price: float, price_array:np.array, quality_array: np.array, marginal_cost_array: np.array,i: int) -> float:

        price_array_copy = copy.deepcopy(price_array)
        price_array_copy[i] = given_price
        new_quantity_demand = self.demand.get_quantity_demand(price_array_copy, quality_array)[i]
        return -1 * (price_array_copy[i] - marginal_cost_array[i]) * new_quantity_demand

    def reaction_function(self, price_array: np.array, quality_array:np.array, marginal_cost_array:np.array, i:int ) -> float:
        # optimal response function
        return  minimize(fun=self.profits_of_firm_i , x0 = np.array(price_array[i]), args = (price_array, quality_array, marginal_cost_array, i)).x[0]

    def vector_reaction(self, nash_prices:np.array, quality_array:np.array, marginal_cost_array:np.array) -> np.array:

        return np.array(nash_prices) - np.array(
            [self.reaction_function(nash_prices, quality_array, marginal_cost_array, i) for i in range(len(nash_prices))]
        )
            

    def competitive_price_compute(self, marginal_cost_array: np.array, quality_array:np.array) -> np.array:
        return fsolve(self.vector_reaction, marginal_cost_array,args=(quality_array,marginal_cost_array))

    def joint_profit(self,price_array:np.array, quality_array:np.array, marginal_cost_array:np.array) -> float:
        return -1*np.sum((price_array - marginal_cost_array) * self.demand.get_quantity_demand(price_array, quality_array))

    def monopoly_price_compute(self, marginal_cost_array: np.array, quality_array:np.array) -> np.array:
        return minimize(fun=self.joint_profit,x0=marginal_cost_array,args=(quality_array, marginal_cost_array),).x


"""
y = Demand()
x = PriceCompute(y)
print(x.competitive_price_compute(np.array([1,1,1,1]),np.array([2,2,2,2])))
print(x.monopoly_price_compute(np.array([1,1,1,1]),np.array([2,2,2,2])))
"""