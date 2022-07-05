
from email import header
from typing import Tuple
from attrs import define, field, validators

import numpy as np
from tabulate import tabulate

from economicenvironment import EconomicEnvironment

@define
class Results:

    env: EconomicEnvironment = field(factory=EconomicEnvironment)
    competitive_profits: np.array = field(init=False)
    average_prices: np.array = field(init=False)
    average_profits: np.array = field(init=False)
    monopoly_profits: np.array = field(init=False)
    normalised_profits: np.array = field(init=False)

    price_history: list = field(factory=list)
    quantity_history: list = field(factory=list)
    reward_history: list = field(factory=list)


    def __attrs_post_init__(self):
        self.average_prices = np.array(self.env.price_history)[-25000:].mean(axis=0)
        self.average_profits = np.array(self.env.reward_history)[-25000:].mean(axis=0)
        self.competitive_profits = self.competitive_profits_compute()
        self.monopoly_profits = self.monopoly_profits_compute()
        self.normalised_profits = self.normalised_measure()
        self.price_history = self.env.price_history
        self.quantity_history = self.env.quantity_history
        self.reward_history = self.env.reward_history

    
    def competitive_profits_compute(self) ->  np.array:
        quality_array = np.array([agent.quality for agent in self.env.agents])
        marginal_cost_array = np.array([agent.marginal_cost for agent in self.env.agents])
        quantity_demand_given_competitive_prices = self.env.demand.get_quantity_demand(self.env.competitive_prices_array, quality_array)
        competitive_profits = (self.env.competitive_prices_array - marginal_cost_array) * quantity_demand_given_competitive_prices
        return competitive_profits

    def monopoly_profits_compute(self) -> np.array:
        quality_array = np.array([agent.quality for agent in self.env.agents])
        marginal_cost_array = np.array([agent.marginal_cost for agent in self.env.agents])
        quantity_demand_given_monopoly_prices = self.env.demand.get_quantity_demand(self.env.monopoly_prices_array, quality_array)
        monopoly_profits = (self.env.monopoly_prices_array - marginal_cost_array)* quantity_demand_given_monopoly_prices
        return monopoly_profits

    def normalised_measure(self) -> np.array:
        return (self.average_profits- self.competitive_profits)/(self.monopoly_profits- self.competitive_profits)

        
    def print_results(self):
        
        name = [agent.get_name() for agent in self.env.agents]
        desc = [agent.get_parameters() for agent in self.env.agents]

        print(tabulate({"Name": name,
                    "Bertrand-Nash Price": self.env.competitive_prices_array,
                    "Monopoly Price": self.env.monopoly_prices_array,
                    "Average Price": self.average_prices,
                    "Bertrand-Nash Profit": self.competitive_profits,
                    "Monopoly Profit": self.monopoly_profits,
                    "Normalised Profits": self.normalised_profits,
                }, 
                headers="keys"))
            
        print(tabulate({"Name":name, "Description":desc },headers="keys"))
        print("\n")