
from email import header
from typing import Tuple

import numpy as np
from tabulate import tabulate

def competitive_profits_compute(env) ->  np.array:
    quality_array = np.array([agent.quality for agent in env.agents])
    marginal_cost_array = np.array([agent.marginal_cost for agent in env.agents])
    quantity_demand_given_competitive_prices = env.demand.get_quantity_demand(env.competitive_prices_array, quality_array)
    competitive_profits = (env.competitive_prices_array - marginal_cost_array) * quantity_demand_given_competitive_prices
    return competitive_profits

def monopoly_profits_compute(env) -> np.array:
    quality_array = np.array([agent.quality for agent in env.agents])
    marginal_cost_array = np.array([agent.marginal_cost for agent in env.agents])
    quantity_demand_given_monopoly_prices = env.demand.get_quantity_demand(env.monopoly_prices_array, quality_array)
    monopoly_profits = (env.monopoly_prices_array - marginal_cost_array)* quantity_demand_given_monopoly_prices
    return monopoly_profits

def normalised_measure(average_profits: np.array, competitive_profits: np.array, monopoly_profits: np.array) -> np.array:
    return (average_profits- competitive_profits)/(monopoly_profits- competitive_profits)

def get_agents_parameters(env):
    name = [agent.get_name() for agent in env.agents]
    desc = [agent.get_parameters() for agent in env.agents]
    
    print(tabulate({"Name":name, "Description":desc },headers="keys"))
    
def results(env):
    
    name = [agent.get_name() for agent in env.agents]

    average_prices = np.array(env.price_history).mean(axis=0)
    average_profits = np.array(env.reward_history).mean(axis=0)
    competitive_profits = competitive_profits_compute(env)
    monopoly_profits = monopoly_profits_compute(env)
    normalised_profits = normalised_measure(average_profits, competitive_profits, monopoly_profits)

    print(tabulate({"Name": name,
                "Competitive Price": env.competitive_prices_array,
                "Monopoly Price": env.monopoly_prices_array,
                "Average Price": average_prices,
                "Competitive Profit": competitive_profits,
                "Monopoly Profit": monopoly_profits,
                "Normalised Profits": normalised_profits,
            }, 
            headers="keys"))
    print("\n")