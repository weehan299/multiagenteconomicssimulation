from attrs import define, field, validators
from typing import List
import numpy as np
import random
from tqdm import tqdm


from pricecompute import PriceCompute
from demand import Demand
from agent import Agent
from validators import validate_total_periods,validate_action_space_num,validate_xi

########## economic environment ##########
#agents
#action space
#competitive and monopoly prices <- demand <- quality, price
#state space
#demand, marginal costs, quality


@define
class EconomicEnvironment:
    agents: List[Agent] = field(factory=list)

    demand: Demand = field(factory=Demand)
    action_space: np.array = field(init=False)
    marginal_cost_array: np.array = field(init=False)
    quality_array: np.array = field(init=False)
    competitive_prices_array: np.array = field(init=False)
    monopoly_prices_array: np.array = field(init=False)

    action_space_num: int = field(default = 15, validator=[validators.instance_of(int), validate_action_space_num] )
    total_periods: int = field(default=1000, validator=[validators.instance_of(int), validate_total_periods])
    xi: float = field(default=0.1, validator=[validators.instance_of(float), validate_xi])

    price_history: list = field(factory=list)
    quantity_history: list = field(factory=list)
    reward_history: list = field(factory=list)

    def __attrs_post_init__(self):
        self.marginal_cost_array = np.array([agent.marginal_cost for agent in self.agents])
        self.quality_array = np.array([agent.quality for agent in self.agents])
        self.monopoly_prices_array = PriceCompute(demand= self.demand).monopoly_price_compute(self.marginal_cost_array, self.quality_array)
        self.competitive_prices_array = PriceCompute(demand = self.demand).competitive_price_compute(self.marginal_cost_array, self.quality_array)
        self.action_space = self.init_action_space(min(self.competitive_prices_array),max(self.monopoly_prices_array),self.xi, self.action_space_num)

    @staticmethod
    def init_action_space(competitive_price:float, monopoly_price: float, xi:float, step: int) -> np.array:
        return np.linspace(competitive_price - xi, monopoly_price + xi, step)
        
    def run_simulation(self):
        prev_state = random.choices(self.action_space, k=len(self.agents))
        curr_state = np.array(
            [agent.pick_strategy(prev_state, self.action_space, 0) for agent in self.agents]
        )
        quantity_array = self.demand.get_quantity_demand(curr_state, self.quality_array)
        prev_reward_array = (curr_state - self.marginal_cost_array) * quantity_array

        for time in tqdm(range(self.total_periods)):

            next_state = np.array(
                [agent.pick_strategy(curr_state, self.action_space, time) for agent in self.agents]
            )
            
            new_quantity_array = self.demand.get_quantity_demand(next_state, self.quality_array)
            reward_array = (next_state - self.marginal_cost_array) * new_quantity_array

            for agent, action, prev_action, reward, prev_reward in zip(
                self.agents, next_state, curr_state, reward_array, prev_reward_array
            ):
                agent.learn(
                    old_state = prev_state,
                    curr_state= curr_state,
                    new_state= next_state,
                    action_space= self.action_space,
                    prev_reward = prev_reward,
                    reward= reward,
                    prev_action = prev_action,
                    action = action,
                        )
            prev_state = curr_state
            curr_state = next_state
            prev_reward_array = reward_array

            self.price_history.append(prev_state)
            self.quantity_history.append(quantity_array)
            self.reward_history.append(reward_array)
                
            