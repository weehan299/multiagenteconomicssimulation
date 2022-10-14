from attr import define,field
import abc
import numpy as np
import math
from agents.agent import Agent, Binary_State_QLearning
from typing import Tuple, List

@define
class Reward(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array):
        raise NotImplementedError
        

    def get_name(self):
        return type(self).__name__

@define
class StandardReward(Reward):
    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array: np.array) -> np.array:
        return (price_array - marginal_cost_array) * quantity_array


@define
class ExpectedReward(Reward):
    agents: List[Binary_State_QLearning] = field(factory=list)

    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array: np.array) -> np.array:
        new_price_array = [agent.expected_value for agent in self.agents]
        #print([agent.prob_weights for agent in self.agents])
        return (new_price_array - marginal_cost_array) * quantity_array

