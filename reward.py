from attr import define,field
import abc
import numpy as np
import math
import itertools
from agents.agent import Agent, Binary_State_QLearning
from typing import Tuple, List

@define
class Reward(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array,action_space:np.array):
        raise NotImplementedError
        

    def get_name(self):
        return type(self).__name__

@define
class StandardReward(Reward):
    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array: np.array,action_space:np.array) -> np.array:
        return (price_array - marginal_cost_array) * quantity_array

@define
class ExpectedReward(Reward):
    agents: List[Binary_State_QLearning] = field(factory=list)

    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array: np.array,action_space:np.array) -> np.array:
        new_price_array = [agent.expected_value for agent in self.agents]
        return (new_price_array - marginal_cost_array) * quantity_array


@define
class MixedStrategyPayoff(Reward):
    agents: List[Binary_State_QLearning] = field(factory=list)
    
    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array: np.array,action_space:np.array) -> np.array:
        print("#####################") 
        print("Price Array: ",price_array)
        print("Action Space: ",action_space)
        reward = np.zeros(len(self.agents))
        prod_prob = 1
        for i in range(len(self.agents)):
            index_in_action_space = np.where(action_space == price_array[i])[0][0]
            prod_prob *= self.agents[i].prob_weights[index_in_action_space]
            print(self.agents[i].prob_weights[index_in_action_space])
        for state in itertools.product(self.action_space, repeat=len(self.agents)):
            pass
        pass

            
