from attr import define,field
import abc
import numpy as np
import math
import itertools
from demand import Demand
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
    demand: Demand = field(factory=Demand)

    def get_reward(self, price_array:np.array, marginal_cost_array:np.array, quantity_array,action_space:np.array):
        return (price_array - marginal_cost_array) * quantity_array
        
    # TODO: all probability weights for each agents, 
    def get_average_reward(self, marginal_cost_array:np.array, quality_array:np.array, action_space:np.array,prob_weights_array:np.array) -> np.array:
        # for each state, you have to find its corresponding probability (product of prob weights of each agents)
        # then you have to get the corresponding demand for firm i
        result = np.zeros(len(marginal_cost_array)) 
        for state in itertools.product(action_space, repeat=len(marginal_cost_array)):
            #print(state)
            prob_prod = 1
            for i in range(len(state)):
                #print(action_space, state[i], np.where(action_space == state[i])[0][0], prob_weights_array[i][np.where(action_space == state[i])[0][0]])
                #this is the probability of firm i doing the action that it did in the state
                prob_prod *= prob_weights_array[i][np.where(action_space == state[i])[0][0]]
            quantity_array = self.demand.get_quantity_demand(state, quality_array)
            #print("prob prod: ", prob_prod, (state - marginal_cost_array)*quantity_array, prob_prod*(state - marginal_cost_array)*quantity_array)
            result += prob_prod*(state - marginal_cost_array)*quantity_array
        #print("result: ", result)
        return result
            
    
    def get_conditional_reward(self, marginal_cost_array:np.array, quality_array:np.array, action_space:np.array,prob_weights_array:np.array) -> np.array:
        # prob_prod will be an array, 
        result = []
        for conditioned_price in action_space:
            conditioned_payoff = np.zeros(len(marginal_cost_array)) 
            for state in itertools.product(action_space, repeat=len(marginal_cost_array)):
                if conditioned_price in state: # conditioned price must be in the state (can be for any agent)
                    #print("state: ", state, "conditioned price: ", conditioned_price)
                    prob_prod = np.ones(len(state))
                    for j in range(len(prob_prod)):
                        for i in range(len(state)):
                            if i!=j:
                                if state[j] != conditioned_price:
                                    prob_prod[j] = 0
                                else:
                                    prob_prod[j] *= prob_weights_array[i][np.where(action_space == state[i])[0][0]]
                                #print("inner prob prod: ", prob_prod)
                    #print("outer prob prod: ", prob_prod)
                    quantity_array = self.demand.get_quantity_demand(state, quality_array)
                    #print("prob prod: ", prob_prod, (state - marginal_cost_array)*quantity_array, prob_prod*(state - marginal_cost_array)*quantity_array)
                    conditioned_payoff += prob_prod*(state - marginal_cost_array)*quantity_array
            result.append(conditioned_payoff)
        result = np.transpose(np.asarray(result))
        #print("result: ", result)
        return result


            
