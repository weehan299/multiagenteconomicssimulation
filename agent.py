
from attrs import define, field,validators
import abc
import numpy as np
import itertools
import random
from typing import Dict,Tuple
import copy

from validators import validate_alpha, validate_gamma, validate_beta

@define
class Agent(metaclass=abc.ABCMeta):
    quality: float = field(default = 2)
    marginal_cost: float = field(default=1)

    @abc.abstractclassmethod
    def pick_strategy(self, state:np.array, action_space:np.array, t:int):
        raise NotImplementedError
        
    @abc.abstractclassmethod
    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        action:float
            ):

        raise NotImplementedError

    def get_name(self):
        return type(self).__name__


@define
class QLearning(Agent):

    """ Q learning that stores Q in dictionary form"""

    Q: Dict = field(default=None)
    alpha: float = field(default = 0.1,validator=[validators.instance_of(float), validate_alpha]) #learning rate
    gamma: float = field(default = 0.95,validator=[validators.instance_of(float), validate_gamma]) # discount rate
    beta: float = field(default = 0.00001,validator=[validators.instance_of(float), validate_beta]) #exploration parameter
    
    stable_status: bool = field(default=False)

    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        if not self.Q:
            self.Q = self.initQ(len(state), action_space)

        if np.exp(- self.beta * t) > np.random.rand():
            return random.choice(action_space)
        else:
            return self.exploit(state)
    
    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):

        old_action_value_array = copy.deepcopy(list(self.Q[tuple(curr_state)].values()))
        old_action_value = self.Q[tuple(curr_state)][action]
        new_action_value = self.Q[tuple(new_state)][self.exploit(new_state)]
        self.Q[tuple(curr_state)][action] = (1-self.alpha) * old_action_value +  self.alpha * (reward + self.gamma * new_action_value )
        
        #check convergence
        new_action_value_array = list(self.Q[tuple(curr_state)].values())
        self.stable_status = (np.argmax(old_action_value_array) == np.argmax(new_action_value_array))
    


    def exploit(self, state: np.array) -> float:

        optimal_actions = [
            action for action, value in self.Q[tuple(state)].items() if value == max(self.Q[tuple(state)].values())
        ]
        return random.choice(optimal_actions) 
    
    
    def get_parameters(self) -> str:
        return ": quality={}, mc={}, alpha={},beta={}, gamma={} " .format(
            self.quality, self.marginal_cost, self.alpha, self.beta, self.gamma
        )

    @staticmethod
    def initQ(num_agents:int, action_space:np.array) -> Dict:
        Q = {}
        for state in itertools.product(action_space, repeat=num_agents):
            Q[state] = dict((price,0) for price in action_space)
        return Q
    
    
    
    
    

    
    
    
    
    
    
    
    
@define
class QLearning2(Agent):
    """ Q learning that stores Q in a martrix form"""

    Q: np.array = field(default = None)
    alpha: float = field(default = 0.1,validator=[validators.instance_of(float), validate_alpha]) #learning rate
    gamma: float = field(default = 0.95,validator=[validators.instance_of(float), validate_gamma]) #discount rate
    beta: float = field(default = 0.001,validator=[validators.instance_of(float), validate_beta])  # exploration decay rate

    state_dim: Tuple = field(default = (None,))
    n_action_space: int = field(default = 0)
    
    stable_status: bool = field(default=False)

    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> int:
        if self.Q is None:
            self.Q = self.initQ(len(state), action_space)

        if np.exp(- self.beta * t) > np.random.rand():
            return random.choice(action_space)
        else:
            return self.exploit(state, action_space)

    def exploit(self, state: np.array, action_space:np.array) -> float:
        state_index = tuple(np.squeeze([np.where(action_space == i) for i in state]))
        optimal_actions = np.where(self.Q[state_index] == max(self.Q[state_index]))[0]
        return action_space[random.choice(optimal_actions)]
        #return action_space[np.argmax(self.Q[state_index])]   #cant do this because most state all the state action value will be 0 so when you argmax, the value you get is the first largest value which is 0

    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):

        curr_state_index = tuple(np.squeeze([np.where(action_space == i) for i in curr_state]))
        new_state_index = tuple(np.squeeze([np.where(action_space == i) for i in new_state]))
        action_index = np.squeeze(np.where(action_space == action))
        new_action_index = np.squeeze(np.where(action_space == self.exploit(new_state,action_space)))
        old_action_value_array = copy.deepcopy(self.Q[curr_state_index])
        
        old_action_value = copy.deepcopy(self.Q[curr_state_index][action_index])
        new_action_value = copy.deepcopy(self.Q[new_state_index][new_action_index])
        self.Q[curr_state_index][action_index] = (1-self.alpha) * old_action_value +  self.alpha * (reward + self.gamma * new_action_value )
        
        #check convergence
        new_action_value_array = self.Q[curr_state_index]
        self.stable_status = np.argmax(new_action_value_array) == np.argmax(old_action_value_array)
    

    def initQ(self,num_agents:int, action_space:np.array) -> np.array:
        self.state_dim = tuple(len(action_space) for _ in range(num_agents))
        self.n_action_space = len(action_space)
        Q = np.zeros(self.state_dim + (self.n_action_space,))
        return Q
    
    def get_parameters(self) -> str:
        return ": quality={}, mc={}, alpha={}, gamma={} ,beta={}" .format(
            self.quality, self.marginal_cost, self.alpha, self.gamma, self.beta
        )




@define
class SARSA(QLearning):

    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):
        
        old_action_value_array = copy.deepcopy(list(self.Q[tuple(curr_state)].values()))
        old_action_value = self.Q[tuple(old_state)][prev_action]
        curr_action_value = self.Q[tuple(curr_state)][action]
        self.Q[tuple(old_state)][prev_action] = (1-self.alpha) * old_action_value +  self.alpha * (reward + self.gamma * curr_action_value )

        #check convergence
        new_action_value_array = copy.deepcopy(list(self.Q[tuple(curr_state)].values()))
        self.stable_status = np.argmax(old_action_value_array) == np.argmax(new_action_value_array)
        
        
        
