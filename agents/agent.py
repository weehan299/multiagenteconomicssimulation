
from attrs import define, field,validators
import abc
import numpy as np
import itertools
import random
from typing import Dict, List,Tuple
import copy

from validators import validate_alpha, validate_gamma, validate_beta
from policy import Policy, TimeDecliningExploration

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
class QLearningWithMemory(Agent):

    """ Q learning that stores Q in dictionary form"""
    # a agent state is the set of all past prices. 

    Q: Dict = field(default=None)
    policy: Policy = field(factory = TimeDecliningExploration)
    memory_length: int = field(default = 1)
    memory: list = field(default = None)

    old_action_value: float = field(init=False)
    curr_action_value: float= field(init=False)

    alpha: float = field(default = 0.1,validator=[validators.instance_of(float), validate_alpha]) #learning rate
    gamma: float = field(default = 0.95,validator=[validators.instance_of(float), validate_gamma]) # discount rate
    
    stable_status: bool = field(default=False)


    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        if not self.Q:
            self.memory = self.init_memory(len(state), self.memory_length,action_space)
            self.Q = self.init_Q(len(state), self.memory_length, action_space)
        
        state_with_memory = self.append_memory_to_state(state) 
        
        Q_value_array =list(self.Q[tuple(state_with_memory)].values())
        prob_weights = self.policy.give_prob_weights_for_each_action(Q_value_array,t)
        return random.choices(action_space,weights=prob_weights,k=1)[0]

    
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


        self.update_memory(old_state)
        curr_state_with_memory = self.append_memory_to_state(curr_state)
        new_state_with_memory = np.concatenate((new_state, curr_state))

    
        old_action_value_array = copy.deepcopy(list(self.Q[tuple(curr_state_with_memory)].values()))
        old_action_value = copy.deepcopy(self.Q[tuple(curr_state_with_memory)][action])
        new_action_value = self.Q[tuple(new_state_with_memory)][self.get_argmax(new_state_with_memory)]
        self.Q[tuple(curr_state_with_memory)][action] = (1-self.alpha) * old_action_value +  self.alpha * (reward + self.gamma * new_action_value )

        self.old_action_value = old_action_value
        self.curr_action_value = self.Q[tuple(curr_state_with_memory)][action]
        
        #check convergence
        new_action_value_array = list(self.Q[tuple(curr_state_with_memory)].values())
        #self.stable_status = 1 if all(np.absolute(np.subtract(old_action_value_array , new_action_value_array)) < 1e-6) else 0
        #self.stable_status = 1 if abs(old_action_value - self.Q[tuple(curr_state)][action]) < 1e-6 else 0
        self.stable_status = (np.argmax(old_action_value_array) == np.argmax(new_action_value_array))

    def append_memory_to_state(self, state:np.array) -> np.array:
        temp = copy.deepcopy(self.memory.flatten())
        result = np.concatenate((state,temp))
        return result
        
    def update_memory(self, curr_state:np.array):
        self.memory[1:] = self.memory[:-1]
        self.memory[0] = curr_state

    def get_argmax(self, state: np.array) -> float:
        optimal_actions = [
            action for action, value in self.Q[tuple(state)].items() if value == max(self.Q[tuple(state)].values())
        ]
        return random.choice(optimal_actions) 
    
    
    def get_parameters(self) -> str:
        return ": quality={}, mc={}, alpha={}, gamma={}, policy = {} " .format(
            self.quality, self.marginal_cost, self.alpha, self.gamma, self.policy.get_name()
        )

    @staticmethod
    def init_Q(num_agents:int, memory_length:int, action_space:np.array) -> Dict:
        Q = {}
        for state in itertools.product(action_space, repeat=num_agents*(memory_length+1)):
            Q[state] = dict((price,0) for price in action_space)
        return Q

    @staticmethod
    def init_memory(num_agents:int, memory_length:int,action_space:np.array) -> np.array:
        #initialise with random memory from the start
        memory = np.array([random.choices(action_space, k=num_agents) for i in range(memory_length)])
        print(memory)
        return memory
    
    
    
    
    
    
@define
class QLearning(Agent):

    """ Q learning that stores Q in dictionary form"""

    Q: Dict = field(default=None)
    policy: Policy = field(factory = TimeDecliningExploration)

    old_action_value: float = field(init=False)
    curr_action_value: float= field(init=False)

    alpha: float = field(default = 0.1,validator=[validators.instance_of(float), validate_alpha]) #learning rate
    gamma: float = field(default = 0.95,validator=[validators.instance_of(float), validate_gamma]) # discount rate
    
    stable_status: bool = field(default=False)


    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        if not self.Q:
            self.Q = self.initQ(len(state), action_space)

        Q_value_array =list(self.Q[tuple(state)].values())
        prob_weights = self.policy.give_prob_weights_for_each_action(Q_value_array,t)
        return random.choices(action_space,weights=prob_weights,k=1)[0]



    
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
        old_action_value = copy.deepcopy(self.Q[tuple(curr_state)][action])
        new_action_value = self.Q[tuple(new_state)][self.get_argmax(new_state)]
        self.Q[tuple(curr_state)][action] = (1-self.alpha) * old_action_value +  self.alpha * (reward + self.gamma * new_action_value )

        self.old_action_value = old_action_value
        self.curr_action_value = self.Q[tuple(curr_state)][action]
        
        #check convergence
        new_action_value_array = list(self.Q[tuple(curr_state)].values())
        #self.stable_status = 1 if all(np.absolute(np.subtract(old_action_value_array , new_action_value_array)) < 1e-6) else 0
        #self.stable_status = 1 if abs(old_action_value - self.Q[tuple(curr_state)][action]) < 1e-6 else 0
        self.stable_status = (np.argmax(old_action_value_array) == np.argmax(new_action_value_array))

    
    def get_argmax(self, state: np.array) -> float:
        optimal_actions = [
            action for action, value in self.Q[tuple(state)].items() if value == max(self.Q[tuple(state)].values())
        ]
        return random.choice(optimal_actions) 
    
    
    def get_parameters(self) -> str:
        return ": quality={}, mc={}, alpha={}, gamma={}, policy = {} " .format(
            self.quality, self.marginal_cost, self.alpha, self.gamma, self.policy.get_name()
        )

    @staticmethod
    def initQ(num_agents:int, action_space:np.array) -> Dict:
        Q = {}
        for state in itertools.product(action_space, repeat=num_agents):
            Q[state] = dict((price,0) for price in action_space)
        return Q
    
    
    
    
    
    
    
    
    
    
    
    
    
@define
class QLearning2(Agent):
    """ Q learning that stores Q in a matrix form"""

    Q: np.array = field(default = None)
    policy: Policy = field(factory = TimeDecliningExploration)

    alpha: float = field(default = 0.1,validator=[validators.instance_of(float), validate_alpha]) #learning rate
    gamma: float = field(default = 0.95,validator=[validators.instance_of(float), validate_gamma]) #discount rate

    state_dim: Tuple = field(default = (None,))
    n_action_space: int = field(default = 0)
    
    stable_status: bool = field(default=False)

    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        if self.Q is None:
            self.Q = self.initQ(len(state), action_space)

        state_index = tuple(np.squeeze([np.where(action_space == i) for i in state]))
        Q_value_array =self.Q[tuple(state_index)]
        prob_weights = self.policy.give_prob_weights_for_each_action(Q_value_array,t)
        #print(prob_weights)
        return random.choices(action_space,weights=prob_weights,k=1)[0]


    def get_argmax(self, state: np.array, action_space:np.array) -> float:
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
        new_action_index = np.squeeze(np.where(action_space == self.get_argmax(new_state,action_space)))
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
        return ": quality={}, mc={}, alpha={}, gamma={} " .format(
            self.quality, self.marginal_cost, self.alpha, self.gamma
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
        
        
        
        
@define
class ConstantPricer(Agent):
    stable_status: bool = field(default=True)
    constant_price: bool=field(init=False)
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
        pass

    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        self.constant_price = action_space[len(action_space)//2]
        return self.constant_price
    
    def get_parameters(self) -> str:
        return "Constant Price at: {}".format(round(self.constant_price,5))