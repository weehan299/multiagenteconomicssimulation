from attrs import define, field,validators
import abc
import numpy as np
import random
from typing import Dict,Tuple,List
from validators import validate_beta
import copy


@define
class Policy(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def give_prob_weights_for_each_action(self, Q_value_array:List, t:int) -> List:
        raise NotImplementedError
        
    def get_name(self):
        return type(self).__name__

@define
class TimeDecliningExploration(Policy):
    beta: float = field(default = 0.00001,validator=[validators.instance_of(float), validate_beta]) # exploration parameter

    def give_prob_weights_for_each_action(self, Q_value_array:List, t:int) -> List:
        if np.exp(- self.beta * t) > np.random.rand():
            return [1 for i in Q_value_array]
        else:
            return self.exploit(Q_value_array)
    
    def exploit(self,Q_value_array: np.array) -> List:
        return [1 if q == max(Q_value_array) else 0 for q in Q_value_array]

    def get_name(self):
        return "(" + type(self).__name__+ ": beta = " + str(self.beta) + ")"

@define
class Boltzmann(Policy):
    #larger the temperature, the more it explores
    temperature_array: List = field(init=False)
    temp_max:float = field(default = 1)
    temp_min:float = field(default = 0.01)
    tot_steps: int = field(default = 1000000)

    def __attrs_post_init__(self):
        first_half =  np.linspace(self.temp_min, self.temp_max, num=self.tot_steps//2)
        second_half = [self.temp_min for number in range(self.tot_steps)]
       # self.temperature_array = np.linspace(self.temp_min, self.temp_max, num=self.tot_steps)
        self.temperature_array = np.concatenate([first_half,second_half])
        print(self.temperature_array)
    def give_prob_weights_for_each_action(self, Q_value_array:List, t:int) -> List:
        T = self.temperature_array[t]
        exponent = np.true_divide(Q_value_array - np.max(Q_value_array), T)
        return np.exp(exponent) / np.sum(np.exp(exponent))

    def get_name(self):
        return "(" + type(self).__name__+ ": temp_max = " + str(self.temp_max) + ", temp_min = "+ str(self.temp_min) +")"
        

