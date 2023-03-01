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
            return [1/len(Q_value_array) for i in Q_value_array]
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
    lambda0: float = field(default = 1000)
    lambda1: float = field(default = 0.999)


    def give_prob_weights_for_each_action(self, Q_value_array:List, t:int) -> List:
        T = 16*self.lambda0*(t+1)**(-self.lambda1)
        exponent = np.true_divide(Q_value_array - np.max(Q_value_array), T)  #prevent numerical overflow
        return np.exp(exponent) / np.sum(np.exp(exponent))

    def get_name(self):
        return "(" + type(self).__name__+ ": lambda0 = " + str(self.lambda0) + ", lambda1 = "+ str(self.lambda1) +")"
        
# constant change to 16 for regular q learning with action space size = 15
