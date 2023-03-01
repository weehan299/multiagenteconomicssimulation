from attr import define,field
import numpy as np
import math
from typing import Tuple,List


@define
class History:
    """
    Stores data history of all firms
    """
    limit: int = field(default = 300000)
    #dont write default=[] if not attributes will not be reinitialised everytime you run. 
    Q_history: List[List] = field(factory = list)  
    price_history: List[List] = field(factory = list)
    state_history: List[List] = field(factory = list)
    quantity_history: List[List] = field(factory = list)
    reward_history:  List[List] = field(factory = list)
    prob_weights_history: List[List] = field(factory = list)
    
    
    def add_to_price_history(self, price_array: np.array) -> List:
        if len(self.price_history) < self.limit:
            self.price_history.append(price_array)
        else:
            self.price_history.pop(0)
            self.price_history.append(price_array)

    def add_to_reward_history(self, reward_array: np.array) -> List:
        if len(self.reward_history) < self.limit:
            self.reward_history.append(reward_array)
        else:
            self.reward_history.pop(0)
            self.reward_history.append(reward_array)
