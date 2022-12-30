from attr import define,field
import numpy as np
import math
from typing import Tuple,List


@define
class History:
    """
    Stores data history of all firms
    """
    #dont write default=[] if not attributes will not be reinitialised everytime you run. 
    Q_history: List[List] = field(factory = list)  
    price_history: List[List] = field(factory = list)
    state_history: List[List] = field(factory = list)
    quantity_history: List[List] = field(factory = list)
    reward_history:  List[List] = field(factory = list)
    prob_weights_history: List[List] = field(factory = list)