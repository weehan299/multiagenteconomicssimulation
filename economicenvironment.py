from attrs import define, field, validators
from typing import List
import numpy as np
import random
from tqdm import tqdm


from pricecompute import PriceCompute
from demand import Demand
from agents.agent import Agent
from validators import validate_total_periods,validate_action_space_num,validate_xi



@define
class EconomicEnvironment:
    """ Calculates demand

    Attributes
    ----------
    agents : List[Agent]
        List of Agent which can be made up of different strategies
    demand : Demand
        calculates multinomial logit demand based on prices and quality of firm

    marginal_cost_array: list
        marginal costs of each firm
    quality_array : list
        quality of each firm
    competitive_prices_array : list
        computes the Nash Price of each firm (same for every firm)
    monopoly_prices_array : list
        computes the Monopoly Price of each firm (same for every firm)

    action_space : list
        set of pricing actions that firms can take ranging from Nash Price - xi to Monopoly Price + xi
    action_space_num : int
        specify the number of elements in the action space (default = 15)

    total_periods : int
        number of period that the simulation will run for (default = 1000000)
    
    tstable : int
        specify number of period for convergence criteria (default = 100000)
    tscore : int
        to track algorithm reached convergence
    

    price_history : list 
        record prices played by agents for data analysis purpose
    quantity_history : list
        record quantity demand of agents' product for data analysis purpose
    reward_history : list = field(factory=list)
        record rewards earned by agents for data analysis purpose
    """
    agents: List[Agent] = field(factory=list)

    demand: Demand = field(factory=Demand)
    action_space: np.array = field(init=False)
    marginal_cost_array: np.array = field(init=False)
    quality_array: np.array = field(init=False)
    competitive_prices_array: np.array = field(init=False)
    monopoly_prices_array: np.array = field(init=False)

    action_space_num: int = field(default = 15, validator=[validators.instance_of(int), validate_action_space_num] )
    total_periods: int = field(default=1000000, validator=[validators.instance_of(int), validate_total_periods])
    xi: float = field(default=0.1, validator=[validators.instance_of(float), validate_xi])

    tstable:int = 200000
    tscore:int = 0
    

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

        for t in tqdm(range(self.total_periods)):
            
            next_state = np.array(
                [agent.pick_strategy(curr_state, self.action_space, t) for agent in self.agents]
            )

            """
            #used for artificially introducing low price point to determine behaviour of firm after convergence
            if self.tscore == self.tstable:
                print("Converged after {} period!".format(t))
                print("original:", next_state)
                next_state = np.array(
                   [self.action_space[0]] +  [agent.pick_strategy(curr_state, self.action_space, t) for agent in self.agents[1:]]
                )
                print("new:", next_state)
            
            if self.tscore == self.tstable+50:
                break
            """

            if self.tscore == self.tstable:
                break

            if self.tscore >= 199990:
                print(agent.Q for agent in self.agents)


            new_quantity_array = self.demand.get_quantity_demand(next_state, self.quality_array)
            reward_array = (next_state - self.marginal_cost_array) * new_quantity_array

            for agent, action, prev_action, reward, prev_reward in zip(
                self.agents, next_state, curr_state, reward_array, prev_reward_array):


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

            #check for convergence
            self.check_stable()


    
    def run_simulation_dont_provide_other_players_info(self):

        prev_state_of_all_agents = random.choices(self.action_space, k=len(self.agents))
        curr_state_of_all_agents = np.array(
            [self.agents[i].pick_strategy(np.array([prev_state_of_all_agents[i]]), self.action_space, 0) for i in range(len(self.agents))]
        )

        quantity_array = self.demand.get_quantity_demand(curr_state_of_all_agents, self.quality_array)
        prev_reward_array = (curr_state_of_all_agents - self.marginal_cost_array) * quantity_array

        for t in tqdm(range(self.total_periods)):
            
            next_state_of_all_agents = np.array(
                [self.agents[i].pick_strategy(np.array([curr_state_of_all_agents[i]]), self.action_space, t) for i in range(len(self.agents))]
            )

            if self.tscore == self.tstable:
                break


            new_quantity_array = self.demand.get_quantity_demand(next_state_of_all_agents, self.quality_array)
            reward_array = (next_state_of_all_agents - self.marginal_cost_array) * new_quantity_array

            for agent, prev_state, next_state, curr_state, reward, prev_reward in zip(
                self.agents, prev_state_of_all_agents, next_state_of_all_agents, curr_state_of_all_agents, reward_array, prev_reward_array):

                agent.learn(
                    old_state = np.array([prev_state]),
                    curr_state = np.array([curr_state]),
                    new_state = np.array([next_state]),
                    action_space= self.action_space,
                    prev_reward = prev_reward,
                    reward= reward,
                    prev_action = curr_state,
                    action = next_state,
                        )


            prev_state_of_all_agents = curr_state_of_all_agents
            curr_state_of_all_agents = next_state_of_all_agents
            prev_reward_array = reward_array
            

            self.price_history.append(prev_state_of_all_agents)
            self.quantity_history.append(quantity_array)
            self.reward_history.append(reward_array)

            #check for convergence
            self.check_stable()

    def check_stable(self):
            if False in [agent.stable_status == 1 for agent in self.agents]:
                self.tscore = 0
            else:
                self.tscore += 1


    def check_stable2(self):
            if False in [abs(agent.old_action_value - agent.curr_action_value) < 1e-6 for agent in self.agents]:
                self.tscore = 0
            else:
                self.tscore += 1


        
                
            