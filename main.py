from agents.agent import ConstantPricer, QLearning, QLearning2, SARSA, QLearningWithMemory
from economicenvironment import EconomicEnvironment
from policy import Boltzmann, TimeDecliningExploration
from showresults import Results



def run(*args, **kwargs):

    agent1 = QLearning(alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3), temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
    agent2 = QLearning(alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95), 
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3),temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
    agent3 = ConstantPricer()

    agent4 = QLearningWithMemory(memory_length= 1, alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3), temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))

    agent5 = QLearningWithMemory(memory_length= 1, alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3), temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))


    # agent3 = SARSA()
    
    num_agent = kwargs.get("num_agent",2)
    agents = [agent1, agent2]
    #for i in range(num_agent):
        #agents.append(QLearning(alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95), 
                                #policy=Boltzmann(temp_max=kwargs.get("temp_max",1), temp_min=kwargs.get("temp_min",0.01),
                                                 #tot_steps = kwargs.get("total_periods", 1000000))))
    env = EconomicEnvironment(
        total_periods=kwargs.get("total_periods",1000000),
        action_space_num=kwargs.get("action_space_num",15),
        agents= agents
    )
    env.run_simulation_dont_provide_other_players_info()

    results = Results(env)
    return results

if __name__ == "__main__":
    results = run(num_agent=2)
    results.print_results()