from agents.agent import Binary_State_QLearning, ConstantPricer, QLearning, QLearning2, SARSA, QLearningWithMemory,TitforTat, Mixed_Strategy_Binary_State_QLearning
from economicenvironment import EconomicEnvironment
from policy import Boltzmann, TimeDecliningExploration
from reward import ExpectedReward, StandardReward, MixedStrategyPayoff
from showresults import Results
from demand import Demand


def run(*args, **kwargs):

    agent1 = QLearning(alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95), marginal_cost=1,
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3), temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
    agent2 = QLearning(alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95), marginal_cost=0.25,
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3),temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))

    memory_agent1 = QLearningWithMemory(memory_length= 1, alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                       # policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))

    memory_agent2 = QLearningWithMemory(memory_length= 1, alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))

    memory_agent3 = QLearningWithMemory(memory_length= 1, alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
    
    memory_agent4 = QLearningWithMemory(memory_length= 1, alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))

    binary_state_agent1 = Binary_State_QLearning(memory_length= 1, alpha = kwargs.get("alpha",0.2), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
    
    binary_state_agent2 = Binary_State_QLearning(memory_length= 1, alpha = kwargs.get("alpha",0.2), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))

    mixed_strategy_agent1 = Mixed_Strategy_Binary_State_QLearning(memory_length= 0, alpha = kwargs.get("alpha",0.2), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
    
    mixed_strategy_agent2 = Mixed_Strategy_Binary_State_QLearning(memory_length= 0, alpha = kwargs.get("alpha",0.2), gamma = kwargs.get("gamma", 0.95),
                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999)))
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05)))
    
    constant_agent = ConstantPricer()
    titfortat_agent = TitforTat()

    
    num_agents = kwargs.get("num_agents",2)
    #agents = [mixed_strategy_agent1, mixed_strategy_agent2]
    #agents = [binary_state_agent1, binary_state_agent2]
    #agents = [memory_agent1,memory_agent2]
    #agents = [agent1,agent2]
    agents = []
    
    """
    # for variations in costs
    mc=kwargs.get("marginal_costs",[1,0.25])
    for i in range(len(mc)):
        agents.append(QLearning(alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95), marginal_cost= mc[i],
                        #policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999))))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05))))
    """

    for i in range(num_agents):
        agents.append(QLearning(alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        #policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999))))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05))))
        #agents.append(QLearningWithMemory(memory_length= kwargs.get("memory_length",1), alpha = kwargs.get("alpha",0.125), gamma = kwargs.get("gamma", 0.95),
                        #policy=TimeDecliningExploration(beta = kwargs.get("beta", 1e-05))))
        #                        policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999))))
        #agents.append(Mixed_Strategy_Binary_State_QLearning(memory_length= 0, alpha = kwargs.get("alpha",0.2), gamma = kwargs.get("gamma", 0.95),
                        #policy=Boltzmann(lambda0=kwargs.get("lambda0",1000), lambda1=kwargs.get("lambda1",0.999))))

    env = EconomicEnvironment(
        total_periods=kwargs.get("total_periods",3000000),
        action_space_num=kwargs.get("action_space_num",15),
        agents= agents,
        reward= StandardReward(),
        #reward=MixedStrategyPayoff(),
        xi=kwargs.get("xi",0.1),
        demand= Demand(mu=kwargs.get("mu",0.25))
    )

    #env.run_simulation_dont_provide_other_players_info()
    env.run_simulation()
    #env.run_simulation_for_mixed_strategy_agents()

    results = Results(env=env)
    return results

if __name__ == "__main__":
    results = run(num_agents=2)
    results.print_results()
