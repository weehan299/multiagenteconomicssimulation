from agent import QLearning, QLearning2, SARSA
from economicenvironment import EconomicEnvironment
from showresults import Results



def run(*args, **kwargs):

    agent1 = QLearning(alpha = kwargs.get("alpha",0.125), beta = kwargs.get("beta", 1e-05), gamma = kwargs.get("gamma", 0.95))
    agent2 = QLearning2(alpha = kwargs.get("alpha",0.125), beta = kwargs.get("beta", 1e-05), gamma = kwargs.get("gamma", 0.95))
    agent3 = SARSA()
    
    #num_agent = kwargs.get("num_agent",2)
    #agents = []
    #for i in range(num_agent):
    #agents.append(QLearning(alpha = kwargs.get("alpha",0.125), beta = kwargs.get("beta", 1e-05), gamma = kwargs.get("gamma", 0.95)))

    env = EconomicEnvironment(
        total_periods=kwargs.get("total_periods",1000000),
        action_space_num=kwargs.get("action_space_num",15),
        agents= [agent1,agent2]
    )
    env.run_simulation()

    results = Results(env)
    return results

if __name__ == "__main__":
    results = run(num_agent=2)
    results.print_results()