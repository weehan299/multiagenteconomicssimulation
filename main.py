########## economic environment ##########
#agents
#action space
#nash and monopoly prices
#state space
#demand


########## agent  ##########
# quality
# costs
# strategy
# price (something you choose from action space) method


########### Q learning ##############
#inherit from agent strategy
# Q matrix
# discount factor delta
# learning rate alpha
# decision choice: epsilongreedy


from agent import QLearning, QLearning2, SARSA
from economicenvironment import EconomicEnvironment
from showresults import results, get_agents_parameters

agent1 = QLearning()
agent2 = QLearning()
agent3 = SARSA()


def run(*args, **kwargs):
    env = EconomicEnvironment(
        total_periods=kwargs.get("total_periods",1000000),
        action_space_num=kwargs.get("action_space_num",15),
        agents=[
            agent1,
            agent2,
        ]
    )
    env.run_simulation()
    results(env)
    get_agents_parameters(env)
    return env

run()
