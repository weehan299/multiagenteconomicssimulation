from demand import Demand
from reward import MixedStrategyPayoff
import numpy as np

demand1 = Demand()
reward = MixedStrategyPayoff()

#action_space =np.array([1.47292183,1.92498094])

action_space =np.array([1.5,1.6])
action_nash = np.array([action_space[0],action_space[0]])
action_mono = np.array([action_space[1],action_space[1]])
action_both =np.array([action_space[0],action_space[1]])
action_both_reverse =np.array([action_space[1],action_space[0]])

prob_weights1 = [0.4,0.6]
prob_weights2 = [0.4,0.6]
prob_weights_array = np.array([prob_weights1,prob_weights2])

marginal_cost_array = np.array([1,1])
quality_array = np.array([2,2])

quantity_array_nash = demand1.get_quantity_demand(action_nash,quality_array)
quantity_array_mono = demand1.get_quantity_demand(action_mono,quality_array)
quantity_array_both = demand1.get_quantity_demand(action_both,quality_array)
quantity_array_both_reverse = demand1.get_quantity_demand(action_both_reverse,quality_array)

print("quantity_array_nash: ",quantity_array_nash) #[0.47137733 0.47137733]
print("quantity_array_mono: ",quantity_array_mono) #[0.7779766  0.12754382]
print("quantity_array_both: ",quantity_array_both) #[0.7779766  0.12754382]

print(quantity_array_nash)

avg_reward = prob_weights1[0]*prob_weights2[0]* (action_nash - marginal_cost_array) * quantity_array_nash +  \
    prob_weights1[1]*prob_weights2[1]* (action_mono - marginal_cost_array) * quantity_array_mono +  \
    prob_weights1[0]*prob_weights2[1]* (action_both - marginal_cost_array) * quantity_array_both +  \
    prob_weights1[1]*prob_weights2[0]* (action_both_reverse - marginal_cost_array) * quantity_array_both_reverse
print("test avg reward: ", avg_reward)

#P + T
cond_reward_1_nash = prob_weights2[0]* (action_nash[0] - marginal_cost_array[0]) * quantity_array_nash[0] +  \
    prob_weights2[1]* (action_both[0] - marginal_cost_array[0]) * quantity_array_both[0]

# S + R
cond_reward_1_mono = prob_weights2[0]* (action_both_reverse[0] - marginal_cost_array[0]) * quantity_array_both_reverse[0] +  \
    prob_weights2[1]* (action_mono[0] - marginal_cost_array[0]) * quantity_array_mono[0]
    

cond_reward_2_nash = prob_weights1[0]* (action_nash[1] - marginal_cost_array[1]) * quantity_array_nash[1] +  \
    prob_weights1[1]* (action_both_reverse[1] - marginal_cost_array[1]) * quantity_array_both_reverse[1]

cond_reward_2_mono = prob_weights1[0]* (action_both[1] - marginal_cost_array[1]) * quantity_array_both[1] +  \
    prob_weights1[1]* (action_mono[1] - marginal_cost_array[1]) * quantity_array_mono[1]

#T = 0.36792212
T = ((action_both - marginal_cost_array) * quantity_array_both)[0]
#R = 0.33749046
R = ((action_mono - marginal_cost_array) * quantity_array_mono)[0]
#P =  0.22292463
P = ((action_nash - marginal_cost_array) * quantity_array_nash)[0]
#S = 0.1179756
S = ((action_both - marginal_cost_array) * quantity_array_both)[1]


print("T,R,P,S: ", T,R,P,S)
print("T+P, R+S: ", T+P, R+S)
print("2R, T+S ", 2*R, T+S)



average_reward_array = reward.get_average_reward(marginal_cost_array,quality_array, action_space, prob_weights_array)
conditional_reward_array = reward.get_conditional_reward(marginal_cost_array, quality_array, action_space, prob_weights_array)
print("########## agents conditional prices ###############")
#print(cond_reward_1_nash,cond_reward_1_mono)
#print(cond_reward_2_nash,cond_reward_2_mono)

print("hahahah",prob_weights_array * (conditional_reward_array - average_reward_array))
print("koookok", prob_weights1[0]*prob_weights1[1] * (prob_weights1[1] * T + prob_weights1[0] * P - prob_weights1[1]*R - prob_weights1[0]*S))

print("average reward: ", average_reward_array, "conditional reward: ", conditional_reward_array)
print("prob * (conditional reward - average reward): ", prob_weights_array * (conditional_reward_array - average_reward_array))
#print((cond_reward_1_nash - average_reward_array[0]) *prob_weights1[0])


"""
print("############################################### \n")


marginal_cost_array = np.array([1,1])
quality_array = np.array([2,2])
action_space =np.array([1.47292183,1.92498094, 2])
prob_weights1 = [0.5,0.3,0.2]
prob_weights2 = [0.2,0.4,0.3]
prob_weights_array = np.array([prob_weights1,prob_weights2])

average_reward_array = reward.get_average_reward(marginal_cost_array,quality_array, action_space, prob_weights_array)
conditional_reward_array = reward.get_conditional_reward(marginal_cost_array, quality_array, action_space, prob_weights_array)


print("average reward: ", average_reward_array, "conditional reward: ", conditional_reward_array)

print(sum(prob_weights1*conditional_reward_array[0]), sum(prob_weights2*conditional_reward_array[1])) # this should equal average reward
"""