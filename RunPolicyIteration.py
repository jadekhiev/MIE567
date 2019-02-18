import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import Gridworld as gw
import ValueIteration as vi
import PolicyIteration as pi

def initialize_plot():
    plt.plot()
    plt.ylabel('Value at Initial State (5,1)')
    plt.xlabel('Iteration')
    plt.title("Policy Iteration with Changing Gamma")
    return plt

# run policy iteration code here
gridworld = gw.Gridworld()
init_vals= {}
plt = initialize_plot()
gammaRange = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
finalPolicyDf = pd.DataFrame(index=gridworld.states(), columns=gammaRange)

for gamma in gammaRange:
    policy = pi.PolicyIteration(gridworld, gamma)
    policy.policy_iteration()

    #create plot
    V = policy.get_V()
    init_vals[gamma] = [V[v][(5,1)] for v in V]
    plt.plot(init_vals[gamma], label = gamma)
    
    #get final policy for each gamma
    finalPolicyDf[gamma]=pd.DataFrame.from_dict(value.get_pi(), orient='index')[0]

plt.legend()
plt.show()

print(finalPolicyDf)