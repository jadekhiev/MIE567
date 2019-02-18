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
    plt.title("Value Iteration with Changing Windspeed in Windy Gridworld")    
    return plt

# run value iteration code here
gridworld = gw.Gridworld()
init_vals= {}
plt = initialize_plot()
gamma = 0.9
finalPolicyDf = pd.DataFrame(index=gridworld.states(), columns=gammaRange)

pRange = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for p in pRange:
    value = vi.ValueIteration(gridworld, gamma)
    value.set_p(p)                              #add wind (stochasticity)
    value.value_iteration()
    
    #create plot
    V = value.get_V()
    init_vals[p] = [V[v][(5,1)] for v in V]
    plt.plot(init_vals[p], label = p)
    #get final policy for each gamma
    finalPolicyDf[gamma]=pd.DataFrame.from_dict(value.get_pi(), orient='index')[0]
    
plt.legend()
plt.show()

print(finalPolicyDf)