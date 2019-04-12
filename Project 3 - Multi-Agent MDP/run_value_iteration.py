from scipy.stats.kde import gaussian_kde
import InvaderDefender as id
import ValueIteration as vi
import matplotlib.pyplot as plt
import numpy as np

def plot_map(self):
# A procedure for plotting the final value function(s) as a 2D heatmap.
# Here, you need to create two plots: 
#    1) The first plot shows the final value function from the invader's 
#    perspective as a function of the invader's starting position, 
#    when the defender starting position is fixed at x = 0; y = 5. 
#    2) Similarly, the second plot shows the final value function from the defender's
#    perspective as a function of the defender's starting position when the invader's 
#    starting position is fixed at x = 0; y = 0.
    print("need to figure out how to heatmap")
    
def heatmap(playerstr, U):
    gridstates = [
            (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),
            (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
            (3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
            (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),
            (5,1),(5,2),(5,3),(5,4),(5,5),(5,6),
            (6,1),(6,2),(6,3),(6,4),(6,5),(6,6)
            ]
    if playerstr == 'invader':
        player = 0
    else:            # defender
        player = 1
    
    tensordict = {}

    for gridstate in gridstates:
        cell = []
        for x,y in U.items():
            if x[player] == gridstate:
                cell.append(y)
        cell = np.average(cell)
        tensordict[gridstate] = cell
    
    twodim = np.zeros([6,6])

    for x, y in tensordict.items():
        twodim[x[0]-1][x[1]-1] = y
            
    plt.imshow(twodim, cmap='RdPu', interpolation='nearest')
    plt.show()
    return twodim

def initialize_plot(title):
    plt.plot()
    plt.ylabel('Row')
    plt.xlabel('Column')
    plt.title(title)    
    return plt

#def run_value_iteration():
# A procedure for the main loop of the value iteration algorithm. 
# This procedure should be responsible for initializing the value function(s), 
# performing the iterations to update the value function(s), and monitoring convergence. 
# Your algorithm should stop as soon as the delta, 
# the maximum absolute change in the value function between iteration k - 1 and 
# iteration k is less than 1e-6. At the end of this procedure, 
# you should plot the delta value at each iteration as a line plot. 
# You should also plot the final value functions as heatmaps as described in the previous bullet point. 
# You should run your algorithm by creating a python file called run_value_iteration.py that runs your experiment. You may use gamma = 0:95.

# run value iteration code here
invaderdefender = id.InvaderDefender()
init_vals= {}
plt = initialize_plot("Invader - Defender Heatmap")
#gammaRange = [0.95]
#finalPolicyDf = pd.DataFrame(index=gridworld.states(), columns=gammaRange)

gamma = 0.95

value = vi.ValueIteration(invaderdefender, gamma)
U, pi_p, pi_q, delta = value.value_iteration()
print("defender")
heatmap("defender", U)
print("invader")
heatmap("invader", U)

plt = initialize_plot("delta")
plt.plot(delta)
plt.show()
    
#    return U, pi_p, pi_q

