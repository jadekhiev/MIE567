# run multiple trials and plot results
# Imports
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math
import Flags as fw
import TdLambda as td

flags = Flags()
alpha = 0.3
gamma = 0.99
epsilon_rng = [0.01, 0.1, 0.25]
lamda = 0.9

episode = list(range(0,500)) # list from 1-500
trial = list(range(0,20)) # list from 1-20
G0_vals = {}
Q_vals = {}

for epsilon in epsilon_rng:
    returns = pd.DataFrame(index=trial, columns=episode)
    Q = pd.DataFrame()
    for i in tqdm(range(10)):
        td_lambda = td.TdLambda(flags, alpha, gamma, epsilon, lamda)
        progress = td_lambda.train_trial()
        returns.iloc[i] = progress
        if i==0: 
            Q = pd.DataFrame.from_dict(td_lambda.Q, orient='index')
        else: 
            Q = pd.concat([Q,pd.DataFrame.from_dict(td_lambda.Q, orient='index')])
            Q = Q.groupby(Q.index).mean()
    G0_vals[epsilon] = returns
    Q_vals[epsilon] = Q

def initialize_plot():
    plt.plot()
    plt.ylabel('G_0 Mean')
    plt.xlabel('Episode')
    plt.title('TD(Lambda) with Changing Epsilon [Average Mean over 10 Trials]')
    plt.rcParams['figure.dpi'] = 150
    return plt

plt = initialize_plot()

for epsilon in epsilon_rng:
    G0_mean = G0_vals[epsilon].mean()
    G0_std = G0_vals[epsilon].std()
    plt.plot(episode, G0_mean, label = epsilon)
    
plt.legend()
plt.show(block=True)

def initialize_plot():
    plt.plot()
    plt.ylabel('G_0 StdDev')
    plt.xlabel('Episode')
    plt.title('TD(Lambda) with Changing Epsilon [Average SD over 10 Trials]')
    plt.rcParams['figure.dpi'] = 150
    return plt

plt = initialize_plot()

for epsilon in epsilon_rng:
    G0_mean =  G0_vals[epsilon].mean()
    G0_std = G0_vals[epsilon].std()
#    plt.errorbar(episode, G0_mean, G0_std, label = epsilon, markersize=1)
    plt.scatter(episode, G0_std, label = epsilon, s=2)

plt.legend()
plt.show(block=True)