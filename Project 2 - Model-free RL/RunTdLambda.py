# Imports
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math

# For each algorithm you implement, you should have another Python file that allows to
# run your experiment. Here, you will have to initialize your domain, and the algorithm
# with your parameter settings, and run your algorithm.

def initialize_plot():
    plt.plot()
    plt.ylabel('G_0')
    plt.xlabel('Iteration')
    plt.title("TD(Lambda) with Changing Epsilon")
    return plt

alpha = 0.3
gamma = 0.99
epsilon_rng = [0.01, 0.1, 0.25]
lamda = 0.9

flags = Flags()
Q0_vals= {}
plt = initialize_plot()

for epsilon in epsilon_rng:
    td_lambda = TdLambda(flags, alpha, gamma, epsilon, lamda)
    progress = td_lambda.train_trial()
    Q0_vals[epsilon] = progress
    plt.plot(Q0_vals[epsilon], label = epsilon)

plt.legend()
plt.show(block=True)

td_lambda.test_greedy()