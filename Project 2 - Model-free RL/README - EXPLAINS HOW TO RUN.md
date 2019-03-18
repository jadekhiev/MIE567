# MIE567
Coursework for MIE567

## Scripts that do not need to be run:
### Part A
Flags.py - creates Flags domain
MonteCarlo.py - creates First-Visit On-Policy MC class
QLearning.py - creates Q-Learning class
Sarsa.py - creates SARSA class
TdLambda.py - creates TD(Lambda) class

## Scripts that do need to be run:
*Each script outputs a matplotlib graph _this needs to be closed before the final policy for epsilon = 0.25 is output_*

### RunMC.py
*Runs the first-visit on-policy Q-Learning algorithm as discussed in class.* 
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25] 

### RunQLearning.py
*Runs the off-policy Q-Learning algorithm as discussed in class.* 
* Learning rate, alpha = 0.3
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25] 

### RunSarsa.py
*Runs the on-policy one-step one-step Sarsa algorithm as discussed in class.* 
* Learning rate, alpha = 0.3
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25] 

### RunTdLambda.py
*Runs the on-policy TD(lambda) algorithm using traces as discussed in class.*
* Learning rate, alpha = 0.3 
* Trace parameter, lambda = 0.9 
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25]

## Multiple Trial Scripts:
* Same as above scripts but plots G_0 returns for multiple trials (specified in script name)
* Average Q-Values were also extracted from these scripts. The output can be found in csv files under the folder: Q-Values

### RunMC_10Trials.py
*Runs the first-visit on-policy Q-Learning algorithm as discussed in class.* 
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25] 

### RunQLearning_20Trials.py
*Runs the off-policy Q-Learning algorithm as discussed in class.* 
* Learning rate, alpha = 0.3
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25] 

### RunSarsa_20Trials.py
*Runs the on-policy one-step one-step Sarsa algorithm as discussed in class.* 
* Learning rate, alpha = 0.3
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25] 

### RunTdLambda_10Trials.py
*Runs the on-policy TD(lambda) algorithm using traces as discussed in class.*
* Learning rate, alpha = 0.3 
* Trace parameter, lambda = 0.9 
* Discount factor, gamma = 0.99 
* Experiments with epsilon in [0.01, 0.1, 0.25]