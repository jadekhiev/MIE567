# MIE567
Coursework for MIE567


## Scripts that do not need to be run:
### Part A
Gridworld.py - creates gridworld (and also windy gridworld)
PolicyIteration.py - creates policy iteration class
ValueIteration.py - creates value iteration class

## Scripts that do need to be run:
*Each script outputs a matplotlib graph _this needs to be closed before the final policy is output_*

### Part B
* RunPI.py - Deterministic Policy Iteration, varying Gamma [0.90,...,0.99] (0.01 increment)

### Part C
* RunVI.py - Deterministic Value Iteration, varying Gamma [0.90,...,0.99] (0.01 increment)

### Windy A
* N/A - Gridworld.py accepts a p value for windy gridworld, otherwise defaults to p=0 for deterministic gridworld

### Windy B
* RunWindyVI.py - Windy Policy Iteration, varying Gamma [0.90,...,0.99] (0.01 increment)
* RunWindyVI_wind.py - Windy Value Iteration, varying p [0.1,..., 0.9] (0.10 increment)

### Windy C
* RunWindyPI.py - Windy Policy Iteration, varying Gamma [0.90,...,0.99] (0.01 increment)
* RunWindyPI_wind.py - Windy Policy Iteration, varying p [0.1,..., 0.9] (0.10 increment)
