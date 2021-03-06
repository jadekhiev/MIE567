import numpy as np
import random
import math

class MonteCarlo:
    def __init__(self, domain, gamma, epsilon):
    # sets the domain instance, and all relevant parameters for each algorithm 
    # (e.g. learning rate, epsilon, etc.).
        self.domain = domain
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Added this feature to represent whether this instance has been run, if so, print out the good policy
        self.final = False
        self.policy = []

    def initialize_values(self):
    # A procedure called initialize values that initializes the Q-values to zero for all
    # states in the MDP and all relevant other counters and memory.
        self.Q = {}
        self.Returns = {}
        for state in self.domain.get_all_states():
            self.Q[state]={}
            self.Returns[state]={}
            for action in self.domain.actions():
                self.Q[state][action]=0
                self.Returns[state][action]=[]
    
    def sample_greedy(self, state):
        # if more than one action have the max value, select randomly from those top actions
        top = max(self.Q[state].values())
        if math.isnan(top): print(state)
        topmoves = [k for k, v in self.Q[state].items() if v == top]
        greedy_a = random.choice(topmoves)
        return greedy_a        
    
    def sample_epsilon_greedy(self):
    # takes a state of the MDP as argument and returns an action sampled 
    # according to the epsilon-greedy policy defined by the epsilon parameter 
    # set in the init method and the Q-values.
        choice = np.random.choice(['policy', 'rand'], p = [(1-self.epsilon), self.epsilon])
        if choice == 'policy':
            action = self.sample_greedy(self.state)
        else:
            action = random.choice(self.domain.actions())                
        return action
    
    def test_greedy(self): 
    # generates an episode from the MDP using the greedy policy 
    # (we are testing, so please do NOT use the epsilon-greedy policy for this!) 
    # with respect to the current Q-values, and returns G0, the discounted return 
    # starting from the initial state (please refer to your lecture or tutorial notes). 
    # This function will be used to inspect the rate at which your algorithm 
    # converges to a good policy as it trains.
        self.state = self.domain.initial_state()
        terminal = False
        R = []
        i = 0
        
        # run the episode to collect data
        while not terminal and i < 200:
            S = self.state
            a = self.sample_greedy(S)
            
            # Added in a feature where it will print the policy out once the algo has been run
            print('state: ' + str(S) + ', action: ' + a) if self.final else ""
            S_next, reward = self.domain.transition(S, a)
            R.append(reward)
            self.state = S_next
            terminal = self.domain.is_terminal(self.state)
            i += 1

        # calculate G_0
        G_0 = 0        
        for i in range(len(R)):
            G_0 = G_0 + R[i]*pow(self.gamma, i)
        return G_0    
    
    def train_episode(self):
    # Each algorithm has a main training loop that generates an episode from the MDP
    # according to the epsilon-greedy policy and uses this to update the Q-values. 
    # How you implement this method is up to you, but it is easiest to refer to the pseudo-code
    # of each algorithm discussed in class or in the Sutton and Barto book.
        
        terminal = False
        i = 0
        T = [] # episode
        
        self.state = self.domain.initial_state()
        
        #generate one episode
        while (not terminal) and i < 200:            
            S = self.state
            a = self.sample_epsilon_greedy()
            S_next, R = self.domain.transition(S, a)
            T.append((R,S,a))
                     
            self.state = S_next
            terminal = self.domain.is_terminal(self.state)
            i += 1
        
        #self.G = 0 
        for t in range(len(T)):
            S_t = T[t][1] # T = ((R,S,a))
            A_t = T[t][2] 
            
            # Find the first occurance of the (state, action) pair in the episode
            first_visit = next(i for i,episode in enumerate(T) if episode[1] == S_t and episode[2] == A_t)

            # Sum up all rewards since the first occurance
            G = sum([episode[0]*(self.gamma**i) for i,episode in enumerate(T[first_visit:])])
            
            self.Returns[S_t][A_t].append(G)
            self.Q[S_t][A_t] = np.mean(self.Returns[S_t][A_t])
  
    def train_trial(self):
    # Training typically does not only happen for one episode, but rather for many episodes. 
    # Create a function that first initializes your Q-values using your method initialize values,
    # and then proceeds to train using a predetermined number of episodes using your method train episode 
    # In addition, at the end of each episode, you should test the greedy policy you've obtained 
    # using your method test_greedy. 
    # Keep track of this progress in a list or a numpy array, and
    # return it in this procedure once training completes
    
        self.initialize_values()
        progress = []
        
        for i in range(2000):
            self.train_episode()
            Q_0 = self.test_greedy()
            progress.append(Q_0)
        self.final = True
        
        return progress