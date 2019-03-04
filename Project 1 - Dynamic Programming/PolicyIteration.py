import random

class PolicyIteration :
    
    def __init__(self, gridworld , gamma):
        # initialize the domain and discount factor
        self. gridworld = gridworld
        self. gamma = gamma
        self.k = 0 # policy iteration numbers
        self.V = {}  # value matrix
        self.pi = {}
        self.p = 0
    
    def initialize_policy(self):
        # policy iteration also requires to maintain a policy
        # initialize the policy pi_0 [s] = 0 ( LEFT ) for all states and the value function V_0[s] = 0
        # implement the policy (self.pi) as a dictionary that maps states to actions in {LEFT , UP , RIGHT , DOWN }
        self.pi[self.k] = {key: 'L' for key in self.gridworld.states()}
        
    def initialize_values(self):
        # you can use the corresponding procedure you defined in ValueIteration
        self.V[self.k] = {key: 0.0 for key in self.gridworld.states()} 
    
    def compute_backup(self, state, action):
        # use this as a helper function to return the necessary quantity 
        # E{R(s, a, S ') + gamma * V_k[S ']} in value iteration 
        next_state = self.gridworld.next_state(state, action)
        V_s = self.V[self.k][next_state]
        p = self.gridworld.pr(state,action,next_state) - self.p       # updated for windy
        R_s = self.gridworld.reward(state, action, next_state)
        backup = p*(R_s + self.gamma*V_s)
        
        if self.p > 0:
            next_state_windy = self.gridworld.next_state(state, 'D')
            V_s_windy = self.V[self.k][next_state]                             #updated for windy
            R_s_windy = self.gridworld.reward(state, 'D', next_state_windy)
            backup = backup + (1-p)*(R_s_windy + self.gamma*V_s_windy)
        
        return backup 
    
    def greedy_action(self, state ):
        # use the previous function compute_backup to return the (
        # deterministic ) greedy action a* = pi[s] derived from V
        V_temp = {}
        for action in self.gridworld.actions():
            V_temp[action]=self.compute_backup(state, action)
        
        # if more than one action have the max value, select randomly from those top actions
        top = max(V_temp.values())
        topmoves = [k for k, v in V_temp.items() if v == top]
        greedy_a = random.choice(topmoves)
        return greedy_a
    
    def policy_evaluation_step(self, state ):
        # use this function to implement the policy evaluation step for one iteration 
        #, e.g. return the value V_{k +1}[ s] = E[R(s, pi[s], S ') + gamma * V_k[S ']]
        action = self.pi[self.k][state]
        V_new = self.compute_backup(state, action)
        # you should not modify self.V
        return V_new
    
    def policy_evaluation(self, tolerance =1e-6) :
        # in policy evaluation , we solve V = P * (R + gamma * V) for a fixed policy by iterating 
            # V_{k +1}[ s] = E[R(s, pi[s], S ') + gamma * V_k[S ']] for each state s until V_k converges uniformly           
        # use this method to perform the necessary iterations until 
            # V_k[s] (self.V) converges to within the specified tolerance
        # this procedure should not return a value , but should modify self.V
        V_old = self.V[self.k]
        while True:
            delta = 0
            V_new = {}
            for state in self.gridworld.states():
                V_new[state] = self.policy_evaluation_step(state)
                delta = max(delta, abs(V_new[state] - V_old[state]))        
            if delta < tolerance:
                break
            else:
                V_old = V_new
        self.V[self.k+1] = V_new
        
    def policy_improvement(self):
        # implement the procedure here for policy improvement , 
        # e.g. pi_{k +1}[ s] = argmax_a E[R(s, a, S ') + gamma * V_k[S']]
        pi_new = {}
        improvement = False    #the opposite of "stable" is "improvement", which is more intuitive given the function name
        
        for state in self.gridworld.states():
            pi_new[state] = self.greedy_action(state)
            if pi_new[state] != self.pi[self.k][state]:
                improvement = True
        self.pi[self.k + 1] = pi_new
        return improvement
        
    
    def policy_iteration(self, tolerance =1e-6):
        # implement the complete policy iteration algorithm using the procedures you defined earlier
        # your method should include: 
        
        # initialization of the policy and value function 
        self.initialize_policy()
        self.initialize_values()
        
        # perform each iteration of policy iteration 
            # which includes policy evaluation followed by improvement, until the policy no longer improves
        improving = True
        while improving: 
            self.policy_evaluation()
            improving = self.policy_improvement()
            self.k = self.k + 1
        
        # you should perform monitoring and plotting the value function at the initial state , 
            # like you did for value iteration  
            
    def set_p(self,p):
        self.p = p
        
    def get_V(self):
        return self.V # value grid
    
    def get_k(self):
        return self.k # number of iterations
    
    def get_pi(self):
        finalIter = self.k - 1
        return self.pi[finalIter] # best policy