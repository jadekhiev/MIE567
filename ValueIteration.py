import random
class ValueIteration(object):

    def __init__(self, gridworld, gamma):
        # initialize the domain and discount factor
        self.gridworld = gridworld
        self.gamma = gamma
        self.k = 0 # iteration numbers
        self.V = {}  # value matrix
        self.pi = {}
        self.p = 0
        
    def initialize_values(self):
        # you need to initialize the value function V_0[s] = 0 for all states
        # implement V as a dictionary that maps states to values - this will guarantee constant - time reading and writing
        # initialize states with 0s
        self.V[self.k] = {key: 0.0 for key in self.gridworld.states()} 
        self.pi[self.k] = {key: 0 for key in self.gridworld.states()}
        
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

    def greedy_action(self, state):
        # use the previous function compute_backup to return the ( deterministic ) greedy policy pi[a | s] derived from V
        V_temp = {}
        for action in self.gridworld.actions():
            V_temp[action]=self.compute_backup(state, action)
            
        # if more than one action have the max value, select randomly from those top actions
        top = max(V_temp.values())
        topmoves = [k for k, v in V_temp.items() if v == top]
        greedy_a = random.choice(topmoves)
        return greedy_a 

    def update_values(self):
        # this method updates the value function for one iteration., e.g. V_{k}[s] -> V_{k +1}[ s] over all s
        # first , build V_{k+1} as a dictionary
        # implement the rest of the procedure for computing V_new = V_{k+1}
        V_new = {}
        V_intermediate = {key: 0 for key in self.gridworld.states()}
        
        # at this stage , you should use the function compute_backup and you should only modify V_new - 
        for state in self.gridworld.states():
            V_temp = []
            for action in self.gridworld.actions():
                V_temp.append(self.compute_backup(state, action))
            V_new[state] = (1/4)*sum(V_temp)
        
        self.pi[self.k] = {state: self.greedy_action(state) for state in self.gridworld.states()} # return best action for each state
        # you should read but not modify the values of self.V
        # once V_new is computed , we can update k = k + 1 - do not modify these lines in your code
        self.k = self.k + 1
        # self.V_old , self.V = self.V, V_new
        return V_new        
    
    def value_iteration(self, tolerance = 1e-6) :
        # use this to implement the complete value iteration algorithm - 
        # you should make use of the procedures initialize_values and update_values that you have written above
        # this method does not need to return anything since the final V_k should be in self.V
        # store , and when converged plot , the backed -up value function at the initial state of the MDP 
            # according to the greedy policy derived from V_k as a function of k
        # also , print the final greedy policy in the console
        # stop your iterations when the difference between V_{k+1}(self.V) and V_k (self. V_old ) is less than tolerance
        self.initialize_values() # init v(s) = 0 for all states
        V_new = {}  

        while True:
            delta = 0
            # update V[k]
            self.V[self.k] = self.update_values() # also runs greedy_action
            for state in self.gridworld.states():
                key = self.k - 1
                V_old = self.V[key][state] # V_old
                delta = max(delta, abs(self.V[self.k][state] - V_old)) #V_new - V_old
            if delta < tolerance: 
                break
                
    def set_p(self,p):
        self.p = p
        
    def get_V(self):
        return self.V # value grid
    
    def get_k(self):
        return self.k # number of iterations
    
    def get_pi(self):
        finalIter = self.k - 1
        return self.pi[finalIter] # best policy