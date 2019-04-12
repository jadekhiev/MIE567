import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class ValueIteration:
    def __init__(self, domain, gamma):
    # An init method that initializes the domain and the discount factor.
        self.domain = domain
        self.gamma = gamma
        
        self.k = 0 # iteration numbers
        self.U = {}  # value matrix
        self.pi_p = {} # mixed strategy def
        self.pi_q = {} # mixd strategy inv
        self.delta = []
        
    def initialize_values(self):
    # A procedure that initializes the value function(s) V to zero for all states. 
    # Whether you have one value function in total, or one for each agent, 
    # depends on your answer to 1(a). 
    # If you have one, it may be defined for the defender or invader, 
    # but please be consistent in defining all your other variables.
    
    # Initialize U0 with arbitrary values
    
        self.U[self.k] = {key: 0.0 for key in self.domain.states()}
        self.G = {key: 0 for key in self.domain.states()}
    
    def compute_return(self, s, a):
    # A function that takes the current game state s and actions from both agents a1, a2 and 
    # computes the return, e.g. Q(vector_s, (a1, a2)).
    
        #R(s,a) + gamma * sum of [transition (s,a,s')*Un(s')] 
        # there is no reward for being in a terminal state
        
        nextState, reward = self.domain.transition(s,a)
        transition = 1 # because this is a deterministic game, there is no 'wind' or stochasticity
        G_sa = reward + self.gamma*transition*self.U[self.k][nextState]
        
        return G_sa
        
    
    def compute_payoff(self, s):
    # A function that takes the current game state s and computes a payoff matrix, 
    # where the entries are Q(vector_s, (a1, a2)) indexed by a1 and a2.
        actions = self.domain.actions()
        payoff = np.zeros((4,4))
        
        # here, a1 is the invader, and a2 is the defender, meaning that defender actions will go
        # along the rows, and invader is the column player. 
        
        for i in range(4):
            a1 = actions[i]
            for j in range(4):
                a2 = actions[j]
                payoff[j,i] = self.compute_return(s, (a1, a2))
        
        self.payoff = payoff
        return payoff
            
    
    def compute_lp(self, Q):
    # A function that takes an arbitrary payoff matrix Q, formulates the LP for each agent and solves it. 
    # This function should return the optimal objective value(s) of the final solution, 
    # followed by the minimax equilibria for defender and for the invader. 
    # To solve the LP using the simplex method, you should use the function linprog 
    # from the scipy.optimize package. 
    # The LP formulations you obtained in the form of 1(d) will make it easy for you 
    # to define the appropriate arguments to call the linprog function. 
    # Make sure that you read documentation about this function before calling it 
    # (there are some nuances, for example make sure to specify correct bounds on the variables as a 
    # tuple of pairs (x_lb, x_ub) for each variable).
    
        ## Set up solver inputs
        # c = [vector p, v]
        c = [0, 0, 0, 0, -1]

        # A - minimax constraints, v must be the minimum reward
        q_neg = -1*Q       
        v_col = np.ones((4,1))
        A_p2 = np.concatenate((q_neg,v_col),1)
        b = [0, 0, 0, 0]

        # Aeq, vector p = 1
        Aeq = [[1, 1, 1, 1, 0]]
        beq = [[1]]

        # bound tuples
        bounds = ((0,1),(0,1),(0,1),(0,1),(None, None))

        ## Run the solver to solve for defender 
        res_p2 = linprog(c, A_ub=A_p2, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, options={"bland":True,"disp": False})
        
        #Reformulate for invader
        # only difference is that Q is transposed (still negative)
        q_neg = np.transpose(q_neg)
        A_p1 = np.concatenate((q_neg,v_col),1)
        res_p1 = linprog(c, A_ub=A_p1, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, options={"disp": False})
        
        #default vector
        default = [0.25,0.25,0.25,0.25]
        
        # if it didn't work, print Error
        if not res_p1['success']:
            #print(res_p1['message'])
            p =default
        else:
            # extract p vector for invader strategy
            p = res_p1['x'][:4]
        
        if not res_p2['success']:
            #print(res_p2['message'])
            print(Q)
            q = default
        else:
            q = res_p2['x'][:4]

        if res_p1['success']: 
            value = res_p1['fun']*-1
        elif res_p2['success']:
            value = res_p2['fun']*-1
        else:
            value = np.average(Q)

        
        #minimax_defender = q
        #minimax_invader = p
        pi = {'p':p, 'q':q}
            
        return {'value': value, 'pi':pi}
    
    def value_iteration(self, tolerance = 1e-6):
    # A routine that performs one iteration of the value iteration algorithm.
    # This should keep track of the policy, the old value function(s) and the new value function(s).

        # use this to implement the complete value iteration algorithm - 
        # you should make use of the procedures initialize_values and update_values that you have written above
        # this method does not need to return anything since the final V_k should be in self.V store, and when
        # converged plot , the backed -up value function at the initial state of the MDP according to the greedy policy
        # derived from V_k as a function of k also , print the final greedy policy in the console
        # stop your iterations when the difference between V_{k+1}(self.V) and V_k (self. V_old ) is less than tolerance
        
        self.initialize_values() # init U(s)= 0 for all states
        self.k = 0
        pi = {}
        
        i = 1

        while True and self.k < 100:
            delta = 0
            print("iteration: " + str(self.k))
            self.U[self.k+1]={}
            for state in self.domain.states():
                # if game is not in terminal state, compute payoffs
                terminal, winner = self.domain.is_terminal(state)

                #self.G[state] = {action: self.compute_return(state, action) for action in self.domain.actions()}
                self.G[state] = self.compute_payoff(state)
                self.U[self.k+1][state] = self.compute_lp(self.G[state])['value']
                delta = max(delta, abs(self.U[self.k+1][state]-self.U[self.k][state])) #V_new - V_old
            self.delta.append(delta)    
            if delta < tolerance: 
                break
            self.k += 1

        print("defender")
        heatmap("defender", self.U[self.k])
        print("invader")
        heatmap("invader", self.U[self.k])
            
        for state in self.domain.states():
            self.G[state] = self.compute_payoff(state)
            pi = self.compute_lp(self.G[state])['pi']
            self.pi_p[state] = pi['p']
            self.pi_q[state] = pi['q']
        return self.U[self.k], self.pi_p, self.pi_q, self.delta