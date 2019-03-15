class Flags:
    def initial_state(self):
    # return the initial state of this MDP
        initial_state = (0,5,1)
        return initial_state

    def get_all_states(self):
    # return a list containing all the possible states in this MDP
        states = [
                (0,1,1),    (1,1,1),    (2,1,1),    (3,1,1),    (4,1,1),
                (0,1,2),    (1,1,2),    (2,1,2),    (3,1,2),    (4,1,2),
                (0,1,3),    (1,1,3),    (2,1,3),    (3,1,3),    (4,1,3),
                (0,1,4),    (1,1,4),    (2,1,4),    (3,1,4),    (4,1,4),
                (0,1,5),    (1,1,5),    (2,1,5),    (3,1,5),
                (0,2,1),    (1,2,1),    (2,2,1),    (3,2,1),    (4,2,1),
                (0,2,2),                (2,2,2),    (3,2,2),    (4,2,2),
                (0,2,3),    (1,2,3),    (2,2,3),    (3,2,3),    (4,2,3),
                (0,2,4),    (1,2,4),    (2,2,4),    (3,2,4),    (4,2,4),
                (0,2,5),    (1,2,5),    (2,2,5),    (3,2,5),    (4,2,5),
                (0,3,1),    (1,3,1),    (2,3,1),    (3,3,1),    (4,3,1),
                (0,3,2),    (1,3,2),    (2,3,2),    (3,3,2),    (4,3,2),
                (0,3,3),    (1,3,3),    (2,3,3),    (3,3,3),    (4,3,3),
                (0,3,4),    (1,3,4),    (2,3,4),    (3,3,4),    (4,3,4),
                (0,3,5),    (1,3,5),    (2,3,5),    (3,3,5),    (4,3,5),
                (0,4,1),    (1,4,1),                (3,4,1),    (4,4,1),
                (0,4,2),    (1,4,2),    (2,4,2),    (3,4,2),    (4,4,2),
                (0,4,3),    (1,4,3),    (2,4,3),    (3,4,3),    (4,4,3),
                (0,4,4),    (1,4,4),    (2,4,4),    (3,4,4),    (4,4,4),
                            (1,4,5),    (2,4,5),    (3,4,5),    (4,4,5),
                (0,5,1),    (1,5,1),    (2,5,1),    (3,5,1),    (4,5,1),
                (0,5,2),    (1,5,2),    (2,5,2),    (3,5,2),    (4,5,2),
                (0,5,3),    (1,5,3),    (2,5,3),    (3,5,3),    (4,5,3),
                (0,5,4),    (1,5,4),    (2,5,4),    (3,5,4),    (4,5,4),
                (0,5,5),    (1,5,5),    (2,5,5),                (4,5,5),

                (5,1,5) # TERMINAL STATE
                ]
        return states
    
    def actions(self):
    # return possible actions
        actions = ['U','D','L','R']
        return actions
    
    def next_state(self, state, action):
        stateSpace ={ # ( number of flags captured, row index, column index )
                    (0,1,1):{'U':(0,1,1),'D':(0,2,1),'L':(0,1,1),'R':(0,1,2)},
                    (0,1,2):{'U':(0,1,2),'D':(0,2,2),'L':(0,1,1),'R':(0,1,3)},
                    (0,1,3):{'U':(0,1,3),'D':(0,2,3),'L':(0,1,2),'R':(0,1,4)},
                    (0,1,4):{'U':(0,1,4),'D':(0,2,4),'L':(0,1,3),'R':(0,1,5)},
                    (0,1,5):{'U':(0,1,5),'D':(0,2,5),'L':(0,1,4),'R':(0,1,5)},
                    (0,2,1):{'U':(0,1,1),'D':(0,3,1),'L':(0,2,1),'R':(0,2,2)},
                    (0,2,2):{'U':(0,1,2),'D':(0,3,2),'L':(0,2,1),'R':(0,2,3)},
                    (0,2,3):{'U':(0,1,3),'D':(0,3,3),'L':(0,2,2),'R':(0,2,4)},
                    (0,2,4):{'U':(0,1,4),'D':(0,3,4),'L':(0,2,3),'R':(0,2,5)},
                    (0,2,5):{'U':(0,1,5),'D':(0,3,5),'L':(0,2,4),'R':(0,2,5)},
                    (0,3,1):{'U':(0,2,1),'D':(0,4,1),'L':(0,3,1),'R':(0,3,2)},
                    (0,3,2):{'U':(0,2,2),'D':(0,4,2),'L':(0,3,1),'R':(0,3,3)},
                    (0,3,3):{'U':(0,2,3),'D':(0,4,3),'L':(0,3,2),'R':(0,3,4)},
                    (0,3,4):{'U':(0,2,4),'D':(0,4,4),'L':(0,3,3),'R':(0,3,5)},
                    (0,3,5):{'U':(0,2,5),'D':(1,4,5),'L':(0,3,4),'R':(0,3,5)}, # flag captured {'D'}
                    (0,4,1):{'U':(0,3,1),'D':(0,5,1),'L':(0,4,1),'R':(0,4,2)},
                    (0,4,2):{'U':(0,3,2),'D':(0,5,2),'L':(0,4,1),'R':(0,4,3)},
                    (0,4,3):{'U':(0,3,3),'D':(0,5,3),'L':(0,4,2),'R':(0,4,4)},
                    (0,4,4):{'U':(0,3,4),'D':(0,5,4),'L':(0,4,3),'R':(1,4,5)}, # flag captured {'R'}
                    (0,5,1):{'U':(0,4,1),'D':(0,5,1),'L':(0,5,1),'R':(0,5,2)},
                    (0,5,2):{'U':(0,4,2),'D':(0,5,2),'L':(0,5,1),'R':(0,5,3)},
                    (0,5,3):{'U':(0,4,3),'D':(0,5,3),'L':(0,5,2),'R':(0,5,4)},
                    (0,5,4):{'U':(0,4,4),'D':(0,5,4),'L':(0,5,3),'R':(0,5,5)},
                    (0,5,5):{'U':(1,4,5),'D':(0,5,5),'L':(0,5,4),'R':(0,5,5)}, # flag captured {'U'}
                    
                    (1,1,1):{'U':(1,1,1),'D':(1,2,1),'L':(1,1,1),'R':(1,1,2)},
                    (1,1,2):{'U':(1,1,2),'D':(2,2,2),'L':(1,1,1),'R':(1,1,3)}, # flag captured {'D'}
                    (1,1,3):{'U':(1,1,3),'D':(1,2,3),'L':(1,1,2),'R':(1,1,4)},
                    (1,1,4):{'U':(1,1,4),'D':(1,2,4),'L':(1,1,3),'R':(1,1,5)},
                    (1,1,5):{'U':(1,1,5),'D':(1,2,5),'L':(1,1,4),'R':(1,1,5)},
                    (1,2,1):{'U':(1,1,1),'D':(1,3,1),'L':(1,2,1),'R':(2,2,2)}, # flag captured {'R'}
                    (1,2,3):{'U':(1,1,3),'D':(1,3,3),'L':(2,2,2),'R':(1,2,4)}, # flag captured {'L'}
                    (1,2,4):{'U':(1,1,4),'D':(1,3,4),'L':(1,2,3),'R':(1,2,5)},
                    (1,2,5):{'U':(1,1,5),'D':(1,3,5),'L':(1,2,4),'R':(1,2,5)},
                    (1,3,1):{'U':(1,2,1),'D':(1,4,1),'L':(1,3,1),'R':(1,3,2)},
                    (1,3,2):{'U':(2,2,2),'D':(1,4,2),'L':(1,3,1),'R':(1,3,3)}, # flag captured {'U'}
                    (1,3,3):{'U':(1,2,3),'D':(1,4,3),'L':(1,3,2),'R':(1,3,4)},
                    (1,3,4):{'U':(1,2,4),'D':(1,4,4),'L':(1,3,3),'R':(1,3,5)},
                    (1,3,5):{'U':(1,2,5),'D':(1,4,5),'L':(1,3,4),'R':(1,3,5)},
                    (1,4,1):{'U':(1,3,1),'D':(1,5,1),'L':(1,4,1),'R':(1,4,2)},
                    (1,4,2):{'U':(1,3,2),'D':(1,5,2),'L':(1,4,1),'R':(1,4,3)},
                    (1,4,3):{'U':(1,3,3),'D':(1,5,3),'L':(1,4,2),'R':(1,4,4)},
                    (1,4,4):{'U':(1,3,4),'D':(1,5,4),'L':(1,4,3),'R':(1,4,5)},
                    (1,4,5):{'U':(1,3,5),'D':(1,5,5),'L':(1,4,4),'R':(1,4,5)},
                    (1,5,1):{'U':(1,4,1),'D':(1,5,1),'L':(1,5,1),'R':(1,5,2)},
                    (1,5,2):{'U':(1,4,2),'D':(1,5,2),'L':(1,5,1),'R':(1,5,3)},
                    (1,5,3):{'U':(1,4,3),'D':(1,5,3),'L':(1,5,2),'R':(1,5,4)},
                    (1,5,4):{'U':(1,4,4),'D':(1,5,4),'L':(1,5,3),'R':(1,5,5)},
                    (1,5,5):{'U':(1,4,5),'D':(1,5,5),'L':(1,5,4),'R':(1,5,5)},
                    
                    (2,1,1):{'U':(2,1,1),'D':(2,2,1),'L':(2,1,1),'R':(2,1,2)},
                    (2,1,2):{'U':(2,1,2),'D':(2,2,2),'L':(2,1,1),'R':(2,1,3)},
                    (2,1,3):{'U':(2,1,3),'D':(2,2,3),'L':(2,1,2),'R':(2,1,4)},
                    (2,1,4):{'U':(2,1,4),'D':(2,2,4),'L':(2,1,3),'R':(2,1,5)},
                    (2,1,5):{'U':(2,1,5),'D':(2,2,5),'L':(2,1,4),'R':(2,1,5)},
                    (2,2,1):{'U':(2,1,1),'D':(2,3,1),'L':(2,2,1),'R':(2,2,2)},
                    (2,2,2):{'U':(2,1,2),'D':(2,3,2),'L':(2,2,1),'R':(2,2,3)},
                    (2,2,3):{'U':(2,1,3),'D':(2,3,3),'L':(2,2,2),'R':(2,2,4)},
                    (2,2,4):{'U':(2,1,4),'D':(2,3,4),'L':(2,2,3),'R':(2,2,5)},
                    (2,2,5):{'U':(2,1,5),'D':(2,3,5),'L':(2,2,4),'R':(2,2,5)},
                    (2,3,1):{'U':(2,2,1),'D':(3,4,1),'L':(2,3,1),'R':(2,3,2)}, 
                    (2,3,2):{'U':(2,2,2),'D':(2,4,2),'L':(2,3,1),'R':(2,3,3)}, # flag captured {'D'}
                    (2,3,3):{'U':(2,2,3),'D':(2,4,3),'L':(2,3,2),'R':(2,3,4)},
                    (2,3,4):{'U':(2,2,4),'D':(2,4,4),'L':(2,3,3),'R':(2,3,5)},
                    (2,3,5):{'U':(2,2,5),'D':(2,4,5),'L':(2,3,4),'R':(2,3,5)},
                    (2,4,2):{'U':(2,3,2),'D':(2,5,2),'L':(3,4,1),'R':(2,4,3)}, # flag captured {'L'}
                    (2,4,3):{'U':(2,3,3),'D':(2,5,3),'L':(2,4,2),'R':(2,4,4)},
                    (2,4,4):{'U':(2,3,4),'D':(2,5,4),'L':(2,4,3),'R':(2,4,5)},
                    (2,4,5):{'U':(2,3,5),'D':(2,5,5),'L':(2,4,4),'R':(2,4,5)},
                    (2,5,1):{'U':(3,4,1),'D':(2,5,1),'L':(2,5,1),'R':(2,5,2)}, # flag captured {'U'}
                    (2,5,2):{'U':(2,4,2),'D':(2,5,2),'L':(2,5,1),'R':(2,5,3)},
                    (2,5,3):{'U':(2,4,3),'D':(2,5,3),'L':(2,5,2),'R':(2,5,4)},
                    (2,5,4):{'U':(2,4,4),'D':(2,5,4),'L':(2,5,3),'R':(2,5,5)},
                    (2,5,5):{'U':(2,4,5),'D':(2,5,5),'L':(2,5,4),'R':(2,5,5)},
                    
                    (3,1,1):{'U':(3,1,1),'D':(3,2,1),'L':(3,1,1),'R':(3,1,2)},
                    (3,1,2):{'U':(3,1,2),'D':(3,2,2),'L':(3,1,1),'R':(3,1,3)},
                    (3,1,3):{'U':(3,1,3),'D':(3,2,3),'L':(3,1,2),'R':(3,1,4)},
                    (3,1,4):{'U':(3,1,4),'D':(3,2,4),'L':(3,1,3),'R':(3,1,5)},
                    (3,1,5):{'U':(3,1,5),'D':(3,2,5),'L':(3,1,4),'R':(3,1,5)},
                    (3,2,1):{'U':(3,1,1),'D':(3,3,1),'L':(3,2,1),'R':(3,2,2)},
                    (3,2,2):{'U':(3,1,2),'D':(3,3,2),'L':(3,2,1),'R':(3,2,3)},
                    (3,2,3):{'U':(3,1,3),'D':(3,3,3),'L':(3,2,2),'R':(3,2,4)},
                    (3,2,4):{'U':(3,1,4),'D':(3,3,4),'L':(3,2,3),'R':(3,2,5)},
                    (3,2,5):{'U':(3,1,5),'D':(3,3,5),'L':(3,2,4),'R':(3,2,5)},
                    (3,3,1):{'U':(3,2,1),'D':(3,4,1),'L':(3,3,1),'R':(3,3,2)},
                    (3,3,2):{'U':(3,2,2),'D':(3,4,2),'L':(3,3,1),'R':(3,3,3)},
                    (3,3,3):{'U':(3,2,3),'D':(3,4,3),'L':(3,3,2),'R':(3,3,4)},
                    (3,3,4):{'U':(3,2,4),'D':(3,4,4),'L':(3,3,3),'R':(3,3,5)},
                    (3,3,5):{'U':(3,2,5),'D':(3,4,5),'L':(3,3,4),'R':(3,3,5)},
                    (3,4,1):{'U':(3,3,1),'D':(3,5,1),'L':(3,4,1),'R':(3,4,2)},
                    (3,4,2):{'U':(3,3,2),'D':(3,5,2),'L':(3,4,1),'R':(3,4,3)},
                    (3,4,3):{'U':(3,3,3),'D':(3,5,3),'L':(3,4,2),'R':(3,4,4)},
                    (3,4,4):{'U':(3,3,4),'D':(3,5,4),'L':(3,4,3),'R':(3,4,5)},
                    (3,4,5):{'U':(3,3,5),'D':(4,5,5),'L':(3,4,4),'R':(3,4,5)}, # flag captured {'D'}
                    (3,5,1):{'U':(3,4,1),'D':(3,5,1),'L':(3,5,1),'R':(3,5,2)},
                    (3,5,2):{'U':(3,4,2),'D':(3,5,2),'L':(3,5,1),'R':(3,5,3)},
                    (3,5,3):{'U':(3,4,3),'D':(3,5,3),'L':(3,5,2),'R':(3,5,4)},
                    (3,5,4):{'U':(3,4,4),'D':(3,5,4),'L':(3,5,3),'R':(4,5,5)}, # flag captured {'R'}
                    
                    (4,1,1):{'U':(4,1,1),'D':(4,2,1),'L':(4,1,1),'R':(4,1,2)},
                    (4,1,2):{'U':(4,1,2),'D':(4,2,2),'L':(4,1,1),'R':(4,1,3)},
                    (4,1,3):{'U':(4,1,3),'D':(4,2,3),'L':(4,1,2),'R':(4,1,4)},
                    (4,1,4):{'U':(4,1,4),'D':(4,2,4),'L':(4,1,3),'R':(5,1,5)}, # flag captured {'R'}
                    (4,2,1):{'U':(4,1,1),'D':(4,3,1),'L':(4,2,1),'R':(4,2,2)},
                    (4,2,2):{'U':(4,1,2),'D':(4,3,2),'L':(4,2,1),'R':(4,2,3)},
                    (4,2,3):{'U':(4,1,3),'D':(4,3,3),'L':(4,2,2),'R':(4,2,4)},
                    (4,2,4):{'U':(4,1,4),'D':(4,3,4),'L':(4,2,3),'R':(4,2,5)},
                    (4,2,5):{'U':(5,1,5),'D':(4,3,5),'L':(4,2,4),'R':(4,2,5)}, # flag captured {'U'}
                    (4,3,1):{'U':(4,2,1),'D':(4,4,1),'L':(4,3,1),'R':(4,3,2)},
                    (4,3,2):{'U':(4,2,2),'D':(4,4,2),'L':(4,3,1),'R':(4,3,3)},
                    (4,3,3):{'U':(4,2,3),'D':(4,4,3),'L':(4,3,2),'R':(4,3,4)},
                    (4,3,4):{'U':(4,2,4),'D':(4,4,4),'L':(4,3,3),'R':(4,3,5)},
                    (4,3,5):{'U':(4,2,5),'D':(4,4,5),'L':(4,3,4),'R':(4,3,5)},
                    (4,4,1):{'U':(4,3,1),'D':(4,5,1),'L':(4,4,1),'R':(4,4,2)},
                    (4,4,2):{'U':(4,3,2),'D':(4,5,2),'L':(4,4,1),'R':(4,4,3)},
                    (4,4,3):{'U':(4,3,3),'D':(4,5,3),'L':(4,4,2),'R':(4,4,4)},
                    (4,4,4):{'U':(4,3,4),'D':(4,5,4),'L':(4,4,3),'R':(4,4,5)},
                    (4,4,5):{'U':(4,3,5),'D':(4,5,5),'L':(4,4,4),'R':(4,4,5)},
                    (4,5,1):{'U':(4,4,1),'D':(4,5,1),'L':(4,5,1),'R':(4,5,2)},
                    (4,5,2):{'U':(4,4,2),'D':(4,5,2),'L':(4,5,1),'R':(4,5,3)},
                    (4,5,3):{'U':(4,4,3),'D':(4,5,3),'L':(4,5,2),'R':(4,5,4)},
                    (4,5,4):{'U':(4,4,4),'D':(4,5,4),'L':(4,5,3),'R':(4,5,5)},
                    (4,5,5):{'U':(4,4,5),'D':(4,5,5),'L':(4,5,4),'R':(4,5,5)},
                    (5,1,5):{'U':(5,1,5),'D':(5,1,5),'L':(5,1,5),'R':(5,1,5)}
                    }

        return stateSpace[state][action]
    
    def is_terminal(self, state):
    # this function should return a Boolean indicating
    # whether or not state is a terminal state ; 
    # in other words , does the game end at the specified state
    # or does the robot keep playing ?
    # TERMINAL STATE IS (5,1,5) - returns True if algo should ends o/w returns False
    # use following line in your algo to use this function
    # self.domain.is_terminal(state):
        if state == (5,1,5): return True
        else: return False
    

    def transition(self, state , action):
    # this function should simulate the intended behavior of the Flags domain
    # in particular , given the specified state and action, 
    # this function should return a tuple containing two things :
    # 1. the next state to which we are transitioning to in the next period
    # 2. the reward obtained according to your reward function when transitioning to the next state    

    #        Flag states: 
    #        1. (4,5) 
    #        2. (2,2)
    #        3. (4,1)
    #        4. (5,5)
    #        5. (1,5)
    #        - Assume punishment for visiting any of the flag states prematurely
    #        - Assume no punishment for revisiting a state that previously held a captured flag

        flagStates = [(4,5),(2,2),(4,1),(5,5),(1,5)]
        nextState = self.next_state(state,action)
        
        # state[0] = number of flags 
        if state[0] == nextState[0]:
            
            # if flag state hit prematurely: -10 
            # next spot on grid = (nextState[1], nextState[2]) = (row #, col #)
            # flagStates[state[0]:4] <- slice the list to only include the flags not yet captured
                # flagState = ordered list containing the all the flag states
                # [state[0]:4] = [number of flags captured : end of list] = subset representative of remaining flags
                # if we've captured 4 flags, we no longer have this restriction i.e. no flags left on board except goal
            if state[0] != 4 and (nextState[1],nextState[2]) in flagStates[state[0]:5]: reward = -10
            
            # if no flag captured: -1
            else: reward = -1

        # if flag captured: +25
        else: reward = 25

        return nextState, reward