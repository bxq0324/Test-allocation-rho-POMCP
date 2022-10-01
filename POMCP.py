from solver import Solver
from helper import rand_choice, randint, round
from belief_tree import BeliefTree
import numpy as np
import time
from helper import ucb

MAX = np.inf

class UtilityFunction():
    @staticmethod
    def ucb1(c):
        def algorithm(action):
            return action.V + c * ucb(action.parent.N, action.N)
        return algorithm
    
    @staticmethod
    def mab_bv1(min_cost, c=1.0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            ucb_value = ucb(action.parent.N, action.N)
            return action.mean_reward / action.mean_cost + c * ((1. + 1. / min_cost) * ucb_value) / (min_cost - ucb_value)
        return algorithm

    @staticmethod
    def sa_ucb(c0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            return action.V + c0 * action.parent.budget * ucb(action.parent.N, action.N)
        return algorithm


class POMCP(Solver):
    def __init__(self, model):
        Solver.__init__(self, model)
        self.tree = None
        self.model=model
        #self.simulation_time = None  # in seconds
        self.max_particles = None   # maximum number of particles can be supplied by hand for a belief node
        self.reinvigorated_particles_ratio = None  # ratio of max_particles to mutate 
        self.utility_fn = None
        self.states=[]
        self.belief=[]
        
    def add_configs(self, budget=float('inf'), initial_belief=None,
                    max_particles=300, reinvigorated_particles_ratio=0.1, utility_fn='ucb1', C=0.5):
        # acquaire utility function to choose the most desirable action to try
        if utility_fn == 'ucb1':
            self.utility_fn = UtilityFunction.ucb1(C)
        elif utility_fn == 'sa_ucb':
            self.utility_fn = UtilityFunction.sa_ucb(C)
        elif utility_fn == 'mab_bv1':
            if self.model.costs is None:
                raise ValueError('Must specify action costs if utility function is MAB_BV1')
            self.utility_fn = UtilityFunction.mab_bv1(min(self.model.costs), C)

        # other configs
        self.max_particles = max_particles
        self.reinvigorated_particles_ratio = reinvigorated_particles_ratio
        
        # initialise belief search tree
        root_particles = self.model.states     
        self.tree = BeliefTree(budget, root_particles)
        self.tree.root.belief=initial_belief

    def compute_belief(self):
        base = [0.0] * self.model.num_states
        for i in range (self.model.num_states):
            base[i] = round(self.model.belief[i], 6)
        return base

    def rollout(self, c_states,c_belief,state, h, depth, max_depth, budget):
        """
        Perform randomized recursive rollout search starting from 'h' util the max depth has been achived
        :param state: starting state's index
        :param h: history sequence
        :param depth: current planning horizon
        :param max_depth: max planning horizon
        :return:
        """
        if depth > max_depth or budget <= 0:
            return 0
        
       
        ai = rand_choice(self.model.legal_actions(c_states))

        #print(ai)
        #print(c_states)
        if ai>state[2]:
            ai=rand_choice(self.model.get_legal_actions(state))
        
        n_states,n_belief,sj, oj, r, cost= self.model.simulate_action(c_states,c_belief,state, ai)
       
        return r + self.model.discount * self.rollout(n_states,n_belief,sj, h + [ai, oj], depth + 1, max_depth, budget-cost)
        
    def simulate(self,c_states,c_belief,state, max_depth, depth=0, h=[], parent=None, budget=None):
        """
        Perform MCTS simulation on a POMCP belief search tree
        :param state: starting state's index
        :return:
        """
        # Stop recursion once we are deep enough in our built tree
        if depth > max_depth:
            return 0
        
        
        if not h:
            obs_h=None
            node_h = self.tree.root
        else:
            obs_h=h[-1]
            #particles,belief=self.model.update_belief(p_states,p_belief,h[-2],obs_h)
            node_h = self.tree.find_or_create(h, name=obs_h , parent=parent,
                                          budget=budget,particle=c_states,observation=obs_h)
            node_h.belief=c_belief
        

        # ===== ROLLOUT =====
        # Initialize child nodes and return an approximate reward for this
        # history by rolling out until max depth
            #log.warning("Warning: {} is not in the search tree".format(root.h + [action, obs]))
            # The step result randomly produced a different observation
            
        if not node_h.children:
            # always reach this line when node_h was just now created
            for ai in self.model.get_legal_actions(state):
                #print(ai)
                #print(len(node_h.children))
                #print(state)
                #print(self.model.get_legal_actions(state))
                #print(ai)
                cost = self.model.cost_function(ai)
                # only adds affordable actions
                if budget - cost >= 0:
                    self.tree.add(h + [ai], name=ai, parent=node_h, action=ai, cost=cost)
                #print(len(node_h.children))
            #print(node_h.children)    
            c_states=node_h.B
            c_belief=node_h.belief
            return self.rollout(c_states,c_belief,state, h, depth, max_depth, budget)
        
        # ===== SELECTION =====
        # Find the action that maximises the utility value
        #np.random.shuffle(node_h.children)
        #print(node_h.children)
        #print(sorted(node_h.children, key=self.utility_fn, reverse=True)[0].action)
        #print(node_h.children)
        np.random.shuffle(node_h.children)
        #print(node_h.children)
        for i in range (len(node_h.children)):
            node_ha = sorted(node_h.children, key=self.utility_fn, reverse=True)[i]
            if node_ha.action in self.model.legal_actions(node_h.B):
                    break
        #print(self.model.legal_actions(node_h.B))
        #print(node_ha.action)
             
               
        
        # ===== SIMULATION =====
        # Perform monte-carlo simulation of the state under the action
        #c_states=node_h.B
        #c_belief=node_h.belief
        #print(self.model.legal_actions(c_states))
        n_states,n_belief,sj, oj, reward, cost = self.model.simulate_action(c_states,c_belief,state, node_ha.action)
        #print(sj)
        #print(oj)
        R = reward + self.model.discount * self.simulate(n_states,n_belief,sj, max_depth, depth + 1, h=h + [node_ha.action, oj],parent=node_ha, budget=budget-cost)
        #print(reward)
        #print(R)
        # ===== BACK-PROPAGATION =====
        # Update the belief node for h

        node_h.N += 1

        # Update the action node for this action
        node_ha.update_stats(cost, reward)
        node_ha.N += 1
        node_ha.V += (R - node_ha.V) / node_ha.N
        

        return R

    def solve(self,T):
        """
        Solves for up to T steps
        """
        #begin = time.time()
        #n = 0
        #while time.time() - begin < self.simulation_time:
        for i in range(70):
            #n += 1
            state = self.tree.root.sample_state()
            #print(state)
            self.simulate(self.tree.root.B,self.tree.root.belief,state, max_depth=T, h=self.tree.root.h, budget=self.tree.root.budget)
        #log.info('# Simulation = {}'.format(n))

    def get_action(self, belief):
        """
        Choose the action maximises V
        'belief' is just a part of the function signature but not actually required here
        """
        root = self.tree.root
        action_vals=[]
        for action in root.children:
            if action.V<0:
                action_vals.append((action.V,action.action))
        #print(action_vals)
        return  max(action_vals)[1]

    def update_belief(self, belief, action, obs):
        """
        Updates the belief tree given the environment feedback.
        extending the history, updating particle sets, etc
        """
        m, root = self.model, self.tree.root

        #####################
        # Find the new root #
        #####################
        new_root = root.get_child(action).get_child(obs)
        if new_root is None:
            #log.warning("Warning: {} is not in the search tree".format(root.h + [action, obs]))
            # The step result randomly produced a different observation
            action_node = root.get_child(action)
            #if action_node.children:
                # grab any of the beliefs extending from the belief node's action node (i.e, the nearest belief node)
                #log.info('grabing a bearest belief node...')
                #new_root = rand_choice(action_node.children)
           # else:
                # or create the new belief node and rollout from there
                #log.info('creating a new belief node')
                #self.states.append(particles)
            n_states,n_belief=self.model.update_belief(root.B,root.belief, action,obs)
            new_root = self.tree.add(h=action_node.h + [obs], name=obs, parent=action_node, observation=obs,
                                         particle=n_states, budget=root.budget - action_node.cost)
                
            new_root.belief=n_belief
         
        ##################
        # Fill Particles #
        ##################
        
        

        #####################
        # Advance and Prune #
        #####################
        
        new_belief=new_root.belief
        self.tree.prune(root, exclude=new_root)
        self.tree.root = new_root
       
        #print(self.tree.root.B)
        self.states.append(new_root.B)
        self.belief.append(new_belief)

        ###########################
        # Particle Reinvigoration #
        ###########################
        
        return new_belief
    def draw(self, beliefs):
        """
        Dummy
        """
        pass
