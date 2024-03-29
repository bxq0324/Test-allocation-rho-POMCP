
import numpy as np
from scipy.stats import poisson
from scipy.stats import dirichlet
from scipy.stats import beta
from scipy.stats import binom


class SAIRS(object):
   
    
    def __init__(self, states,belief):
        
        self.states=states
        self.belief=belief
        idx=np.random.choice(np.linspace(0,len(states)-1,len(states)),p=self.belief)
        self.curr_state = self.states[int(idx)]
        self.discount=0.95
        
    @property
    def num_states(self):
        return len(self.states)

    @property
    def num_actions(self):
        return len(self.actions)

    def get_g(self,state):
        p_fp=0.05
        p_fn=0.1
        pd_a=0.8
        pd_i=0.8
        g=p_fp+(1-p_fp-p_fn)*(pd_a*state[0][1]+pd_i*state[0][2])
        return g
    
    def update_belief(self,c_states,c_belief, action,obs):     #update belief by rejective sampling
        n=len(self.states)
        new_states=[]
        while len(new_states)<n:
            idx=np.random.choice(np.linspace(0,len(self.states)-1,len(self.states)),p=c_belief)
            s_state=c_states[int(idx)]
            n_state=self.next_state(s_state,action,obs)
            #if n_state[2]>=0:
            if self.gen_observation(n_state,action)[0]==obs[0]:
                    new_states.append(n_state)
        #print(new_states)
        bel=np.ones(len(self.states))
        for i in range(len(self.states)):
            #print(self.get_g(new_states[i]))
            bel[i]=self.observation_function(action, new_states[i], obs)
            #print(bel[i])
        new_belief=bel/np.sum(bel)
        #print(new_belief)
        return new_states,new_belief
       
    def legal_actions(self,states):
        min_p=states[0][2]
        for i in range(len(states)):
            if min_p>=states[i][2]:
                min_p=states[i][2]
        return np.linspace(0,min_p,min_p+1)
            
            
    
    
    def get_legal_actions(self, state):
        return np.linspace(0,state[2],state[2]+1)
    
    
    def next_state(self,si,action,obs):
       
        Si,Ai,Ii,Ri=si[0]
        qi,betai,sigmai,gammai,kappai,deltai=si[1]
        pi=si[2]
        Sj=Si-betai*Si*(Ai+Ii)+deltai*Ri
        Aj=Ai+qi*betai*Si*(Ai+Ii)-(kappai+sigmai)*Ai
        Ij=Ii+(1-qi)*betai*Si*(Ai+Ii)+sigmai*Ai-gammai*Ii
        Rj=Ri+kappai*Ai+gammai*Ii-deltai*Ri
        nsta=[]
        nsta.append(np.array([Sj,Aj,Ij,Rj]))
        nsta.append(si[1])
        nsta.append(si[2]+obs[1]-action)
        if action in self.get_legal_actions(si):
            return nsta
        else:
            #print(self.get_legal_actions(si))
            print("error")
        
    def gen_observation(self,state,action):
        obs=np.ones(2)
        p_test=np.random.binomial(action,self.get_g(state))
        n_arrival=np.random.poisson(3)
        obs[0]=int(p_test)
        obs[1]=int(n_arrival)
        return obs
                         

    def observation_function(self, action, state, obs):
        return binom.pmf(obs[0],action,self.get_g(state))*poisson.pmf(obs[1],3)
       
       

    def transition_function(self, action, si, sj,obs):
        Sj,Aj,Ij,Rj=sj[0]
        Si,Ai,Ii,Ri=si[0]
        qj,betaj,sigmaj,gammaj,kappaj,deltaj=sj[1]
        qi,betai,sigmai,gammai,kappai,deltai=si[1]
        pj=sj[2]
        pi=si[2]
        ds=-betai*Si*(Ai+Ii)+deltai*Ri
        da=qi*betai*Si*(Ai+Ii)-(kappai+sigmai)*Ai
        di=(1-qi)*betai*Si*(Ai+Ii)+sigmai*Ai-gammai*Ii
        dr=kappai*Ai+gammai*Ii-deltai*Ri      
        if (Sj-Si==ds)&(Aj-Ai==da)&(Ij-Ii==di)&(Rj-Ri==dr)&(sj[1]==si[1]).all()&(pj==pi+obs[1]-action):
            prob=1
        else:
            prob=0  
        return prob

    def reward_function(self,blief):
        n=len(blief)
        n_ent=0
        for i in range(len(self.states)):
            n_ent=n_ent+blief[i]*np.log(blief[i])
        return n_ent
 

    def cost_function(self, action):
        return 0
           
            
    def simulate_action(self,c_states,c_belief, si, ai, debug=False):
        """
        Query the resultant new state, observation and rewards, if action ai is taken from state si

        si: current state
        ai: action taken at the current state
        return: next state, observation and reward
        """
        # get new observation
        obs=self.gen_observation(si,ai)
        #print(obs)
        # get new state
        state = self.next_state(si,ai,obs)
        
        #print(state)

        if debug:
            print('taking action {} at state {}'.format(ai ,si))
            print('transition probs: {}'.format(s_probs))
            print('obs probs: {}'.format(o_probs))

        # get new reward
        # reward = self.reward_function(ai, si, sj, observation) #  --- THIS IS MORE GENERAL!
       
        n_states,n_belief=self.update_belief(c_states,c_belief,ai,obs)
        #print(self.belief)
        reward = self.reward_function(n_belief)
        #print(reward)
        cost = self.cost_function(ai)
        
             # --- THIS IS TMP SOLUTION!
        #cost = self.cost_function(ai)
        return n_states,n_belief,state, obs, reward,cost

    def take_action(self,action):
        """
        Accepts an action and changes the underlying environment state
        
        action: action to take
        return: next state, observation and reward
        """
        #print(self.states)
        #print(self.curr_state)
        n_states,n_belief,state, observation, reward,cost= self.simulate_action(self.states,self.belief,self.curr_state,action)
        self.states=n_states
        self.belief=n_belief
        self.curr_state = state
        #print(self.curr_state)
        return state, observation, reward,cost

  