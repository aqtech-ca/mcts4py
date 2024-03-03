from __future__ import annotations
from enum import Enum
from mcts4py.MDP import *
import random
import numpy as np

#At t=0,S=80.For each time interval, the stock price either go up by 25% or decrease by 25% with equal probablity. 
# For American call option, it should never be exercised early. The expected reward for excise is always less or equals to the holding value.


class AmMDP(MDP):

    def __init__(self,K,S,T,r):

        self.K = K #strike
        self.S0=S  #original price
        self.S=S #price dynamic
        self.T=T #Time 
        self.r=r  #interest rate
        self.m=2  # number of discrete time 
        self.time=0 #time dynamic
        self.dt=self.T/self.m #time interval 

    def initial_state(self):
        print('initial')
        self.S=self.S0 # initiate price as original price 
        self.time=0   #initiate time as 0 
        return "0"   #state as 0

    def is_terminal(self, state) -> bool:
        print('state=',state)
        print('is terminal?', state == "1")
        if (state == "1"):  
            return True
        return False

    def reward(self, previous_state, action, state):
        print('reward',self.is_terminal(state))
        if self.is_terminal(state):
            print('reward value',max(np.exp(-self.r*self.time)*(self.S-self.K),0))
            return max(np.exp(-self.r*self.time)*(self.S-self.K),0)
        else:
            return 0

    def transition(self, state, action):
        self.time+=self.dt
        a=np.random.uniform(0,1)
        print('time=',self.time)
        print('price=',self.S)
        print('a=',a)
        self.S=self.S*(a>=0.5)*1.25+self.S*(a<0.5)*0.75
        print('price_new=',self.S)
        return action

    def actions(self,state):
        #0: keep
        #1: exercise
        if self.time>=self.T:
            return ["1"]
        else:
            return ["0","1"]