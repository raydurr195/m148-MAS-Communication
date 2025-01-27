import numpy as np
import gym
from gym import spaces
import ray.rllib

class Werewolf_1(gym.Env):
    def __init__(self, num_agents = 5, comm_rounds = 4):
        self.num_agents = num_agents
        self.roles = ['werewolf', 'villager']
        self.state = {}

        self.comm_max = comm_rounds
        self.comm_round = 0
        self.phase = 1 #stage 1 is comm, 2 is vote, 3 is werewolf eliminate
        self.comm_max = comm_rounds #there is a maximum number of communication rounds equal to comm_rounds for phase 1
        self.comm_round = 0 #start with a communication round of 0

        self.action_space = spaces.Dict( {
                'comm' : spaces.Discrete(4), #0 is lie, 1 is accuse, 2 is tell truth, 3 is defend
                'vote' : spaces.Discrete(self.num_agents)
        }) #apparently this action space is inefficient since in stage 1 the vote portion of the space is unused and vis versa for stages 2 and 3
        
        self.observation_space = spaces.Dict( {
            'role': spaces.Discrete(2), #0 is villager, 1 is werewolf
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            #public_x is an nxn matrix representing the number of accusations/votes/defenses 
            #agent i has levied at agent j
            'trust': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),
            'known_werewolves': spaces.MultiBinary(self.num_agents), #this will be used so werewolves will know who other werewolves are
            'life_status': spaces.MultiBinary(self.num_agents) #1 means alive, 0 is dead
        })
        

