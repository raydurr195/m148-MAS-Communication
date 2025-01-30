import numpy as np
import gym
from gym import spaces
import ray.rllib

class Werewolf_1(gym.Env):
    def __init__(self, num_agents = 7, comm_rounds = 4, num_wolf = 1, max_days = 15):
        self.num_agents = num_agents
        self.num_wolf = num_wolf
        self.roles = ['werewolf', 'villager']
        self.wolves = []
        self.agent_id = list(range(self.num_agents))
        self.state = {}

        self.comm_max = comm_rounds
        self.comm_round = 0
        self.max_days = max_days
        self.phase = 1 #stage 1 is comm, 2 is vote, 3 is werewolf eliminate
        self.comm_max = comm_rounds #there is a maximum number of communication rounds equal to comm_rounds for phase 1
        self.comm_round = 0 #start with a communication round of 0

        self.action_space = spaces.MultiDiscrete([#this action space is divided into two different discrete spaces 
                4, #this space goes from 0-3 and is used in all stages
                #in the communication stage: 0 is lie, 1 is accuse, 2 is tell truth, 3 is defend//all other phases any entry will represent a vote action or for seer the watch option
                self.num_agents  #the second is the target and is used in all stages
                ]) 
        self.observation_space = spaces.Dict( {
            'role': spaces.Discrete(2), #0 is villager, 1 is werewolf
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            #public_x is an nxn matrix representing the number of accusations/votes/defenses 
            #agent i has levied at agent j
            'trust': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),
            'known_werewolves': spaces.MultiBinary(self.num_agents), #this will be used so werewolves will know who other werewolves are//might want to delete and just modify role observation
            'life_status': spaces.MultiBinary(self.num_agents), #1 means alive, 0 is dead

            'phase' : spaces.Discrete(3), #shows current phase
            'comm_round' : spaces.Discrete(self.comm_max) #shows current comm round
        })
        
        self.reset() #initializes new starting env
    
    def reset(self):
        wolves = np.random.choice(self.agent_id, size = self.num_wolf, replace = False) #randomly choose num_wolf amount of wolves from the agents(these are index numbers)
        self.wolves = wolves

        self.phase = 0
        self.state = { #this state should be the global state, we will define each individual agent's observation state later
            'role': np.array([1 if i in self.wolves else 0 for i in self.agent_id]),
            'public_accusation':  np.zeros((self.num_agents, self.num_agents), dtype=np.float32),
            'public_vote':  np.zeros((self.num_agents, self.num_agents), dtype=np.float32),
            'public_defense':  np.zeros((self.num_agents, self.num_agents), dtype=np.float32),

            'trust': np.full((self.num_agents, self.num_agents), 0.5),
            'known_werewolves': np.array([1 if i in self.wolves else 0 for i in range(self.num_agents)]),
            'life_status': np.ones(self.num_agents, dtype = np.int32), #1 means alive, 0 is dead

            'phase' : 0,
            'comm_round' : 0
        }

        observations = self.get_obs() #reset function returns observations
        return observations 
    def get_obs(self):
        for agent in self.agent_id:
            if agent in self.wolves:
                role = np.array([])