import gymnasium
from gymnasium import spaces
import numpy as np

from pettingzoo import ParallelEnv

class werewolf():
    metadata = {'name' : 'werewolf_v1'}

    def __init__(self, num_agents = 7, comm_rounds = 4, num_wolf = 1, max_days = 15):
        self.render_mode = None
        self.num_agents = num_agents
        self.num_wolf = num_wolf
        self.roles = ['werewolf', 'villager']

        self.comm_max = comm_rounds
        self.max_days = max_days
        self.comm_max = comm_rounds #there is a maximum number of communication rounds equal to comm_rounds for phase 0

        self.possible_agents = [f'player_{i}' for i in range(self.num_agents)]
    def action_space(self, agent):
        return spaces.MultiDiscrete([#this action space is divided into two different discrete spaces 
                4, #this space goes from 0-3 and is used in all stages
                #in the communication stage: 0 is lie, 1 is accuse, 2 is tell truth, 3 is defend//all other phases any entry will represent a vote action or for seer the watch option
                self.num_agents  #the second is the target and is used in all stages
                ]) 
    def observation_space(self, agent):
        return spaces.Dict( {
            'role': spaces.MultiDiscrete([2 for i in range(self.num_agents)]), #0 is villager, 1 is werewolf
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            #public_x is an nxn matrix representing the number of accusations/votes/defenses 
            #agent i(row) has levied at agent j(column)
            'trust': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),
            'life_status': spaces.MultiBinary(self.num_agents), #1 means alive, 0 is dead

            'phase' : spaces.Discrete(3), #shows current phase
            'comm_round' : spaces.Discrete(self.comm_max), #shows current comm round
            'day' : spaces.Discrete(self.max_days)
        })
    
    def reset(self):
        #initializes a new enviornment
        self.agents = self.possible_agents[:]
        wolves = np.random.choice(self.agent_id, size = self.num_wolf, replace = False) #randomly choose num_wolf amount of wolves from the agents(these are index numbers)
        self.wolves = wolves

        self.phase = 0 #stage 0 is comm, 1 is vote, 2 is werewolf eliminate
        self.comm_round = 0 #start with a communication round of 0
        self.day = 0
        
        infos = {agent: {} for agent in self.agents}
        self.state = self.get_obs_res() #self.state should be a dictionary where each key is the name of the agent(from self.agents) and the value is the observation

    def get_obs_res(self):
        observations = {}
        for agent in self.agents:
            role = np.zeros((self.num_agents,))
            if agent in self.wolves:
                role[self.wolves] = 1

            obs_n = {
            'role': role,
            'public_accusation': np.zeros((self.num_agents,self.num_agents), dtype=np.float32),
            'public_vote': np.zeros((self.num_agents,self.num_agents), dtype=np.float32),
            'public_defense': np.zeros((self.num_agents,self.num_agents), dtype=np.float32),
            'trust': np.full((self.num_agents,), 0.5, dtype=np.float32),
            'life_status': np.ones((self.num_agents,)),

            'phase' : np.array(0),
            'comm_round' : np.array(0),
            'day' : np.array(0)
        }
            observations.update({agent : obs_n})
        return observations
    def update_matrices(self, actions):
        if self.phase == 0:
            #get old accusation and defense matrices for efficient modifications
            new_acc = self.state[self.agents[0]]['public_accusation']
            new_defense = self.state[self.agents[0]]['public_defense']
            #loop through each agent and action
            for agent, action in actions.items():
                comm_type = action[0] #recall 0 is lie, 1 is accuse, 2 is tell truth, and 3 is defend
                target = action[1] #this should be who an action targets

                agent_id = agent.split('_')[1] #doing this so as agents are terminated and removed from self.agents we keep a good track of agent_ids
                self.agents.index(agent)
                if comm_type == 1: #update accusation matrix 
                    new_acc[agent_id, target] = new_acc[agent_id, target] + 1
                elif comm_type == 3:
                    new_defense[agent_id, target] = new_acc[agent_id, target] + 1
            for agent in self.agents:
                self.state[agent]['public_accusation'] = new_acc
                self.state[agent]['public)defense'] = new_defense