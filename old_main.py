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
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.state = {}

        self.comm_max = comm_rounds
        self.comm_round = 0
        self.max_days = max_days
        self.phase = 1 #stage 1 is comm, 2 is vote, 3 is werewolf eliminate
        self.comm_max = comm_rounds #there is a maximum number of communication rounds equal to comm_rounds for phase 1
        self.comm_round = 0 #start with a communication round of 0

        self.action_space = spaces.MultiDiscrete([#this action space is divided into two different discrete spaces 
                4, #this space goes from 0-3 and is used in the communication stage: 0 is lie, 1 is accuse, 2 is tell truth, 3 is defend
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
            'known_werewolves': spaces.MultiBinary(self.num_agents), #this will be used so werewolves will know who other werewolves are
            'life_status': spaces.MultiBinary(self.num_agents), #1 means alive, 0 is dead

            'phase' : spaces.Discrete(3), #shows current phase
            'comm_round' : spaces.Discrete(self.comm_max) #shows current comm round
        })
        
        self.reset() #initializes new starting env
    
    def reset(self):
        wolves = np.random.choice(range(self.num_agents), size = self.num_wolf, replace = False) #randomly choose num_wolf amount of wolves from the agents(these are index numbers)
        self.wolves = wolves

        self.phase = 1
        self.state = { #this state should be the global state, we will define each individual agent's observation state later
            'role': np.array([1 if i in self.wolves else 0 for i in range(self.num_agents)]),
            'public_accusation':  np.zeros((self.num_agents, self.num_agents), dtype=np.float32),
            'public_vote':  np.zeros((self.num_agents, self.num_agents), dtype=np.float32),
            'public_defense':  np.zeros((self.num_agents, self.num_agents), dtype=np.float32),

            'trust': np.full((self.num_agents, self.num_agents), 0.5),
            'known_werewolves': np.array([1 if i in self.wolves else 0 for i in range(self.num_agents)]),
            'life_status': np.ones(self.num_agents, dtype = np.int32), #1 means alive, 0 is dead

            'phase' : 0,
            'comm_round' : 0

        }
        
    def step(self, action):
        agent_id, target_id = action
        reward = 0
        done = False #game is not done

        if self.state['phase'] == 1:  # Communication phase
            self.state['comm_round'] += 1
            if self.state['comm_round'] >= self.comm_max:
                self.state['phase'] = 2
                self.state['comm_round'] = 0

        elif self.state['phase'] == 2:  # Voting phase
            # Initialize vote count array
            votes = np.zeros(self.num_agents, dtype=int)
            
            # Simulate voting
            for agent in range(self.num_agents):
                if self.state['life_status'][agent] == 1:  # Only alive agents can vote
                    vote = np.random.choice(self.num_agents)  # Randomly vote for an agent
                    votes[vote] += 1
            
            # Determine the agent with the most votes
            eliminated_agent = np.argmax(votes)
            
            # Update the state to mark the eliminated agent as dead
            self.state['life_status'][eliminated_agent] = 0
            
            # Move to the next phase
            self.state['phase'] = 3
        elif self.state['phase'] == 3:  # Werewolf elimination phase
            # Implement werewolf elimination logic here
            self.state['phase'] = 1

        # Check if the game is done
        if np.sum(self.state['alive_agents']) <= 1 or self.state['phase'] > self.max_days:
            done = True

        return self.state, reward, done, {}
    def get_obs(self):
        for agent in self.agent_id:
            if agent in self.wolves:
                role = np.array([])