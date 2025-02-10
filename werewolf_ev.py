import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pettingzoo import ParallelEnv

class werewolf(ParallelEnv):
    metadata = {'name' : 'werewolf_v1'}

    def __init__(self, num_players = 7, comm_rounds = 4, num_wolf = 1, max_days = 15):
        super().__init__()
        self.render_mode = None
        self.num_players = num_players
        self.num_wolf = num_wolf
        self.roles = ['werewolf', 'villager']

        self.max_days = max_days #maximum number of days
        self.comm_max = comm_rounds #there is a maximum number of communication rounds equal to comm_rounds for phase 0
        #we have multiple comm_rounds per day to simulate agents being able to reply to each other
        self.act_space = spaces.MultiDiscrete([#this action space is divided into two different discrete spaces 
                4, #this space goes from 0-3 and is used in all stages
                #in the communication stage: 0 is lie, 1 is accuse, 2 is tell truth, 3 is defend//all other phases any entry will represent a vote action or for seer the watch option
                self.num_players  #the second is the target and is used in all stages
        ])
        self.obs_space = spaces.Dict({
            'role': spaces.MultiDiscrete([3 for i in range(self.num_players)]), #0 is villager, 1 is werewolf
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(self.num_players, self.num_players), dtype=np.float64),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(self.num_players, self.num_players), dtype=np.float64),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(self.num_players, self.num_players), dtype=np.float64),
            #public_x is an nxn matrix representing the number of accusations/votes/defenses 
            #agent i(row) has levied at agent j(column)
            'trust': spaces.Box(low=0, high=1, shape=(self.num_players,), dtype=np.float64),
            'life_status': spaces.MultiBinary(self.num_players), #1 means alive, 0 is dead

            'phase' : spaces.Box(low = 0, high = 2, shape=(1,1)), #shows current phase
            'comm_round' : spaces.Box(low = 0, high = self.comm_max-1, shape=(1,1)), #shows current comm round
            'day' : spaces.Box(low = 0, high = self.max_days-1, shape=(1,1))
        })

        self.observation_spaces = self.obs_space
        self.action_spaces = self.act_space
        self.possible_agents = [f'player_{i}' for i in range(self.num_players)]
    
    def action_space(self, agent):
        return self.act_space
                
    
    def observation_space(self, agent):
        return spaces.flatten_space(self.obs_space)
    
    def reset(self, *, seed = None, options = None):
        #initializes a new enviornment
        self.agents = self.possible_agents[:] #selects agents from possible agents
        wolves = np.random.choice(len(self.agents), size = self.num_wolf, replace = False) #randomly choose num_wolf amount of wolves from the agents(these are index numbers)
        self.wolves = wolves #stores index number of wolves//should remember to delete the entry if wolf is eliminated
        self.phase = 0 #stage 0 is werewolf killing, 1 is communication, 2 is voting
        self.comm_round = 0 #start with a communication round of 0
        self.day = 0 #goes from 0 to self.max_days - 1
        
        infos = {agent: {} for agent in self.agents} #weird thing from petting zoo//not sure why its needed or what it does but documentation shows an empty dict works
        self.state = self.get_obs_res() #self.state should be a dictionary where each key is the name of the agent(from self.agents) and the value is the observation
        obs = self.state
        info = {agent: {} for agent in self.agents}
        return obs, info

    def get_obs_res(self):
        observations = {}
        for agent in self.agents:
            role = np.zeros((self.num_players,), dtype = np.int32)
            if int(agent.split('_')[1]) in self.wolves:
                role[self.wolves] = 1

            obs_n = {
            'role': role,
            'public_accusation': np.zeros((self.num_players,self.num_players), dtype=np.float64),
            'public_vote': np.zeros((self.num_players,self.num_players), dtype=np.float64),
            'public_defense': np.zeros((self.num_players,self.num_players), dtype=np.float64),
            'trust': np.full((self.num_players,), 0.5, dtype=np.float64),
            'life_status': np.ones((self.num_players,)),

            'phase' : np.array(0).reshape((1,1)),
            'comm_round' : np.array(0).reshape((1,1)),
            'day' : np.array(0).reshape((1,1))
        }
            observations.update({agent : spaces.flatten(self.obs_space, obs_n)})
        return observations
    
    # helper function to kill agents
    def update_life_status(self, agent, target, status):
        obs = self.state[agent]
        obs_unflat = spaces.unflatten(self.obs_space, obs)
        new_unflat = obs_unflat['life_status'][target] = status
        self.state[agent][spaces.flatten(self.obs_space, new_unflat)]


    """
    def update_matrices(self, actions):
        if self.phase == 1: #if communication phase then only update public accusations and defense
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
                self.state[agent]['public_defense'] = new_defense
    """
    
    def step(self, actions):
        """
        Update the game state based on actions

        Parameters:
            actions (dict) : dict mapping each agent to their chosen action

        Output:
            observations (dict) : updated game state for each agent
            rewards (dict) : rewards for each agent
            terminations (dict) : whether the game has ended by game rules
            truncations (dict) : whether the game has ended by hitting max days

        TODO: add verbose option??? to print state of game after each day
        TODO: filter so that only living agents can participate
        TODO: use the update_life_status helper function
        """

        # initialize rewards and termination/truncation flags
        # move the below to reset function
        observations = self.state
        observations = {agent : spaces.unflatten(self.obs_space, obs) for agent, obs in observations.items()}
        #actions = {agent : spaces.unflatten(self.act_space, action) for agent,action in actions.items()}
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents} 
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        phase = self.phase
        # actions based on current phase
        # NIGHT
        if phase == 0:
            # Night phase
            # seer sees 
            # TODO: are we allowing multiple seers???
            for agent,action in actions.items():
                agent_id = int(agent.split('_')[1])
                if observations[agent]['role'][agent_id] == 2:  # seer role
                    target = action[1]  # the agent the seer investigates
                    
                    observations[agent]['role'][target]= 1 if observations[self.agents[target]]['role'][target] == 1 else 0

            # werewolves choose a target to kill
            werewolf_actions = [action[1] for agent, action in actions.items() if int(agent.split('_')[1]) in self.wolves]

            target = np.random.choice(werewolf_actions) 
                # ^^ TODO: change this to be a choice by the agent
                # also add a voting thing if there are multiple werewolves


                #should loop through and update life status for every agent
            self.phase = 1
            for agent in self.agents:

                observations[agent]['life_status'][target] = 0
                observations[agent]['phase'] = np.array(self.phase, dtype = np.int32)

            #return observations, rewards, terminations, truncations, {}
            # move to day//I think we should return here, update each agents observations to have a phase equal to 1 and allow them to take more actions


        # DAY: communication phase
        # TODO: what else......
        elif phase == 1 : 
            accusations = np.zeros((self.num_players,self.num_players), dtype=np.float64) #store accusations
            defenses = np.zeros((self.num_players,self.num_players), dtype=np.float64)
            for agent,action in actions.items():
                agent_id = int(agent.split('_')[1])
                # only alive agents communicate
                if observations[agent]['life_status'][agent_id] == 1:  
                    #maybe punish for voting for a dead person/voting when you are dead?
                    if action[0] == 1:#if the agent decides to accuse another agent
                        target = action[1]  # agent they accuse
                        accusations[agent_id,target] +=1
                    elif action[0] == 3: #if the agent decides to defend another agent
                        target = action[1]
                        defenses[agent_id,target] += 1
            self.comm_round += 1
            if self.comm_round >= self.comm_max: # move onto voting phase
                self.phase = 2
            for agent in self.agents:
                #update all agents accusations and defenses matrix
                observations[agent]['public_accusation'] += accusations
                observations[agent]['public_defense'] += defenses
                observations[agent]['comm_round'] = np.array(self.comm_round,dtype = np.int32)
                observations[agent]['phase'] = np.array(self.phase,dtype = np.int32)
            
           # return observations, rewards, terminations, truncations, {}



        elif phase == 2 :

            # first, reset phase and move to next day
            self.phase = 0
            self.comm_round = 0
            self.day += 1
            # Voting phase
            votes = np.zeros((1,self.num_players))
            for agent, action in actions.items():
                if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:
                    # only LIVING agents can vote
                    #punish for voting for a dead player/a dead player voting?
                    target = action[1]
                    votes[0,target] += 1

                # eliminate agent that gets the most votes
                max_votes = np.max(votes)
                targets = np.where(votes== max_votes)[1]
                target = np.random.choice(targets)
                
                for agent in self.agents:
                    observations[agent]['life_status'][target] = 0
                    observations[agent]['comm_round'][0] = np.array(self.comm_round,dtype=np.int32)
                    observations[agent]['phase'][0] = np.array(self.phase,dtype=np.int32)
                    observations[agent]['day'][0] = np.array(self.day,dtype=np.int32)
    

        # Reward agents for surviving another day
        for agent in self.agents:
            if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:
                rewards[agent] += 2
            else:
                rewards[agent] -= 2

        # check for terminations
        num_werewolves = sum(observations[agent]['role'][int(agent.split('_')[1])] for agent in self.agents)
        num_villagers = sum(1 - observations[agent]['role'][int(agent.split('_')[1])] for agent in self.agents)

        if num_werewolves >= num_villagers:
            terminations = {agent: True for agent in self.agents}
            # Assign rewards for winning or losing
            for agent in self.agents:
                if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:  # Check if the agent is alive
                    if observations[agent]['role'][int(agent.split('_')[1])] == 1:
                        rewards[agent] = 100  # Werewolves win
                else:
                    rewards[agent] = -100  # Villagers lose
        elif num_werewolves == 0:
            terminations = {agent: True for agent in self.agents}

            # Assign rewards for winning or losing
            for agent in self.agents:
                if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:  # Check if the agent is alive
                    if observations[agent]['role'][int(agent.split('_')[1])] == 0:
                        rewards[agent] = 100  # Villagers win
                else:
                    rewards[agent] = -100  # Werewolves lose
                    
            # rewards = {agent: 1 if self.state[agent]['role'][int(agent.split('_')[1])] == 1 else -1 for agent in self.agents}

        # check for truncations
        if self.day >= self.max_days:
            truncations = {agent: True for agent in self.agents}
        
        # update observations (???)
        #note that we should be updating observations after each phase
        observations = {agent : spaces.flatten(self.obs_space, obs) for agent, obs in observations.items()}
        return observations, rewards, terminations, truncations, infos